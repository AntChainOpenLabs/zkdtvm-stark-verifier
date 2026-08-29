#[cfg(feature = "koalabear")]
use crate::hash::Poseidon2KoalaBearHasherVariable;
use crate::{
    challenger::{
        CanCopyChallenger, CanObserveVariable, DuplexChallengerVariable, FieldChallengerVariable,
        SpongeChallengerShape,
    },
    felt_bytes_to_bn254_var, felts_to_bn254_var,
    fri::{dummy_hash, PolynomialBatchShape, PolynomialShape},
    hash::{FieldHasher, FieldHasherVariable, Posedion2BabyBearHasherVariable},
    words_to_bytes, CircuitConfig, MultiField32ChallengerVariable,
};
use dt_recursion_compiler::{
    circuit::CircuitV2Builder,
    ir::{Builder, Config, DslIr, Felt},
    prelude::Var,
};
use dt_recursion_core::{air::RecursionPublicValues, DIGEST_SIZE, PERMUTATION_WIDTH};
use dt_stark::{
    air::MachineAir,
    baby_bear_poseidon2::{SCBabyBearPoseidon2, ValMmcs},
    count_permutation_constraints, inner_perm,
    machine::SCStarkMachine,
    shape::OrderedShape,
    sumcheck::{
        config::SCStarkGenericConfig,
        keys::SCStarkVerifyingKey,
        proof::{
            SCChipOpenedValues, SCShardCommitment, SCShardOpenedValues, SCShardProof, SumcheckProof,
        },
    },
    Challenge, Challenger, Chip, InnerBasefoldProof, InnerChallenge, StarkGenericConfig, Val,
    PROOF_MAX_NUM_PVS,
};
use dummy::{dummy_unipoly, sc_dummy_pcs_proof};
use hashbrown::HashMap;
use itertools::Itertools;
use p3_air::{Air, BaseAir};
use p3_baby_bear::BabyBear;
use p3_bn254_fr::Bn254Fr;
use p3_challenger::{CanObserve, CanSample, FieldChallenger, GrindingChallenger};
use p3_commit::{ExtensionMmcs, Mmcs};
use p3_dft::Radix2DitParallel;
use p3_field::{extension::BinomialExtensionField, AbstractField, ExtensionField, Field};
use p3_fri::{FriConfig, TwoAdicFriPcs};
use p3_matrix::{dense::RowMajorMatrix, Dimensions};
use p3_symmetric::Permutation;
use p3_uni_stark::{get_max_constraint_degree_sc, get_symbolic_constraints, SymbolicAirBuilder};
use std::iter::{repeat, zip};

#[cfg(feature = "koalabear")]
use dt_primitives::SCField;
#[cfg(feature = "koalabear")]
use dt_recursion_core::stark::SCKoalaBearPoseidon2Outer;
#[cfg(feature = "koalabear")]
use dt_recursion_core::stark::SCOuterValMmcs;
#[cfg(feature = "babybear")]
use dt_recursion_core::stark::{OuterValMmcs, SCBabyBearPoseidon2Outer};

// pub mod pcs;

mod dummy;
#[cfg(feature = "koalabear")]
mod dummy_koalabear;
pub mod pcs;
pub mod polyair_folder;
pub mod polyair_precompute;
pub mod polyair_verifier;
pub mod types;
pub mod utils;
pub mod verifier;
pub mod witness;

pub type SCFriMmcs<C> = ExtensionMmcs<Val<C>, Challenge<C>, <C as SCBabyBearFriConfig>::ValMmcs>;

pub trait SCBabyBearFriConfig:
    SCStarkGenericConfig<
    Challenger = Self::FriChallenger,
    Pcs = TwoAdicFriPcs<
        Val<Self>,
        Radix2DitParallel,
        Self::ValMmcs,
        ExtensionMmcs<Val<Self>, Challenge<Self>, Self::ValMmcs>,
    >,
>
{
    type ValMmcs: Mmcs<Val<Self>, ProverData<RowMajorMatrix<Val<Self>>> = Self::RowMajorProverData>
        + Send
        + Sync;
    type RowMajorProverData: Clone + Send + Sync;
    type FriChallenger: CanObserve<<Self::ValMmcs as Mmcs<Val<Self>>>::Commitment>
        + CanSample<Challenge<Self>>
        + GrindingChallenger<Witness = Val<Self>>
        + FieldChallenger<Val<Self>>;

    fn fri_config(&self) -> &FriConfig<SCFriMmcs<Self>>;

    fn challenger_shape(challenger: &Self::FriChallenger) -> SpongeChallengerShape;

    fn cross_round_log_foldings(&self) -> Vec<usize> {
        Vec::new()
    }
}

#[cfg(feature = "babybear")]
pub trait SCBabyBearFriConfigVariable<C: CircuitConfig<F = Val<Self>>>:
    SCBabyBearFriConfig + FieldHasherVariable<C> + Posedion2BabyBearHasherVariable<C>
{
    type FriChallengerVariable: FieldChallengerVariable<C, <C as CircuitConfig>::Bit>
        + CanObserveVariable<C, <Self as FieldHasherVariable<C>>::DigestVariable>
        + CanCopyChallenger<C>;

    fn challenger_variable(&self, builder: &mut Builder<C>) -> Self::FriChallengerVariable;

    fn commit_recursion_public_values(
        builder: &mut Builder<C>,
        public_values: RecursionPublicValues<Felt<C::F>>,
    );
}

#[cfg(feature = "koalabear")]
pub trait KoalaBearSCFriConfigVariable<C: CircuitConfig<F = Val<Self>>>:
    SCBabyBearFriConfig + FieldHasherVariable<C> + Poseidon2KoalaBearHasherVariable<C>
{
    type FriChallengerVariable: FieldChallengerVariable<C, <C as CircuitConfig>::Bit>
        + CanObserveVariable<C, <Self as FieldHasherVariable<C>>::DigestVariable>
        + CanCopyChallenger<C>;

    fn challenger_variable(&self, builder: &mut Builder<C>) -> Self::FriChallengerVariable;

    fn commit_recursion_public_values(
        builder: &mut Builder<C>,
        public_values: RecursionPublicValues<Felt<C::F>>,
    );
}

#[cfg(feature = "koalabear")]
pub use KoalaBearSCFriConfigVariable as SCBabyBearFriConfigVariable;

impl SCBabyBearFriConfig for SCBabyBearPoseidon2 {
    type ValMmcs = ValMmcs;
    type RowMajorProverData = <ValMmcs as Mmcs<BabyBear>>::ProverData<RowMajorMatrix<BabyBear>>;
    type FriChallenger = <Self as StarkGenericConfig>::Challenger;

    fn fri_config(&self) -> &FriConfig<SCFriMmcs<Self>> {
        self.pcs().fri_config()
    }

    fn challenger_shape(challenger: &Self::FriChallenger) -> SpongeChallengerShape {
        SpongeChallengerShape {
            input_buffer_len: challenger.input_buffer.len(),
            output_buffer_len: challenger.output_buffer.len(),
        }
    }

    fn cross_round_log_foldings(&self) -> Vec<usize> {
        Vec::new()
    }
}

#[cfg(feature = "babybear")]
impl SCBabyBearFriConfig for SCBabyBearPoseidon2Outer {
    type ValMmcs = OuterValMmcs;
    type FriChallenger = <Self as StarkGenericConfig>::Challenger;

    type RowMajorProverData =
        <OuterValMmcs as Mmcs<BabyBear>>::ProverData<RowMajorMatrix<BabyBear>>;

    fn fri_config(&self) -> &FriConfig<SCFriMmcs<Self>> {
        self.pcs().fri_config()
    }

    fn challenger_shape(_challenger: &Self::FriChallenger) -> SpongeChallengerShape {
        unimplemented!("Shape not supported for outer fri challenger");
    }

    fn cross_round_log_foldings(&self) -> Vec<usize> {
        Vec::new()
    }
}
impl<C: CircuitConfig<F = BabyBear, Bit = Felt<BabyBear>>> FieldHasherVariable<C>
    for SCBabyBearPoseidon2
{
    type DigestVariable = [Felt<BabyBear>; DIGEST_SIZE];

    fn hash(builder: &mut Builder<C>, input: &[Felt<<C as Config>::F>]) -> Self::DigestVariable {
        <Self as Posedion2BabyBearHasherVariable<C>>::poseidon2_hash(builder, input)
    }

    fn compress(
        builder: &mut Builder<C>,
        input: [Self::DigestVariable; 2],
    ) -> Self::DigestVariable {
        builder.poseidon2_compress_v2(input.into_iter().flatten())
    }

    fn assert_digest_eq(
        builder: &mut Builder<C>,
        a: Self::DigestVariable,
        b: Self::DigestVariable,
    ) {
        // Push the instruction directly instead of passing through `assert_felt_eq` in order to
        //avoid symbolic expression overhead.
        zip(a, b).for_each(|(e1, e2)| builder.push_op(DslIr::AssertEqF(e1, e2)));
    }

    fn select_chain_digest(
        builder: &mut Builder<C>,
        should_swap: <C as CircuitConfig>::Bit,
        input: [Self::DigestVariable; 2],
    ) -> [Self::DigestVariable; 2] {
        let result0: [Felt<BabyBear>; DIGEST_SIZE] = core::array::from_fn(|_| builder.uninit());
        let result1: [Felt<BabyBear>; DIGEST_SIZE] = core::array::from_fn(|_| builder.uninit());

        (0..DIGEST_SIZE).for_each(|i| {
            builder.push_op(DslIr::Select(
                should_swap,
                result0[i],
                result1[i],
                input[0][i],
                input[1][i],
            ));
        });

        #[cfg(feature = "verify")]
        {
            if should_swap.val.is_zero() {
                result0.into_iter().zip(result1).zip(input[0].into_iter().zip(input[1])).for_each(
                    |((_r0, _r1), (i0, i1))| {
                        // r0.val = i0.val;
                        // r1.val = i1.val;
                    },
                );
            } else {
                result0.into_iter().zip(result1).zip(input[0].into_iter().zip(input[1])).for_each(
                    |((_r0, _r1), (i0, i1))| {
                        // r0.val = i1.val;
                        // r1.val = i0.val;
                    },
                );
            }
        }

        [result0, result1]
    }

    fn print_digest(builder: &mut Builder<C>, digest: Self::DigestVariable) {
        for d in digest.iter() {
            builder.print_f(*d);
        }
    }
}

impl FieldHasher<BabyBear> for SCBabyBearPoseidon2 {
    type Digest = [BabyBear; DIGEST_SIZE];

    fn constant_compress(input: [Self::Digest; 2]) -> Self::Digest {
        let mut pre_iter = input.into_iter().flatten().chain(repeat(BabyBear::zero()));
        let mut pre = core::array::from_fn(move |_| pre_iter.next().unwrap());
        (inner_perm()).permute_mut(&mut pre);
        pre[..DIGEST_SIZE].try_into().unwrap()
    }
}

impl<C: CircuitConfig<F = BabyBear, Bit = Felt<BabyBear>>> Posedion2BabyBearHasherVariable<C>
    for SCBabyBearPoseidon2
{
    fn poseidon2_permute(
        builder: &mut Builder<C>,
        state: [Felt<C::F>; PERMUTATION_WIDTH],
    ) -> [Felt<C::F>; PERMUTATION_WIDTH] {
        builder.poseidon2_permute_v2(state)
    }
}

#[cfg(feature = "babybear")]
impl<C: CircuitConfig<F = BabyBear, Bit = Felt<BabyBear>>> SCBabyBearFriConfigVariable<C>
    for SCBabyBearPoseidon2
{
    type FriChallengerVariable = DuplexChallengerVariable<C>;

    fn challenger_variable(&self, builder: &mut Builder<C>) -> Self::FriChallengerVariable {
        DuplexChallengerVariable::new(builder)
    }

    fn commit_recursion_public_values(
        builder: &mut Builder<C>,
        public_values: RecursionPublicValues<Felt<<C>::F>>,
    ) {
        builder.commit_public_values_v2(public_values);
    }
}

#[cfg(feature = "babybear")]
impl<C: CircuitConfig<F = BabyBear, N = Bn254Fr, Bit = Var<Bn254Fr>>> SCBabyBearFriConfigVariable<C>
    for SCBabyBearPoseidon2Outer
{
    type FriChallengerVariable = MultiField32ChallengerVariable<C>;

    fn challenger_variable(&self, builder: &mut Builder<C>) -> Self::FriChallengerVariable {
        MultiField32ChallengerVariable::new(builder)
    }

    fn commit_recursion_public_values(
        builder: &mut Builder<C>,
        public_values: RecursionPublicValues<Felt<<C>::F>>,
    ) {
        let committed_values_digest_bytes_felts: [Felt<_>; 32] =
            words_to_bytes(&public_values.committed_value_digest).try_into().unwrap();
        let committed_values_digest_bytes: Var<_> =
            felt_bytes_to_bn254_var(builder, &committed_values_digest_bytes_felts);
        builder.commit_committed_values_digest_circuit(committed_values_digest_bytes);

        let vkey_hash = felts_to_bn254_var(builder, &public_values.dt_vk_digest);
        builder.commit_vkey_hash_circuit(vkey_hash);
    }
}

// --- KoalaBear feature-gated impls ---

#[cfg(feature = "koalabear")]
use dt_stark::koalabear_poseidon2::koala_bear_poseidon2::SCKoalaBearPoseidon2;

#[cfg(feature = "koalabear")]
impl SCBabyBearFriConfig for SCKoalaBearPoseidon2 {
    type ValMmcs = dt_stark::koalabear_poseidon2::koala_bear_poseidon2::ValMmcs;
    type RowMajorProverData =
        <dt_stark::koalabear_poseidon2::koala_bear_poseidon2::ValMmcs as Mmcs<
            p3_koala_bear::KoalaBear,
        >>::ProverData<RowMajorMatrix<p3_koala_bear::KoalaBear>>;
    type FriChallenger = <Self as StarkGenericConfig>::Challenger;

    fn fri_config(&self) -> &FriConfig<SCFriMmcs<Self>> {
        self.pcs().fri_config()
    }

    fn challenger_shape(challenger: &Self::FriChallenger) -> SpongeChallengerShape {
        SpongeChallengerShape {
            input_buffer_len: challenger.input_buffer.len(),
            output_buffer_len: challenger.output_buffer.len(),
        }
    }

    fn cross_round_log_foldings(&self) -> Vec<usize> {
        SCKoalaBearPoseidon2::cross_round_log_foldings(self)
    }
}

#[cfg(feature = "koalabear")]
impl FieldHasher<p3_koala_bear::KoalaBear> for SCKoalaBearPoseidon2 {
    type Digest = [p3_koala_bear::KoalaBear; DIGEST_SIZE];

    fn constant_compress(input: [Self::Digest; 2]) -> Self::Digest {
        let mut pre_iter =
            input.into_iter().flatten().chain(repeat(p3_koala_bear::KoalaBear::zero()));
        let mut pre = core::array::from_fn(move |_| pre_iter.next().unwrap());
        (dt_stark::koalabear_poseidon2::koala_bear_poseidon2::my_perm()).permute_mut(&mut pre);
        pre[..DIGEST_SIZE].try_into().unwrap()
    }
}

#[cfg(feature = "koalabear")]
impl<C: CircuitConfig<F = SCField, Bit = Felt<SCField>>> Poseidon2KoalaBearHasherVariable<C>
    for SCKoalaBearPoseidon2
{
    fn poseidon2_permute(
        builder: &mut Builder<C>,
        state: [Felt<C::F>; PERMUTATION_WIDTH],
    ) -> [Felt<C::F>; PERMUTATION_WIDTH] {
        builder.poseidon2_permute_v2(state)
    }
}

#[cfg(feature = "koalabear")]
impl<C: CircuitConfig<F = SCField, Bit = Felt<SCField>>> FieldHasherVariable<C>
    for SCKoalaBearPoseidon2
{
    type DigestVariable = [Felt<SCField>; DIGEST_SIZE];

    fn hash(builder: &mut Builder<C>, input: &[Felt<<C as Config>::F>]) -> Self::DigestVariable {
        <Self as Poseidon2KoalaBearHasherVariable<C>>::poseidon2_hash(builder, input)
    }

    fn compress(
        builder: &mut Builder<C>,
        input: [Self::DigestVariable; 2],
    ) -> Self::DigestVariable {
        builder.poseidon2_compress_v2(input.into_iter().flatten())
    }

    fn assert_digest_eq(
        builder: &mut Builder<C>,
        a: Self::DigestVariable,
        b: Self::DigestVariable,
    ) {
        zip(a, b).for_each(|(e1, e2)| builder.push_op(DslIr::AssertEqF(e1, e2)));
    }

    fn select_chain_digest(
        builder: &mut Builder<C>,
        should_swap: <C as CircuitConfig>::Bit,
        input: [Self::DigestVariable; 2],
    ) -> [Self::DigestVariable; 2] {
        let result0: [Felt<SCField>; DIGEST_SIZE] = core::array::from_fn(|_| builder.uninit());
        let result1: [Felt<SCField>; DIGEST_SIZE] = core::array::from_fn(|_| builder.uninit());

        (0..DIGEST_SIZE).for_each(|i| {
            builder.push_op(DslIr::Select(
                should_swap,
                result0[i],
                result1[i],
                input[0][i],
                input[1][i],
            ));
        });

        #[cfg(feature = "verify")]
        {
            if should_swap.val.is_zero() {
                result0
                    .into_iter()
                    .zip(result1.into_iter())
                    .zip(input[0].into_iter().zip(input[1].into_iter()))
                    .for_each(|((mut r0, mut r1), (i0, i1))| {
                        r0.val = i0.val;
                        r1.val = i1.val;
                    });
            } else {
                result0
                    .into_iter()
                    .zip(result1.into_iter())
                    .zip(input[0].into_iter().zip(input[1].into_iter()))
                    .for_each(|((mut r0, mut r1), (i0, i1))| {
                        r0.val = i1.val;
                        r1.val = i0.val;
                    });
            }
        }

        [result0, result1]
    }

    fn print_digest(builder: &mut Builder<C>, digest: Self::DigestVariable) {
        for d in digest.iter() {
            builder.print_f(*d);
        }
    }
}

#[cfg(feature = "koalabear")]
impl<C: CircuitConfig<F = SCField, Bit = Felt<SCField>>> KoalaBearSCFriConfigVariable<C>
    for SCKoalaBearPoseidon2
{
    type FriChallengerVariable = DuplexChallengerVariable<C>;

    fn challenger_variable(&self, builder: &mut Builder<C>) -> Self::FriChallengerVariable {
        DuplexChallengerVariable::new(builder)
    }

    fn commit_recursion_public_values(
        builder: &mut Builder<C>,
        public_values: RecursionPublicValues<Felt<<C>::F>>,
    ) {
        builder.commit_public_values_v2(public_values);
    }
}

#[cfg(feature = "koalabear")]
impl SCBabyBearFriConfig for SCKoalaBearPoseidon2Outer {
    type ValMmcs = SCOuterValMmcs;
    type FriChallenger = <Self as StarkGenericConfig>::Challenger;
    type RowMajorProverData =
        <SCOuterValMmcs as Mmcs<SCField>>::ProverData<RowMajorMatrix<SCField>>;

    fn fri_config(&self) -> &FriConfig<SCFriMmcs<Self>> {
        self.pcs().fri_config()
    }

    fn challenger_shape(_challenger: &Self::FriChallenger) -> SpongeChallengerShape {
        unimplemented!("Shape not supported for outer fri challenger");
    }

    fn cross_round_log_foldings(&self) -> Vec<usize> {
        SCKoalaBearPoseidon2Outer::cross_round_log_foldings(self)
    }
}

#[cfg(feature = "koalabear")]
impl<C: CircuitConfig<F = SCField, N = Bn254Fr, Bit = Var<Bn254Fr>>> KoalaBearSCFriConfigVariable<C>
    for SCKoalaBearPoseidon2Outer
{
    type FriChallengerVariable = MultiField32ChallengerVariable<C>;

    fn challenger_variable(&self, builder: &mut Builder<C>) -> Self::FriChallengerVariable {
        MultiField32ChallengerVariable::new(builder)
    }

    fn commit_recursion_public_values(
        builder: &mut Builder<C>,
        public_values: RecursionPublicValues<Felt<<C>::F>>,
    ) {
        let committed_values_digest_bytes_felts: [Felt<_>; 32] =
            words_to_bytes(&public_values.committed_value_digest).try_into().unwrap();
        let committed_values_digest_bytes: Var<_> =
            felt_bytes_to_bn254_var(builder, &committed_values_digest_bytes_felts);
        builder.commit_committed_values_digest_circuit(committed_values_digest_bytes);

        let vkey_hash = felts_to_bn254_var(builder, &public_values.dt_vk_digest);
        builder.commit_vkey_hash_circuit(vkey_hash);
    }
}

/// Get a dummy duplex challenger for use in dummy proofs.
pub fn sc_dummy_challenger(config: &SCBabyBearPoseidon2) -> Challenger<SCBabyBearPoseidon2> {
    let mut challenger = config.challenger();
    challenger.input_buffer = vec![];
    challenger.output_buffer = vec![BabyBear::zero(); challenger.sponge_state.len()];
    challenger
}

pub fn check_vk_shape(
    left: &SCStarkVerifyingKey<SCBabyBearPoseidon2>,
    right: &SCStarkVerifyingKey<SCBabyBearPoseidon2>,
) {
    assert_eq!(
        left.chip_information.len(),
        right.chip_information.len(),
        "chip information has different length"
    );
    for (lhs, rhs) in left.chip_information.iter().zip(right.chip_information.iter()) {
        assert_eq!(lhs, rhs, "different chip info");
    }
    //chip ordering
    for (chip, order) in left.chip_ordering.iter() {
        let rhs_order = right.chip_ordering.get(chip).expect("right does not have chip in left");
        if *order != *rhs_order {
            panic!("chip: {:?}, has different order: {} && {}", chip, rhs_order, order);
        }
    }
    assert_eq!(left.chip_ordering.len(), right.chip_ordering.len());
}

pub fn check_shard_proof(
    left: &SCShardProof<SCBabyBearPoseidon2>,
    right: &SCShardProof<SCBabyBearPoseidon2>,
) {
    //commitment
    assert_eq!(
        left.commitment.permutation_commit.is_some(),
        right.commitment.permutation_commit.is_some()
    );
    //opened_values
    assert_eq!(
        left.opened_values.chips.len(),
        right.opened_values.chips.len(),
        "opened values has different length"
    );
    for (idx, (lhs, rhs)) in
        left.opened_values.chips.iter().zip(right.opened_values.chips.iter()).enumerate()
    {
        if lhs.preprocessed.local.len() != rhs.preprocessed.local.len() {
            panic!("No {idx}, different preprocessed local len");
        }
        if lhs.main.local.len() != rhs.main.local.len() {
            panic!(
                "No {idx}, different main local len, left {}, right {}",
                lhs.main.local.len(),
                rhs.main.local.len()
            );
        }
        if lhs.permutation.local.len() != rhs.permutation.local.len() {
            panic!("No {idx}, different permutation local len");
        }
        if lhs.log_height != rhs.log_height {
            panic!("No {idx}, different log_height: {} && {}", lhs.log_height, rhs.log_height);
        }
    }
    //openning proof
    // let lhs_opening_proof_bytes = bincode::serialize(&left.opening_proof).unwrap();
    // let rhs_opening_proof_bytes = bincode::serialize(&right.opening_proof).unwrap();
    // assert_eq!(lhs_opening_proof_bytes.len(),rhs_opening_proof_bytes.len(),"pcs proof error");
    check_basefold(&left.opening_proof, &right.opening_proof);
    //sumcheck proof
    check_sumcheck_proof(&left.sumcheck_proof, &right.sumcheck_proof);
    //dimensions
    assert_eq!(left.dimensions, right.dimensions);
    //chip_orderings
    assert_eq!(left.chip_ordering, right.chip_ordering);
    //public values
    assert_eq!(left.public_values.len(), right.public_values.len());
    tracing::trace!("check proof successfully");
}

fn check_sumcheck_proof(
    left: &SumcheckProof<SCBabyBearPoseidon2>,
    right: &SumcheckProof<SCBabyBearPoseidon2>,
) {
    assert_eq!(left.unipolys.len(), right.unipolys.len(), "different sumcheck proof unipolys len");
    for (idx, (lhs, rhs)) in left.unipolys.iter().zip(right.unipolys.iter()).enumerate() {
        if lhs.evals.len() != rhs.evals.len() {
            panic!(
                "In sumcheck proof, No {idx} unipoly has different degree {} && {}",
                lhs.evals.len(),
                rhs.evals.len()
            );
        }
    }
}
/*

pub struct BasefoldProof<F: Field,  M: Mmcs<F>, Witness, InputProof> {
    pub sumcheck_transcript: SumcheckInstanceProof<F>,
    pub iopp_oracles: Vec<M::Commitment>,
    pub iopp_queries: Vec<QueryProof<F, M>>,
    pub query_openings: InputProof,
    pub grinding_batching_witness: Vec<Witness>,
    pub grinding_query_witness: Vec<Witness>,
}
*/
fn check_basefold(left: &InnerBasefoldProof, right: &InnerBasefoldProof) {
    //sumcheck
    assert_eq!(
        left.sumcheck_transcript.uni_polys.len(),
        right.sumcheck_transcript.uni_polys.len(),
        "different sumcheck proof unipolys len"
    );
    for (idx, (lhs, rhs)) in left
        .sumcheck_transcript
        .uni_polys
        .iter()
        .zip(right.sumcheck_transcript.uni_polys.iter())
        .enumerate()
    {
        if lhs.coeffs.len() != rhs.coeffs.len() {
            panic!(
                "In sumcheck proof, No {idx} unipoly has different degree {} && {}",
                lhs.coeffs.len(),
                rhs.coeffs.len()
            );
        }
    }
    //iopp_oracles
    assert_eq!(left.iopp_oracles.len(), right.iopp_oracles.len(), "different iopp_oracles len");
    let left_queries = left.iopp_queries.clone();
    let right_queries = right.iopp_queries.clone();
    assert_eq!(left_queries.len(), right_queries.len(), "different iopp_queries len");
    for (lhs, rhs) in left_queries.iter().zip(right_queries.iter()) {
        assert_eq!(lhs.commit_phase_openings.len(), rhs.commit_phase_openings.len());
    }
    let left_opening = left.query_openings.per_query.clone();
    let right_opening = right.query_openings.per_query.clone();
    assert_eq!(left_opening.len(), right_opening.len(), "different query_openings len");
    // When per_query is empty (pruned / size-saving mode), skip inner asserts.
    if !left_opening.is_empty() {
        for (lhs, rhs) in left_opening.iter().zip(right_opening.iter()) {
            assert_eq!(lhs.len(), rhs.len());
            for (slhs, srhs) in lhs.iter().zip(rhs.iter()) {
                assert_eq!(slhs.opened_values.len(), srhs.opened_values.len());
                assert_eq!(slhs.opening_proof.len(), srhs.opening_proof.len());
                for (tlhs, trhs) in slhs.opened_values.iter().zip(srhs.opened_values.iter()) {
                    assert_eq!(tlhs.len(), trhs.len());
                }
            }
        }
    }
    assert_eq!(left.grinding_batching_witness.len(), right.grinding_batching_witness.len());
    assert_eq!(left.grinding_query_witness.len(), right.grinding_query_witness.len());
}
/// Make a dummy shard proof for a given proof shape.
pub fn sc_dummy_vk_and_shard_proof<
    A: MachineAir<BabyBear> + for<'a> Air<SymbolicAirBuilder<BabyBear>>,
    AE: MachineAir<BinomialExtensionField<BabyBear, 4>>,
>(
    machine: &SCStarkMachine<SCBabyBearPoseidon2, A, AE>,
    shape: &OrderedShape,
) -> (SCStarkVerifyingKey<SCBabyBearPoseidon2>, SCShardProof<SCBabyBearPoseidon2>) {
    // Make a dummy commitment.
    let commitment =
        SCShardCommitment { main_commit: dummy_hash(), permutation_commit: Some(dummy_hash()) };

    // Get dummy opened values by reading the chip ordering from the shape.
    let chip_ordering = shape
        .inner
        .iter()
        .enumerate()
        .map(|(i, (name, _))| (name.clone(), i))
        .collect::<HashMap<_, _>>();

    let shard_chips = machine.shard_chips_ordered(&chip_ordering).collect::<Vec<_>>();
    let shard_chips_name = shard_chips.iter().map(|chip| chip.name()).collect::<Vec<_>>();

    let mut chip_shapes = vec![];
    shape.inner.iter().for_each(|(_, log_degree)| chip_shapes.push(*log_degree));

    let opened_values = SCShardOpenedValues {
        chips: shard_chips
            .iter()
            .zip_eq(shape.inner.iter())
            .map(|(chip, (_, log_degree))| {
                dummy_opened_values::<_, InnerChallenge, _>(chip, *log_degree)
            })
            .collect(),
        _field: core::marker::PhantomData,
    };

    let mut preprocessed_names_and_dimensions = vec![];
    let mut preprocessed_batch_shape = vec![];
    let mut main_batch_shape = vec![];
    let mut permutation_batch_shape = vec![];

    let mut constraints_map = HashMap::new();

    let mut prep_dimensions = vec![];
    let mut main_dimensions = vec![];
    let mut permutation_dimensions = vec![];

    for (chip, chip_opening) in shard_chips.iter().zip_eq(opened_values.chips.iter()) {
        if !chip_opening.preprocessed.local.is_empty() {
            let prep_shape = PolynomialShape {
                width: chip_opening.preprocessed.local.len(),
                log_degree: chip_opening.log_height,
            };
            prep_dimensions.push(prep_shape.dimensions());
            preprocessed_names_and_dimensions.push((
                chip.name(),
                prep_shape.width,
                prep_shape.log_degree,
            ));
            preprocessed_batch_shape.push(prep_shape);
        }
        let main_shape = PolynomialShape {
            width: chip_opening.main.local.len(),
            log_degree: chip_opening.log_height,
        };
        main_dimensions.push(main_shape.dimensions());
        main_batch_shape.push(main_shape);
        let permutation_shape = PolynomialShape {
            width: chip_opening.permutation.local.len() * 4,
            log_degree: chip_opening.log_height,
        };
        permutation_dimensions.push(permutation_shape.perm_dimensions());
        permutation_batch_shape.push(permutation_shape);
        let num_main_constraints =
            get_symbolic_constraints(&chip.air, chip.preprocessed_width(), PROOF_MAX_NUM_PVS).len();

        let num_permutation_constraints = count_permutation_constraints(
            &chip.sends,
            &chip.receives,
            chip.logup_batch_size(),
            chip.air.commit_scope(),
        );
        constraints_map.insert(chip.name(), num_main_constraints + num_permutation_constraints);
    }
    let dimensions = vec![prep_dimensions, main_dimensions, permutation_dimensions];

    let batch_shapes = vec![
        PolynomialBatchShape { shapes: preprocessed_batch_shape },
        PolynomialBatchShape { shapes: main_batch_shape },
        PolynomialBatchShape { shapes: permutation_batch_shape },
    ];

    let fri_queries = machine.config().fri_config().num_queries;
    let blowup_bits = machine.config().fri_config().log_blowup;
    let log_final_poly_len = machine.config().fri_config().log_final_poly_len;

    let opening_proof =
        sc_dummy_pcs_proof(&batch_shapes, blowup_bits, fri_queries, log_final_poly_len);

    let public_values = (0..PROOF_MAX_NUM_PVS).map(|_| BabyBear::zero()).collect::<Vec<_>>();

    // Get the preprocessed chip information.
    let preprocessed_chip_information: Vec<_> = preprocessed_names_and_dimensions
        .iter()
        .map(|(name, width, log_height)| {
            (name.to_owned(), Dimensions { width: *width, height: 1 << log_height })
        })
        .collect();

    // Get the chip ordering.
    let preprocessed_chip_ordering = preprocessed_names_and_dimensions
        .iter()
        .enumerate()
        .map(|(i, (name, _, _))| (name.to_owned(), i))
        .collect::<HashMap<_, _>>();

    let vk = SCStarkVerifyingKey {
        commit: dummy_hash(),
        pc_start: BabyBear::zero(),
        program_boundary: dt_stark::global_d11::ProgramImageBoundaryV1::Infinity,
        owner_registry: dt_stark::global_d11::BoundaryOwnerRegistryV2::new(Vec::new())
            .expect("empty owner registry"),
        global146_identity: dt_stark::global_d11::GLOBAL146_COMPOSITE_IDENTITY,
        chip_information: preprocessed_chip_information,
        chip_ordering: preprocessed_chip_ordering,
        constraints_map,
    };

    let sumcheck_proof = dummy_sumcheck_proof(shard_chips, &chip_shapes);
    let shard_proof = SCShardProof {
        commitment,
        opened_values,
        opening_proof,
        chip_ordering,
        public_values,
        dimensions,
        sumcheck_proof,
    };
    (vk, shard_proof)
}

fn dummy_opened_values<F: Field, EF: ExtensionField<F>, A: MachineAir<F>>(
    chip: &Chip<F, A>,
    log_degree: usize,
) -> SCChipOpenedValues<F, EF> {
    let preprocessed_width = chip.preprocessed_width();
    let preprocessed = dt_stark::SCAirOpenedValues { local: vec![EF::zero(); preprocessed_width] };
    let main_width = chip.width();
    let main = dt_stark::SCAirOpenedValues { local: vec![EF::zero(); main_width] };
    let permutation_width = chip.permutation_width();
    let permutation = dt_stark::SCAirOpenedValues { local: vec![EF::zero(); permutation_width] };

    SCChipOpenedValues {
        preprocessed,
        main,
        permutation,
        local_cumulative_sum: EF::zero(),
        log_height: log_degree,
        _field: core::marker::PhantomData,
    }
}

fn polyair_dummy_opened_values<
    F: Field + dt_stark::air::PolyAirExtendable<D>,
    EF: ExtensionField<F>,
    A: MachineAir<F>,
    const D: usize,
>(
    chip: &polyair::Chip<A, F, D>,
    log_degree: usize,
) -> SCChipOpenedValues<F, EF> {
    let preprocessed_width = chip.preprocessed_width();
    let preprocessed = dt_stark::SCAirOpenedValues { local: vec![EF::zero(); preprocessed_width] };
    let main_width = chip.width();
    let main = dt_stark::SCAirOpenedValues { local: vec![EF::zero(); main_width] };
    let permutation_width = chip.perm_width();
    let permutation = dt_stark::SCAirOpenedValues { local: vec![EF::zero(); permutation_width] };

    SCChipOpenedValues {
        preprocessed,
        main,
        permutation,
        local_cumulative_sum: EF::zero(),
        log_height: log_degree,
        _field: core::marker::PhantomData,
    }
}

pub fn polyair_sc_dummy_vk_and_shard_proof<A: MachineAir<BabyBear>, const D: usize>(
    machine: &polyair::SCStarkMachine<SCBabyBearPoseidon2, A, D>,
    shape: &OrderedShape,
) -> (SCStarkVerifyingKey<SCBabyBearPoseidon2>, SCShardProof<SCBabyBearPoseidon2>)
where
    BabyBear: dt_stark::air::PolyAirExtendable<D>,
{
    let commitment =
        SCShardCommitment { main_commit: dummy_hash(), permutation_commit: Some(dummy_hash()) };

    let chip_ordering = shape
        .inner
        .iter()
        .enumerate()
        .map(|(i, (name, _))| (name.clone(), i))
        .collect::<HashMap<_, _>>();

    let shard_chips = machine.shard_chips_ordered(&chip_ordering).collect::<Vec<_>>();

    let mut chip_shapes = vec![];
    shape.inner.iter().for_each(|(_, log_degree)| chip_shapes.push(*log_degree));

    let opened_values = SCShardOpenedValues {
        chips: shard_chips
            .iter()
            .zip_eq(shape.inner.iter())
            .map(|(chip, (_, log_degree))| {
                polyair_dummy_opened_values::<_, InnerChallenge, _, D>(chip, *log_degree)
            })
            .collect(),
        _field: core::marker::PhantomData,
    };

    let mut preprocessed_names_and_dimensions = vec![];
    let mut preprocessed_batch_shape = vec![];
    let mut main_batch_shape = vec![];
    let mut permutation_batch_shape = vec![];
    let mut constraints_map = HashMap::new();
    let mut prep_dimensions = vec![];
    let mut main_dimensions = vec![];
    let mut permutation_dimensions = vec![];

    for (chip, chip_opening) in shard_chips.iter().zip_eq(opened_values.chips.iter()) {
        if !chip_opening.preprocessed.local.is_empty() {
            let prep_shape = PolynomialShape {
                width: chip_opening.preprocessed.local.len(),
                log_degree: chip_opening.log_height,
            };
            prep_dimensions.push(prep_shape.dimensions());
            preprocessed_names_and_dimensions.push((
                chip.name(),
                prep_shape.width,
                prep_shape.log_degree,
            ));
            preprocessed_batch_shape.push(prep_shape);
        }
        let main_shape = PolynomialShape {
            width: chip_opening.main.local.len(),
            log_degree: chip_opening.log_height,
        };
        main_dimensions.push(main_shape.dimensions());
        main_batch_shape.push(main_shape);
        let permutation_shape = PolynomialShape {
            width: chip_opening.permutation.local.len() * D,
            log_degree: chip_opening.log_height,
        };
        permutation_dimensions.push(permutation_shape.perm_dimensions());
        permutation_batch_shape.push(permutation_shape);
        constraints_map.insert(chip.name(), chip.num_alpha);
    }
    let dimensions = vec![prep_dimensions, main_dimensions, permutation_dimensions];

    let batch_shapes = vec![
        PolynomialBatchShape { shapes: preprocessed_batch_shape },
        PolynomialBatchShape { shapes: main_batch_shape },
        PolynomialBatchShape { shapes: permutation_batch_shape },
    ];

    let fri_queries = machine.config().fri_config().num_queries;
    let blowup_bits = machine.config().fri_config().log_blowup;
    let log_final_poly_len = machine.config().fri_config().log_final_poly_len;
    let opening_proof =
        sc_dummy_pcs_proof(&batch_shapes, blowup_bits, fri_queries, log_final_poly_len);

    let public_values = (0..PROOF_MAX_NUM_PVS).map(|_| BabyBear::zero()).collect::<Vec<_>>();
    let preprocessed_chip_information: Vec<_> = preprocessed_names_and_dimensions
        .iter()
        .map(|(name, width, log_height)| {
            (name.to_owned(), Dimensions { width: *width, height: 1 << log_height })
        })
        .collect();
    let preprocessed_chip_ordering = preprocessed_names_and_dimensions
        .iter()
        .enumerate()
        .map(|(i, (name, _, _))| (name.to_owned(), i))
        .collect::<HashMap<_, _>>();

    let vk = SCStarkVerifyingKey {
        commit: dummy_hash(),
        pc_start: BabyBear::zero(),
        program_boundary: dt_stark::global_d11::ProgramImageBoundaryV1::Infinity,
        owner_registry: dt_stark::global_d11::BoundaryOwnerRegistryV2::new(Vec::new())
            .expect("empty owner registry"),
        global146_identity: dt_stark::global_d11::GLOBAL146_COMPOSITE_IDENTITY,
        chip_information: preprocessed_chip_information,
        chip_ordering: preprocessed_chip_ordering,
        constraints_map,
    };

    let sumcheck_proof = SumcheckProof {
        unipolys: (0..chip_shapes.iter().copied().max().unwrap_or(0))
            .map(|_| dummy_unipoly(1))
            .collect(),
    };
    let shard_proof = SCShardProof {
        commitment,
        opened_values,
        opening_proof,
        chip_ordering,
        public_values,
        dimensions,
        sumcheck_proof,
    };
    (vk, shard_proof)
}
fn get_chip_degree<A>(chip: &Chip<BabyBear, A>) -> usize
where
    A: MachineAir<BabyBear> + for<'a> Air<SymbolicAirBuilder<BabyBear>>,
{
    let max_degree = 3;
    get_max_constraint_degree_sc(&chip.air, chip.air.preprocessed_width(), PROOF_MAX_NUM_PVS)
        .max(max_degree)
}
fn get_max_degree<A: MachineAir<BabyBear> + for<'a> Air<SymbolicAirBuilder<BabyBear>>>(
    chip: Vec<&Chip<BabyBear, A>>,
) -> usize {
    chip.iter()
        .map(|chip| {
            let ret = get_chip_degree(chip);
            if ret > 3 {
                log::warn!("chip {:?} has degree bigger than 3, is {}", chip.name(), ret);
            }
            ret
        })
        .max()
        .unwrap()
}
fn dummy_sumcheck_proof<A: MachineAir<BabyBear> + for<'a> Air<SymbolicAirBuilder<BabyBear>>>(
    chips: Vec<&Chip<BabyBear, A>>,
    dimensions: &[usize],
) -> SumcheckProof<SCBabyBearPoseidon2> {
    use dt_core_machine::shape::{chip_log_height_threshold, num_skip_rounds};

    let max_height = *dimensions.iter().max().unwrap();
    let max_chip_degree = get_max_degree(chips.to_vec());
    let var_degree = (1usize << num_skip_rounds()) - 1;

    let num_rounds_linear = max_height.saturating_sub(chip_log_height_threshold());
    let num_rounds_nonlinear =
        std::cmp::min(max_height, chip_log_height_threshold()) / num_skip_rounds();

    let linear_degree = max_chip_degree + 1;
    let nonlinear_degree = var_degree * (num_skip_rounds() * max_chip_degree + 1);

    let mut unipolys = Vec::with_capacity(num_rounds_linear + num_rounds_nonlinear);
    for _ in 0..num_rounds_linear {
        unipolys.push(dummy_unipoly(linear_degree));
    }
    for _ in 0..num_rounds_nonlinear {
        unipolys.push(dummy_unipoly(nonlinear_degree));
    }
    SumcheckProof { unipolys }
}

// ======================== KoalaBear dummy functions ========================

#[cfg(feature = "koalabear")]
use p3_koala_bear::KoalaBear;

#[cfg(feature = "koalabear")]
pub fn kb_sc_dummy_vk_and_shard_proof<
    A: MachineAir<KoalaBear> + for<'a> Air<SymbolicAirBuilder<KoalaBear>>,
    AE: MachineAir<Challenge<SCKoalaBearPoseidon2>>,
>(
    machine: &SCStarkMachine<SCKoalaBearPoseidon2, A, AE>,
    shape: &OrderedShape,
) -> (SCStarkVerifyingKey<SCKoalaBearPoseidon2>, SCShardProof<SCKoalaBearPoseidon2>) {
    use crate::fri::kb_dummy_hash;
    use dt_stark::sumcheck::proof::SCShardCommitment;

    let commitment = SCShardCommitment {
        main_commit: kb_dummy_hash(),
        permutation_commit: Some(kb_dummy_hash()),
    };

    let chip_ordering = shape
        .inner
        .iter()
        .enumerate()
        .map(|(i, (name, _))| (name.clone(), i))
        .collect::<HashMap<_, _>>();

    let shard_chips = machine.shard_chips_ordered(&chip_ordering).collect::<Vec<_>>();

    let mut chip_shapes = vec![];
    shape.inner.iter().for_each(|(_, log_degree)| chip_shapes.push(*log_degree));

    type KBChallenge = Challenge<SCKoalaBearPoseidon2>;

    let opened_values = SCShardOpenedValues {
        chips: shard_chips
            .iter()
            .zip_eq(shape.inner.iter())
            .map(|(chip, (_, log_degree))| {
                dummy_opened_values::<_, KBChallenge, _>(chip, *log_degree)
            })
            .collect(),
        _field: core::marker::PhantomData,
    };

    let mut preprocessed_names_and_dimensions = vec![];
    let mut preprocessed_batch_shape = vec![];
    let mut main_batch_shape = vec![];
    let mut permutation_batch_shape = vec![];

    let mut constraints_map = HashMap::new();

    let mut prep_dimensions = vec![];
    let mut main_dimensions = vec![];
    let mut permutation_dimensions = vec![];

    for (chip, chip_opening) in shard_chips.iter().zip_eq(opened_values.chips.iter()) {
        if !chip_opening.preprocessed.local.is_empty() {
            let prep_shape = PolynomialShape {
                width: chip_opening.preprocessed.local.len(),
                log_degree: chip_opening.log_height,
            };
            prep_dimensions.push(prep_shape.dimensions());
            preprocessed_names_and_dimensions.push((
                chip.name(),
                prep_shape.width,
                prep_shape.log_degree,
            ));
            preprocessed_batch_shape.push(prep_shape);
        }
        let main_shape = PolynomialShape {
            width: chip_opening.main.local.len(),
            log_degree: chip_opening.log_height,
        };
        main_dimensions.push(main_shape.dimensions());
        main_batch_shape.push(main_shape);
        let permutation_shape = PolynomialShape {
            width: chip_opening.permutation.local.len() * 4,
            log_degree: chip_opening.log_height,
        };
        permutation_dimensions.push(permutation_shape.perm_dimensions());
        permutation_batch_shape.push(permutation_shape);
        let num_main_constraints =
            get_symbolic_constraints(&chip.air, chip.preprocessed_width(), PROOF_MAX_NUM_PVS).len();

        let num_permutation_constraints = count_permutation_constraints(
            &chip.sends,
            &chip.receives,
            chip.logup_batch_size(),
            chip.air.commit_scope(),
        );
        constraints_map.insert(chip.name(), num_main_constraints + num_permutation_constraints);
    }
    let dimensions = vec![prep_dimensions, main_dimensions, permutation_dimensions];

    let batch_shapes = vec![
        PolynomialBatchShape { shapes: preprocessed_batch_shape },
        PolynomialBatchShape { shapes: main_batch_shape },
        PolynomialBatchShape { shapes: permutation_batch_shape },
    ];

    let fri_queries = machine.config().fri_config().num_queries;
    let blowup_bits = machine.config().fri_config().log_blowup;
    let log_final_poly_len = machine.config().fri_config().log_final_poly_len;

    let opening_proof = dummy_koalabear::sc_dummy_pcs_proof(
        &batch_shapes,
        blowup_bits,
        fri_queries,
        log_final_poly_len,
    );

    let public_values = (0..PROOF_MAX_NUM_PVS).map(|_| KoalaBear::zero()).collect::<Vec<_>>();

    let preprocessed_chip_information: Vec<_> = preprocessed_names_and_dimensions
        .iter()
        .map(|(name, width, log_height)| {
            (name.to_owned(), Dimensions { width: *width, height: 1 << log_height })
        })
        .collect();

    let preprocessed_chip_ordering = preprocessed_names_and_dimensions
        .iter()
        .enumerate()
        .map(|(i, (name, _, _))| (name.to_owned(), i))
        .collect::<HashMap<_, _>>();

    let vk = SCStarkVerifyingKey {
        commit: kb_dummy_hash(),
        pc_start: KoalaBear::zero(),
        program_boundary: dt_stark::global_d11::ProgramImageBoundaryV1::Infinity,
        owner_registry: dt_stark::global_d11::BoundaryOwnerRegistryV2::new(Vec::new())
            .expect("empty owner registry"),
        global146_identity: dt_stark::global_d11::GLOBAL146_COMPOSITE_IDENTITY,
        chip_information: preprocessed_chip_information,
        chip_ordering: preprocessed_chip_ordering,
        constraints_map,
    };

    let sumcheck_proof = kb_dummy_sumcheck_proof(shard_chips, &chip_shapes);
    let shard_proof = SCShardProof {
        commitment,
        opened_values,
        opening_proof,
        chip_ordering,
        public_values,
        dimensions,
        sumcheck_proof,
    };
    (vk, shard_proof)
}

#[cfg(feature = "koalabear")]
pub fn kb_polyair_sc_dummy_vk_and_shard_proof<A: MachineAir<KoalaBear>, const D: usize>(
    machine: &polyair::SCStarkMachine<SCKoalaBearPoseidon2, A, D>,
    shape: &OrderedShape,
) -> (SCStarkVerifyingKey<SCKoalaBearPoseidon2>, SCShardProof<SCKoalaBearPoseidon2>)
where
    KoalaBear: dt_stark::air::PolyAirExtendable<D>,
{
    use crate::fri::kb_dummy_hash;
    use dt_stark::sumcheck::proof::SCShardCommitment;

    let commitment = SCShardCommitment {
        main_commit: kb_dummy_hash(),
        permutation_commit: Some(kb_dummy_hash()),
    };

    let chip_ordering = shape
        .inner
        .iter()
        .enumerate()
        .map(|(i, (name, _))| (name.clone(), i))
        .collect::<HashMap<_, _>>();

    let shard_chips = machine.shard_chips_ordered(&chip_ordering).collect::<Vec<_>>();
    let mut chip_shapes = vec![];
    shape.inner.iter().for_each(|(_, log_degree)| chip_shapes.push(*log_degree));

    type KBChallenge = Challenge<SCKoalaBearPoseidon2>;
    let opened_values = SCShardOpenedValues {
        chips: shard_chips
            .iter()
            .zip_eq(shape.inner.iter())
            .map(|(chip, (_, log_degree))| {
                polyair_dummy_opened_values::<_, KBChallenge, _, D>(chip, *log_degree)
            })
            .collect(),
        _field: core::marker::PhantomData,
    };

    let mut preprocessed_names_and_dimensions = vec![];
    let mut preprocessed_batch_shape = vec![];
    let mut main_batch_shape = vec![];
    let mut permutation_batch_shape = vec![];
    let mut constraints_map = HashMap::new();
    let mut prep_dimensions = vec![];
    let mut main_dimensions = vec![];
    let mut permutation_dimensions = vec![];

    for (chip, chip_opening) in shard_chips.iter().zip_eq(opened_values.chips.iter()) {
        if !chip_opening.preprocessed.local.is_empty() {
            let prep_shape = PolynomialShape {
                width: chip_opening.preprocessed.local.len(),
                log_degree: chip_opening.log_height,
            };
            prep_dimensions.push(prep_shape.dimensions());
            preprocessed_names_and_dimensions.push((
                chip.name(),
                prep_shape.width,
                prep_shape.log_degree,
            ));
            preprocessed_batch_shape.push(prep_shape);
        }
        let main_shape = PolynomialShape {
            width: chip_opening.main.local.len(),
            log_degree: chip_opening.log_height,
        };
        main_dimensions.push(main_shape.dimensions());
        main_batch_shape.push(main_shape);
        let permutation_shape = PolynomialShape {
            width: chip_opening.permutation.local.len() * D,
            log_degree: chip_opening.log_height,
        };
        permutation_dimensions.push(permutation_shape.perm_dimensions());
        permutation_batch_shape.push(permutation_shape);
        constraints_map.insert(chip.name(), chip.num_alpha);
    }
    let dimensions = vec![prep_dimensions, main_dimensions, permutation_dimensions];

    let batch_shapes = vec![
        PolynomialBatchShape { shapes: preprocessed_batch_shape },
        PolynomialBatchShape { shapes: main_batch_shape },
        PolynomialBatchShape { shapes: permutation_batch_shape },
    ];

    let fri_queries = machine.config().fri_config().num_queries;
    let blowup_bits = machine.config().fri_config().log_blowup;
    let log_final_poly_len = machine.config().fri_config().log_final_poly_len;
    let opening_proof = dummy_koalabear::sc_dummy_pcs_proof(
        &batch_shapes,
        blowup_bits,
        fri_queries,
        log_final_poly_len,
    );

    let public_values = (0..PROOF_MAX_NUM_PVS).map(|_| KoalaBear::zero()).collect::<Vec<_>>();
    let preprocessed_chip_information: Vec<_> = preprocessed_names_and_dimensions
        .iter()
        .map(|(name, width, log_height)| {
            (name.to_owned(), Dimensions { width: *width, height: 1 << log_height })
        })
        .collect();
    let preprocessed_chip_ordering = preprocessed_names_and_dimensions
        .iter()
        .enumerate()
        .map(|(i, (name, _, _))| (name.to_owned(), i))
        .collect::<HashMap<_, _>>();

    let vk = SCStarkVerifyingKey {
        commit: kb_dummy_hash(),
        pc_start: KoalaBear::zero(),
        program_boundary: dt_stark::global_d11::ProgramImageBoundaryV1::Infinity,
        owner_registry: dt_stark::global_d11::BoundaryOwnerRegistryV2::new(Vec::new())
            .expect("empty owner registry"),
        global146_identity: dt_stark::global_d11::GLOBAL146_COMPOSITE_IDENTITY,
        chip_information: preprocessed_chip_information,
        chip_ordering: preprocessed_chip_ordering,
        constraints_map,
    };

    let sumcheck_proof = SumcheckProof {
        unipolys: (0..chip_shapes.iter().copied().max().unwrap_or(0))
            .map(|_| dummy_koalabear::dummy_unipoly(1))
            .collect(),
    };
    let shard_proof = SCShardProof {
        commitment,
        opened_values,
        opening_proof,
        chip_ordering,
        public_values,
        dimensions,
        sumcheck_proof,
    };
    (vk, shard_proof)
}

#[cfg(feature = "koalabear")]
fn kb_get_chip_degree<A>(chip: &Chip<KoalaBear, A>) -> usize
where
    A: MachineAir<KoalaBear> + for<'a> Air<SymbolicAirBuilder<KoalaBear>>,
{
    let max_degree = 3;
    get_max_constraint_degree_sc(&chip.air, chip.air.preprocessed_width(), PROOF_MAX_NUM_PVS)
        .max(max_degree)
}

#[cfg(feature = "koalabear")]
fn kb_get_max_degree<A: MachineAir<KoalaBear> + for<'a> Air<SymbolicAirBuilder<KoalaBear>>>(
    chip: Vec<&Chip<KoalaBear, A>>,
) -> usize {
    chip.iter()
        .map(|chip| {
            let ret = kb_get_chip_degree(chip);
            if ret > 3 {
                log::warn!("chip {:?} has degree bigger than 3, is {}", chip.name(), ret);
            }
            ret
        })
        .max()
        .unwrap()
}

#[cfg(feature = "koalabear")]
fn kb_dummy_sumcheck_proof<
    A: MachineAir<KoalaBear> + for<'a> Air<SymbolicAirBuilder<KoalaBear>>,
>(
    chips: Vec<&Chip<KoalaBear, A>>,
    dimensions: &[usize],
) -> SumcheckProof<SCKoalaBearPoseidon2> {
    use dt_core_machine::shape::{chip_log_height_threshold, num_skip_rounds};

    let max_height = *dimensions.iter().max().unwrap();
    let max_chip_degree = kb_get_max_degree(chips.iter().copied().collect());
    let var_degree = (1usize << num_skip_rounds()) - 1;

    let num_rounds_linear = max_height.saturating_sub(chip_log_height_threshold());
    let num_rounds_nonlinear =
        std::cmp::min(max_height, chip_log_height_threshold()) / num_skip_rounds();

    let linear_degree = max_chip_degree + 1;
    let nonlinear_degree = var_degree * (num_skip_rounds() * max_chip_degree + 1);

    let mut unipolys = Vec::with_capacity(num_rounds_linear + num_rounds_nonlinear);
    for _ in 0..num_rounds_linear {
        unipolys.push(dummy_koalabear::dummy_unipoly(linear_degree));
    }
    for _ in 0..num_rounds_nonlinear {
        unipolys.push(dummy_koalabear::dummy_unipoly(nonlinear_degree));
    }
    SumcheckProof { unipolys }
}
