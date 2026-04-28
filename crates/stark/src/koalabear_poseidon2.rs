#![allow(missing_docs)]
use crate::config::{Com, StarkGenericConfig, ZeroCommitment};
use dt_primitives::{
    KoalaBear_BEGIN_EXT_CONSTS, KoalaBear_END_EXT_CONSTS, KoalaBear_PARTIAL_CONSTS,
};
use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::{extension::BinomialExtensionField, AbstractField, Field};
use p3_fri::{
    BatchOpening, CommitPhaseProofStep, FriConfig, FriProof, QueryProof, TwoAdicFriPcs,
    TwoAdicFriPcsProof,
};
use p3_koala_bear::{DiffusionMatrixKoalaBear, KoalaBear};
use p3_merkle_tree::FieldMerkleTreeMmcs;
use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
use p3_symmetric::{Hash, PaddingFreeSponge, TruncatedPermutation};
use basefold::basefold::basefold_pcs::BasefoldProof;
use serde::{Deserialize, Serialize};

pub const DIGEST_SIZE: usize = 8;

/// A configuration for inner recursion.
pub type InnerVal = KoalaBear;
pub type InnerChallenge = BinomialExtensionField<InnerVal, 4>;
pub type InnerPerm =
    Poseidon2<InnerVal, Poseidon2ExternalMatrixGeneral, DiffusionMatrixKoalaBear, 16, 3>;
pub type InnerHash = PaddingFreeSponge<InnerPerm, 16, 8, DIGEST_SIZE>;
pub type InnerDigestHash = Hash<InnerVal, InnerVal, DIGEST_SIZE>;
pub type InnerDigest = [InnerVal; DIGEST_SIZE];
pub type InnerCompress = TruncatedPermutation<InnerPerm, 2, 8, 16>;
pub type InnerValMmcs = FieldMerkleTreeMmcs<
    <InnerVal as Field>::Packing,
    <InnerVal as Field>::Packing,
    InnerHash,
    InnerCompress,
    8,
>;
pub type InnerChallengeMmcs = ExtensionMmcs<InnerVal, InnerChallenge, InnerValMmcs>;
pub type InnerChallenger = DuplexChallenger<InnerVal, InnerPerm, 16, 8>;
pub type InnerDft = Radix2DitParallel;
pub type InnerPcs = TwoAdicFriPcs<InnerVal, InnerDft, InnerValMmcs, InnerChallengeMmcs>;
pub type InnerQueryProof = QueryProof<InnerChallenge, InnerChallengeMmcs>;
pub type InnerCommitPhaseStep = CommitPhaseProofStep<InnerChallenge, InnerChallengeMmcs>;
pub type InnerFriProof = FriProof<InnerChallenge, InnerChallengeMmcs, InnerVal>;
pub type InnerBatchOpening = BatchOpening<InnerVal, InnerValMmcs>;
pub type InnerPcsProof =
    TwoAdicFriPcsProof<InnerVal, InnerChallenge, InnerValMmcs, InnerChallengeMmcs>;
pub type InnerBasefoldProof =
    BasefoldProof<InnerChallenge, InnerChallengeMmcs, InnerVal, Vec<Vec<InnerBatchOpening>>>;

/// The permutation for inner recursion.
#[must_use]
pub fn inner_perm() -> InnerPerm {
    const ROUNDS_F: usize = 8;
    const ROUNDS_P: usize = 20;
    let mut external_round_constants = KoalaBear_BEGIN_EXT_CONSTS.to_vec();
    let internal_round_constants = KoalaBear_PARTIAL_CONSTS.to_vec();
    external_round_constants.extend_from_slice(KoalaBear_END_EXT_CONSTS.as_slice());

    Poseidon2::new(
        ROUNDS_F,
        external_round_constants,
        Poseidon2ExternalMatrixGeneral,
        ROUNDS_P,
        internal_round_constants,
        DiffusionMatrixKoalaBear,
    )
}

/// The FRI config for dt proofs.
#[must_use]
pub fn dt_fri_config() -> FriConfig<InnerChallengeMmcs> {
    let perm = inner_perm();
    let hash = InnerHash::new(perm.clone());
    let compress = InnerCompress::new(perm.clone());
    let challenge_mmcs = InnerChallengeMmcs::new(InnerValMmcs::new(hash, compress));
    let num_queries = match std::env::var("FRI_QUERIES") {
        Ok(value) => value.parse().unwrap(),
        Err(_) => 193,
    };
    FriConfig {
        log_blowup: 1,
        num_queries,
        grinding_bits_query: 20,
        grinding_bits_batching: 10,
        mmcs: challenge_mmcs,
    }
}

/// The FRI config for inner recursion.
#[must_use]
pub fn inner_fri_config() -> FriConfig<InnerChallengeMmcs> {
    let perm = inner_perm();
    let hash = InnerHash::new(perm.clone());
    let compress = InnerCompress::new(perm.clone());
    let challenge_mmcs = InnerChallengeMmcs::new(InnerValMmcs::new(hash, compress));
    let num_queries = match std::env::var("FRI_QUERIES") {
        Ok(value) => value.parse().unwrap(),
        Err(_) => 193,
    };
    FriConfig {
        log_blowup: 1,
        num_queries,
        grinding_bits_query: 20,
        grinding_bits_batching: 10,
        mmcs: challenge_mmcs,
    }
}

/// The recursion config used for recursive reduce circuit.
#[derive(Deserialize)]
#[serde(from = "std::marker::PhantomData<KoalaBearPoseidon2Inner>")]
pub struct KoalaBearPoseidon2Inner {
    pub perm: InnerPerm,
    pub pcs: InnerPcs,
}

impl Clone for KoalaBearPoseidon2Inner {
    fn clone(&self) -> Self {
        Self::new()
    }
}

impl Serialize for KoalaBearPoseidon2Inner {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        std::marker::PhantomData::<KoalaBearPoseidon2Inner>.serialize(serializer)
    }
}

impl From<std::marker::PhantomData<KoalaBearPoseidon2Inner>> for KoalaBearPoseidon2Inner {
    fn from(_: std::marker::PhantomData<KoalaBearPoseidon2Inner>) -> Self {
        Self::new()
    }
}

impl KoalaBearPoseidon2Inner {
    #[must_use]
    pub fn new() -> Self {
        let perm = inner_perm();
        let hash = InnerHash::new(perm.clone());
        let compress = InnerCompress::new(perm.clone());
        let val_mmcs = InnerValMmcs::new(hash, compress);
        let dft = InnerDft {};
        let fri_config = inner_fri_config();
        let pcs = InnerPcs::new(27, dft, val_mmcs, fri_config);
        Self { perm, pcs }
    }
}

impl Default for KoalaBearPoseidon2Inner {
    fn default() -> Self {
        Self::new()
    }
}

impl StarkGenericConfig for KoalaBearPoseidon2Inner {
    type Val = InnerVal;
    type Domain = <InnerPcs as p3_commit::Pcs<InnerChallenge, InnerChallenger>>::Domain;
    type Pcs = InnerPcs;
    // type Challenge = InnerChallenge;
    type Challenger = InnerChallenger;

    fn pcs(&self) -> &Self::Pcs {
        &self.pcs
    }

    fn challenger(&self) -> Self::Challenger {
        InnerChallenger::new(self.perm.clone())
    }
}

impl ZeroCommitment<KoalaBearPoseidon2Inner> for InnerPcs {
    fn zero_commitment(&self) -> Com<KoalaBearPoseidon2Inner> {
        InnerDigestHash::from([InnerVal::zero(); DIGEST_SIZE])
    }
}

pub mod koala_bear_poseidon2 {

    use crate::sumcheck::config::{MlCom, SCStarkGenericConfig};
    use dt_primitives::{
        KoalaBear_BEGIN_EXT_CONSTS, KoalaBear_END_EXT_CONSTS, KoalaBear_PARTIAL_CONSTS,
    };
    use p3_challenger::DuplexChallenger;
    use p3_commit::ExtensionMmcs;
    use p3_dft::Radix2DitParallel;
    use p3_field::{extension::BinomialExtensionField, AbstractField, Field};
    use p3_fri::{FriConfig, TwoAdicFriPcs};
    use p3_koala_bear::{DiffusionMatrixKoalaBear, KoalaBear};
    use p3_merkle_tree::FieldMerkleTreeMmcs;
    use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
    use p3_symmetric::{Hash, PaddingFreeSponge, TruncatedPermutation};
    use basefold::basefold::basefold_pcs::BaseFoldPcs;
    use basefold::basefold::mlpcs::MlPCS;
    use serde::{Deserialize, Serialize};

    use crate::config::{Com, StarkGenericConfig, ZeroCommitment};
    use crate::DIGEST_SIZE;
    pub type Val = KoalaBear;
    pub type Challenge = BinomialExtensionField<Val, 4>;

    pub type Perm = Poseidon2<Val, Poseidon2ExternalMatrixGeneral, DiffusionMatrixKoalaBear, 16, 3>;
    pub type MyHash = PaddingFreeSponge<Perm, 16, 8, DIGEST_SIZE>;
    pub type DigestHash = Hash<Val, Val, DIGEST_SIZE>;
    pub type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    pub type ValMmcs = FieldMerkleTreeMmcs<
        <Val as Field>::Packing,
        <Val as Field>::Packing,
        MyHash,
        MyCompress,
        8,
    >;
    pub type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
    pub type Dft = Radix2DitParallel;
    pub type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
    type Pcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;
    pub type Mlpcs = BaseFoldPcs<Val, ValMmcs, ChallengeMmcs, Challenge, Challenger>;

    #[must_use]
    pub fn my_perm() -> Perm {
        const ROUNDS_F: usize = 8;
        const ROUNDS_P: usize = 20;
        let mut external_round_constants = KoalaBear_BEGIN_EXT_CONSTS.to_vec();
        let internal_round_constants = KoalaBear_PARTIAL_CONSTS.to_vec();
        external_round_constants.extend_from_slice(KoalaBear_END_EXT_CONSTS.as_slice());

        Perm::new(
            ROUNDS_F,
            external_round_constants,
            Poseidon2ExternalMatrixGeneral,
            ROUNDS_P,
            internal_round_constants,
            DiffusionMatrixKoalaBear,
        )
    }

    #[must_use]
    pub fn default_fri_config() -> FriConfig<ChallengeMmcs> {
        let perm = my_perm();
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());
        let challenge_mmcs = ChallengeMmcs::new(ValMmcs::new(hash, compress));
        let num_queries = match std::env::var("FRI_QUERIES") {
            Ok(value) => value.parse().unwrap(),
            Err(_) => 193,
        };
        FriConfig {
            log_blowup: 1,
            num_queries,
            grinding_bits_query: 20,
            grinding_bits_batching: 10,
            mmcs: challenge_mmcs,
        }
    }

    #[must_use]
    pub fn compressed_fri_config() -> FriConfig<ChallengeMmcs> {
        let perm = my_perm();
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());
        let challenge_mmcs = ChallengeMmcs::new(ValMmcs::new(hash, compress));
        let num_queries = match std::env::var("FRI_QUERIES") {
            Ok(value) => value.parse().unwrap(),
            Err(_) => 118,
        };
        FriConfig {
            log_blowup: 2,
            num_queries,
            grinding_bits_query: 20,
            grinding_bits_batching: 10,
            mmcs: challenge_mmcs,
        }
    }

    #[must_use]
    pub fn ultra_compressed_fri_config() -> FriConfig<ChallengeMmcs> {
        let perm = my_perm();
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());
        let challenge_mmcs = ChallengeMmcs::new(ValMmcs::new(hash, compress));
        let num_queries = match std::env::var("FRI_QUERIES") {
            Ok(value) => value.parse().unwrap(),
            Err(_) => 193,
        };
        FriConfig {
            log_blowup: 1,
            num_queries,
            grinding_bits_query: 20,
            grinding_bits_batching: 10,
            mmcs: challenge_mmcs,
        }
    }

    enum KoalaBearPoseidon2Type {
        Default,
        Compressed,
    }

    #[derive(Deserialize)]
    #[serde(from = "std::marker::PhantomData<SCKoalaBearPoseidon2>")]
    pub struct SCKoalaBearPoseidon2 {
        pub perm: Perm,
        pcs: Pcs,
        mlpcs: Mlpcs,
        config_type: KoalaBearPoseidon2Type,
    }

    impl SCKoalaBearPoseidon2 {
        #[must_use]
        pub fn new() -> Self {
            let perm = my_perm();
            let hash = MyHash::new(perm.clone());
            let compress = MyCompress::new(perm.clone());
            let val_mmcs = ValMmcs::new(hash, compress);
            let dft = Dft {};
            let fri_config = default_fri_config();
            let fri_config1 = default_fri_config();

            let pcs = Pcs::new(27, dft, val_mmcs.clone(), fri_config);
            let mlpcs = Mlpcs::new(val_mmcs, fri_config1);
            Self { pcs, mlpcs, perm, config_type: KoalaBearPoseidon2Type::Default }
        }

        #[must_use]
        pub fn compressed() -> Self {
            let perm = my_perm();
            let hash = MyHash::new(perm.clone());
            let compress = MyCompress::new(perm.clone());
            let val_mmcs = ValMmcs::new(hash, compress);
            let dft = Dft {};
            let fri_config = compressed_fri_config();
            let fri_config1 = compressed_fri_config();

            let pcs = Pcs::new(27, dft, val_mmcs.clone(), fri_config);
            let mlpcs = Mlpcs::new(val_mmcs, fri_config1);
            Self { pcs, mlpcs, perm, config_type: KoalaBearPoseidon2Type::Compressed }
        }

        #[must_use]
        pub fn ultra_compressed() -> Self {
            let perm = my_perm();
            let hash = MyHash::new(perm.clone());
            let compress = MyCompress::new(perm.clone());
            let val_mmcs = ValMmcs::new(hash, compress);
            let dft = Dft {};
            let fri_config = ultra_compressed_fri_config();
            let fri_config1 = ultra_compressed_fri_config();

            let pcs = Pcs::new(27, dft, val_mmcs.clone(), fri_config);
            let mlpcs = Mlpcs::new(val_mmcs, fri_config1);
            Self { pcs, mlpcs, perm, config_type: KoalaBearPoseidon2Type::Compressed }
        }
    }

    impl Clone for SCKoalaBearPoseidon2 {
        fn clone(&self) -> Self {
            match self.config_type {
                KoalaBearPoseidon2Type::Default => Self::new(),
                KoalaBearPoseidon2Type::Compressed => Self::compressed(),
            }
        }
    }

    impl Default for SCKoalaBearPoseidon2 {
        fn default() -> Self {
            Self::new()
        }
    }

    impl Serialize for SCKoalaBearPoseidon2 {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: serde::Serializer,
        {
            std::marker::PhantomData::<SCKoalaBearPoseidon2>.serialize(serializer)
        }
    }

    impl From<std::marker::PhantomData<SCKoalaBearPoseidon2>> for SCKoalaBearPoseidon2 {
        fn from(_: std::marker::PhantomData<SCKoalaBearPoseidon2>) -> Self {
            Self::new()
        }
    }

    impl StarkGenericConfig for SCKoalaBearPoseidon2 {
        type Val = KoalaBear;
        type Domain = <Pcs as p3_commit::Pcs<Challenge, Challenger>>::Domain;
        type Pcs = Pcs;
        type Challenger = Challenger;

        fn pcs(&self) -> &Self::Pcs {
            &self.pcs
        }

        fn challenger(&self) -> Self::Challenger {
            Challenger::new(self.perm.clone())
        }
    }

    impl ZeroCommitment<SCKoalaBearPoseidon2> for Pcs {
        fn zero_commitment(&self) -> Com<SCKoalaBearPoseidon2> {
            DigestHash::from([Val::zero(); DIGEST_SIZE])
        }
    }

    impl SCStarkGenericConfig for SCKoalaBearPoseidon2 {
        type Mlpcs = Mlpcs;
        type MlChallenge = <Mlpcs as MlPCS>::ExtensionField;
        type MlPcsProverData = <Mlpcs as MlPCS>::ProverData;
        type MlChallenger = <Mlpcs as MlPCS>::Challenger;

        fn mlpcs(&self) -> &Self::Mlpcs {
            &self.mlpcs
        }

        fn mlchallenger(&self) -> Self::MlChallenger {
            Challenger::new(self.perm.clone())
        }
    }

    impl ZeroCommitment<SCKoalaBearPoseidon2> for Mlpcs {
        fn zero_commitment(&self) -> MlCom<SCKoalaBearPoseidon2> {
            DigestHash::from([Val::zero(); DIGEST_SIZE])
        }
    }
}
