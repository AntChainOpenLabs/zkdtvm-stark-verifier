use super::{dt_dev_mode, poseidon2::bn254_poseidon2_rc3};
use dt_primitives::SCField;
use dt_stark::{
    sumcheck::config::{MlCom, SCStarkGenericConfig},
    Com, StarkGenericConfig, ZeroCommitment,
};
use p3_baby_bear::BabyBear;
use p3_bn254_fr::{Bn254Fr, DiffusionMatrixBN254};
use p3_challenger::MultiField32Challenger;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::{extension::BinomialExtensionField, AbstractField};
use p3_fri::{
    BatchOpening, CommitPhaseProofStep, FriConfig, FriProof, QueryProof, TwoAdicFriPcs,
    TwoAdicFriPcsProof,
};
use p3_merkle_tree::FieldMerkleTreeMmcs;
use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
use p3_symmetric::{Hash, MultiField32PaddingFreeSponge, TruncatedPermutation};
use pcs::basefold::{
    basefold_pcs::{BaseFoldPcs, BasefoldConfig},
    mlpcs::{MlCommitOptions, MlPCS},
};
use serde::{Deserialize, Serialize};

pub const DIGEST_SIZE: usize = 1;

pub const OUTER_MULTI_FIELD_CHALLENGER_WIDTH: usize = 3;
pub const OUTER_MULTI_FIELD_CHALLENGER_RATE: usize = 2;
pub const OUTER_MULTI_FIELD_CHALLENGER_DIGEST_SIZE: usize = 1;
pub const WHIR_OUTER_ROUND_QUERY_COUNTS: &[usize] = &[25, 20, 16, 12, 10, 8, 8, 8];

fn env_var(var: &str) -> Result<String, std::env::VarError> {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = var;
        Err(std::env::VarError::NotPresent)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::env::var(var)
    }
}

fn parse_whir_round_queries(var: &str, value: &str) -> Vec<usize> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }
    trimmed
        .split(',')
        .map(|part| {
            let queries = part
                .trim()
                .parse::<usize>()
                .unwrap_or_else(|_| panic!("{var} must be comma-separated usize values"));
            assert!(queries > 0, "{var} entries must be positive");
            queries
        })
        .collect()
}

fn whir_round_query_counts_from_env(default: &[usize]) -> Vec<usize> {
    env_var("WHIR_OUTER_ROUND_QUERIES")
        .ok()
        .map(|value| parse_whir_round_queries("WHIR_OUTER_ROUND_QUERIES", &value))
        .unwrap_or_else(|| default.to_vec())
}

fn outer_whir_config<M>(fri_config: FriConfig<M>) -> BasefoldConfig<M> {
    BasefoldConfig::new(fri_config)
        .with_round_query_counts(whir_round_query_counts_from_env(WHIR_OUTER_ROUND_QUERY_COUNTS))
        .with_path_pruning_from_env()
}

/// A configuration for outer recursion.
pub type OuterVal = BabyBear;
pub type OuterChallenge = BinomialExtensionField<OuterVal, 4>;
pub type OuterPerm = Poseidon2<Bn254Fr, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBN254, 3, 5>;
pub type OuterHash =
    MultiField32PaddingFreeSponge<OuterVal, Bn254Fr, OuterPerm, 3, 16, DIGEST_SIZE>;
pub type OuterDigestHash = Hash<OuterVal, Bn254Fr, DIGEST_SIZE>;
pub type OuterDigest = [Bn254Fr; DIGEST_SIZE];
pub type OuterCompress = TruncatedPermutation<OuterPerm, 2, 1, 3>;
pub type OuterValMmcs = FieldMerkleTreeMmcs<BabyBear, Bn254Fr, OuterHash, OuterCompress, 1>;
pub type OuterChallengeMmcs = ExtensionMmcs<OuterVal, OuterChallenge, OuterValMmcs>;
pub type OuterDft = Radix2DitParallel;
pub type OuterChallenger = MultiField32Challenger<
    OuterVal,
    Bn254Fr,
    OuterPerm,
    OUTER_MULTI_FIELD_CHALLENGER_WIDTH,
    OUTER_MULTI_FIELD_CHALLENGER_RATE,
>;
pub type OuterPcs = TwoAdicFriPcs<OuterVal, OuterDft, OuterValMmcs, OuterChallengeMmcs>;

pub type OuterQueryProof = QueryProof<OuterChallenge, OuterChallengeMmcs>;
pub type OuterCommitPhaseStep = CommitPhaseProofStep<OuterChallenge, OuterChallengeMmcs>;
pub type OuterFriProof = FriProof<OuterChallenge, OuterChallengeMmcs, OuterVal>;
pub type OuterBatchOpening = BatchOpening<OuterVal, OuterValMmcs>;
pub type OuterPcsProof =
    TwoAdicFriPcsProof<OuterVal, OuterChallenge, OuterValMmcs, OuterChallengeMmcs>;
use pcs::basefold::basefold_pcs::{BasefoldInputProof, BasefoldProof};
pub type OuterBasefoldProof = BasefoldProof<
    OuterChallenge,
    OuterChallengeMmcs,
    OuterVal,
    BasefoldInputProof<OuterVal, OuterValMmcs>,
>;

/// The permutation for outer recursion.
pub fn outer_perm() -> OuterPerm {
    const ROUNDS_F: usize = 8;
    const ROUNDS_P: usize = 56;
    let mut round_constants = bn254_poseidon2_rc3();
    let internal_start = ROUNDS_F / 2;
    let internal_end = (ROUNDS_F / 2) + ROUNDS_P;
    let internal_round_constants =
        round_constants.drain(internal_start..internal_end).map(|vec| vec[0]).collect::<Vec<_>>();
    let external_round_constants = round_constants;
    OuterPerm::new(
        ROUNDS_F,
        external_round_constants,
        Poseidon2ExternalMatrixGeneral,
        ROUNDS_P,
        internal_round_constants,
        DiffusionMatrixBN254,
    )
}

/// The FRI config for outer recursion.
pub fn outer_fri_config() -> FriConfig<OuterChallengeMmcs> {
    let perm = outer_perm();
    let hash = OuterHash::new(perm.clone()).unwrap();
    let compress = OuterCompress::new(perm.clone());
    let challenge_mmcs = OuterChallengeMmcs::new(OuterValMmcs::new(hash, compress));
    let num_queries = if dt_dev_mode() {
        1
    } else {
        match env_var("FRI_QUERIES") {
            Ok(value) => value.parse().unwrap(),
            Err(_) => 25,
        }
    };
    FriConfig {
        log_blowup: 4,
        num_queries,
        grinding_bits_query: 16,
        grinding_bits_batching: 16,
        grinding_bits_folding: 0,
        log_final_poly_len: 5,
        cross_round_log_foldings: Vec::new(),
        num_committed_groups: None,
        mmcs: challenge_mmcs,
    }
}

/// The FRI config for outer recursion.
pub fn outer_fri_config_with_blowup(log_blowup: usize) -> FriConfig<OuterChallengeMmcs> {
    let perm = outer_perm();
    let hash = OuterHash::new(perm.clone()).unwrap();
    let compress = OuterCompress::new(perm.clone());
    let challenge_mmcs = OuterChallengeMmcs::new(OuterValMmcs::new(hash, compress));
    let (default_queries, pow_bits) = match log_blowup {
        1 => (100, 24),
        2 => (50, 24),
        3 => (33, 20),
        _ => (100 / log_blowup, 16),
    };
    let num_queries = if dt_dev_mode() {
        1
    } else {
        match env_var("FRI_QUERIES") {
            Ok(value) => value.parse().unwrap(),
            Err(_) => default_queries,
        }
    };
    let batching_bits = grinding_bits_batching_for_blowup(log_blowup);
    FriConfig {
        log_blowup,
        num_queries,
        grinding_bits_query: pow_bits,
        grinding_bits_batching: batching_bits,
        grinding_bits_folding: 0,
        log_final_poly_len: 5,
        cross_round_log_foldings: Vec::new(),
        num_committed_groups: None,
        mmcs: challenge_mmcs,
    }
}

#[derive(Deserialize)]
#[serde(from = "std::marker::PhantomData<BabyBearPoseidon2Outer>")]
pub struct BabyBearPoseidon2Outer {
    pub perm: OuterPerm,
    pub pcs: OuterPcs,
}

impl Clone for BabyBearPoseidon2Outer {
    fn clone(&self) -> Self {
        Self::new()
    }
}

impl Serialize for BabyBearPoseidon2Outer {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        std::marker::PhantomData::<BabyBearPoseidon2Outer>.serialize(serializer)
    }
}

impl From<std::marker::PhantomData<BabyBearPoseidon2Outer>> for BabyBearPoseidon2Outer {
    fn from(_: std::marker::PhantomData<BabyBearPoseidon2Outer>) -> Self {
        Self::new()
    }
}

impl BabyBearPoseidon2Outer {
    pub fn new() -> Self {
        let perm = outer_perm();
        let hash = OuterHash::new(perm.clone()).unwrap();
        let compress = OuterCompress::new(perm.clone());
        let val_mmcs = OuterValMmcs::new(hash, compress);
        let dft = OuterDft {};
        let fri_config = outer_fri_config();
        let pcs = OuterPcs::new(27, dft, val_mmcs, fri_config);
        Self { pcs, perm }
    }
    pub fn new_with_log_blowup(log_blowup: usize) -> Self {
        let perm = outer_perm();
        let hash = OuterHash::new(perm.clone()).unwrap();
        let compress = OuterCompress::new(perm.clone());
        let val_mmcs = OuterValMmcs::new(hash, compress);
        let dft = OuterDft {};
        let fri_config = outer_fri_config_with_blowup(log_blowup);
        let pcs = OuterPcs::new(27, dft, val_mmcs, fri_config);
        Self { pcs, perm }
    }
}

impl Default for BabyBearPoseidon2Outer {
    fn default() -> Self {
        Self::new()
    }
}

impl StarkGenericConfig for BabyBearPoseidon2Outer {
    type Val = OuterVal;
    type Challenge = OuterChallenge;
    type Domain = <OuterPcs as p3_commit::Pcs<OuterChallenge, OuterChallenger>>::Domain;
    type Pcs = OuterPcs;
    // type PackExt = OuterChallenge;
    type Challenger = OuterChallenger;

    fn pcs(&self) -> &Self::Pcs {
        &self.pcs
    }

    fn challenger(&self) -> Self::Challenger {
        OuterChallenger::new(self.perm.clone()).unwrap()
    }
}

impl ZeroCommitment<BabyBearPoseidon2Outer> for OuterPcs {
    fn zero_commitment(&self) -> Com<BabyBearPoseidon2Outer> {
        OuterDigestHash::from([Bn254Fr::zero(); DIGEST_SIZE])
    }
}

/// The FRI config for testing recursion.
pub fn test_fri_config() -> FriConfig<OuterChallengeMmcs> {
    let perm = outer_perm();
    let hash = OuterHash::new(perm.clone()).unwrap();
    let compress = OuterCompress::new(perm.clone());
    let challenge_mmcs = OuterChallengeMmcs::new(OuterValMmcs::new(hash, compress));
    FriConfig {
        log_blowup: 1,
        num_queries: 1,
        grinding_bits_query: 1,
        grinding_bits_batching: 1,
        grinding_bits_folding: 0,
        log_final_poly_len: 5,
        cross_round_log_foldings: Vec::new(),
        num_committed_groups: None,
        mmcs: challenge_mmcs,
    }
}

pub type OuterMlpcs =
    BaseFoldPcs<OuterVal, OuterValMmcs, OuterChallengeMmcs, OuterChallenge, OuterChallenger>;

#[derive(Deserialize)]
#[serde(from = "std::marker::PhantomData<SCBabyBearPoseidon2Outer>")]
pub struct SCBabyBearPoseidon2Outer {
    pub perm: OuterPerm,
    pcs: OuterPcs,
    mlpcs: OuterMlpcs,
}
impl SCBabyBearPoseidon2Outer {
    pub fn new() -> Self {
        let perm = outer_perm();
        let hash = OuterHash::new(perm.clone()).unwrap();
        let compress = OuterCompress::new(perm.clone());
        let val_mmcs = OuterValMmcs::new(hash, compress);
        let dft = OuterDft {};
        let fri_config_pcs = outer_fri_config();
        let fri_config_basefold = outer_fri_config();
        let pcs = OuterPcs::new(27, dft, val_mmcs.clone(), fri_config_pcs);
        let mlpcs =
            OuterMlpcs::from_config(val_mmcs.clone(), outer_whir_config(fri_config_basefold));
        Self { perm, pcs, mlpcs }
    }

    pub fn new_with_log_blowup(log_blowup: usize) -> Self {
        let perm = outer_perm();
        let hash = OuterHash::new(perm.clone()).unwrap();
        let compress = OuterCompress::new(perm.clone());
        let val_mmcs = OuterValMmcs::new(hash, compress);
        let dft = OuterDft {};
        let fri_config_pcs = outer_fri_config_with_blowup(log_blowup);
        let fri_config_basefold = outer_fri_config_with_blowup(log_blowup);
        let pcs = OuterPcs::new(27, dft, val_mmcs.clone(), fri_config_pcs);
        let mlpcs =
            OuterMlpcs::from_config(val_mmcs.clone(), outer_whir_config(fri_config_basefold));
        Self { pcs, perm, mlpcs }
    }
}
impl Default for SCBabyBearPoseidon2Outer {
    fn default() -> Self {
        Self::new()
    }
}
impl Clone for SCBabyBearPoseidon2Outer {
    fn clone(&self) -> Self {
        Self::new()
    }
}
impl Serialize for SCBabyBearPoseidon2Outer {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        std::marker::PhantomData::<SCBabyBearPoseidon2Outer>.serialize(serializer)
    }
}

impl From<std::marker::PhantomData<SCBabyBearPoseidon2Outer>> for SCBabyBearPoseidon2Outer {
    fn from(_: std::marker::PhantomData<SCBabyBearPoseidon2Outer>) -> Self {
        Self::new()
    }
}
impl StarkGenericConfig for SCBabyBearPoseidon2Outer {
    type Val = OuterVal;
    type Challenge = OuterChallenge;
    type Domain = <OuterPcs as p3_commit::Pcs<OuterChallenge, OuterChallenger>>::Domain;
    type Pcs = OuterPcs;
    // type PackExt = OuterChallenge;
    type Challenger = OuterChallenger;

    fn pcs(&self) -> &Self::Pcs {
        &self.pcs
    }

    fn challenger(&self) -> Self::Challenger {
        OuterChallenger::new(self.perm.clone()).unwrap()
    }
}
impl SCStarkGenericConfig for SCBabyBearPoseidon2Outer {
    type Mlpcs = OuterMlpcs;
    type MlChallenge = OuterChallenge;
    type MlPcsProverData = <OuterMlpcs as MlPCS>::ProverData;
    type MlChallenger = OuterChallenger;

    fn mlpcs(&self) -> &Self::Mlpcs {
        &self.mlpcs
    }

    fn mlchallenger(&self) -> Self::MlChallenger {
        OuterChallenger::new(self.perm.clone()).unwrap()
    }
}
impl ZeroCommitment<SCBabyBearPoseidon2Outer> for OuterPcs {
    fn zero_commitment(&self) -> Com<SCBabyBearPoseidon2Outer> {
        OuterDigestHash::from([Bn254Fr::zero(); DIGEST_SIZE])
    }
}

impl ZeroCommitment<SCBabyBearPoseidon2Outer> for OuterMlpcs {
    fn zero_commitment(&self) -> MlCom<SCBabyBearPoseidon2Outer> {
        OuterDigestHash::from([Bn254Fr::zero(); DIGEST_SIZE])
    }
}

// --- KoalaBear Outer types for SC Prover wrap stage ---

pub type SCOuterVal = SCField;
pub type SCOuterChallenge = BinomialExtensionField<SCOuterVal, 4>;
pub type SCOuterHash =
    MultiField32PaddingFreeSponge<SCOuterVal, Bn254Fr, OuterPerm, 3, 16, DIGEST_SIZE>;
pub type SCOuterDigestHash = Hash<SCOuterVal, Bn254Fr, DIGEST_SIZE>;
pub type SCOuterCompress = TruncatedPermutation<OuterPerm, 2, 1, 3>;
pub type SCOuterValMmcs = FieldMerkleTreeMmcs<SCField, Bn254Fr, SCOuterHash, SCOuterCompress, 1>;
pub type SCOuterBatchOpening = BatchOpening<SCOuterVal, SCOuterValMmcs>;
pub type SCOuterBasefoldProof = BasefoldProof<
    SCOuterChallenge,
    SCOuterChallengeMmcs,
    SCOuterVal,
    BasefoldInputProof<SCOuterVal, SCOuterValMmcs>,
>;

fn outer_stack_log_height_from_env() -> Option<usize> {
    ["WHIR_OUTER_STACK_LOG_HEIGHT", "WHIR_STACK_LOG_HEIGHT"].iter().find_map(|var| {
        env_var(var).ok().map(|value| {
            value.parse::<usize>().unwrap_or_else(|_| panic!("{var} must be a valid usize"))
        })
    })
}
pub type SCOuterChallengeMmcs = ExtensionMmcs<SCOuterVal, SCOuterChallenge, SCOuterValMmcs>;
pub type SCOuterPcs = TwoAdicFriPcs<SCOuterVal, OuterDft, SCOuterValMmcs, SCOuterChallengeMmcs>;
pub type SCOuterChallenger = MultiField32Challenger<
    SCOuterVal,
    Bn254Fr,
    OuterPerm,
    OUTER_MULTI_FIELD_CHALLENGER_WIDTH,
    OUTER_MULTI_FIELD_CHALLENGER_RATE,
>;

pub type SCOuterMlpcs = BaseFoldPcs<
    SCOuterVal,
    SCOuterValMmcs,
    SCOuterChallengeMmcs,
    SCOuterChallenge,
    SCOuterChallenger,
>;

pub const SC_OUTER_WHIR_NUM_COMMITTED_GROUPS: usize = 4;

fn sc_outer_fri_config() -> FriConfig<SCOuterChallengeMmcs> {
    let perm = outer_perm();
    let hash = SCOuterHash::new(perm.clone()).unwrap();
    let compress = SCOuterCompress::new(perm.clone());
    let challenge_mmcs = SCOuterChallengeMmcs::new(SCOuterValMmcs::new(hash, compress));
    let num_queries = if dt_dev_mode() {
        1
    } else {
        match env_var("FRI_QUERIES") {
            Ok(value) => value.parse().unwrap(),
            Err(_) => 25,
        }
    };
    FriConfig {
        log_blowup: 4,
        num_queries,
        grinding_bits_query: 16,
        grinding_bits_batching: 16,
        grinding_bits_folding: 0,
        log_final_poly_len: 6,
        cross_round_log_foldings: Vec::new(),
        num_committed_groups: Some(SC_OUTER_WHIR_NUM_COMMITTED_GROUPS),
        mmcs: challenge_mmcs,
    }
}

fn sc_outer_fri_config_with_blowup(log_blowup: usize) -> FriConfig<SCOuterChallengeMmcs> {
    let perm = outer_perm();
    let hash = SCOuterHash::new(perm.clone()).unwrap();
    let compress = SCOuterCompress::new(perm.clone());
    let challenge_mmcs = SCOuterChallengeMmcs::new(SCOuterValMmcs::new(hash, compress));
    let (default_queries, pow_bits) = match log_blowup {
        1 => (100, 24),
        2 => (50, 24),
        3 => (33, 20),
        _ => (100 / log_blowup, 16),
    };
    let num_queries = if dt_dev_mode() {
        1
    } else {
        match env_var("FRI_QUERIES") {
            Ok(value) => value.parse().unwrap(),
            Err(_) => default_queries,
        }
    };
    let batching_bits = grinding_bits_batching_for_blowup(log_blowup);
    FriConfig {
        log_blowup,
        num_queries,
        grinding_bits_query: pow_bits,
        grinding_bits_batching: batching_bits,
        grinding_bits_folding: 0,
        log_final_poly_len: 6,
        cross_round_log_foldings: Vec::new(),
        num_committed_groups: Some(SC_OUTER_WHIR_NUM_COMMITTED_GROUPS),
        mmcs: challenge_mmcs,
    }
}

fn grinding_bits_batching_for_blowup(log_blowup: usize) -> usize {
    match log_blowup {
        1 => 10,
        2 => 10,
        3 => 6,
        _ => 16,
    }
}

#[derive(Deserialize)]
#[serde(from = "std::marker::PhantomData<SCKoalaBearPoseidon2Outer>")]
pub struct SCKoalaBearPoseidon2Outer {
    pub perm: OuterPerm,
    pcs: SCOuterPcs,
    mlpcs: SCOuterMlpcs,
}

impl SCKoalaBearPoseidon2Outer {
    pub fn new() -> Self {
        let perm = outer_perm();
        let hash = SCOuterHash::new(perm.clone()).unwrap();
        let compress = SCOuterCompress::new(perm.clone());
        let val_mmcs = SCOuterValMmcs::new(hash, compress);
        let dft = OuterDft {};
        let fri_config_pcs = sc_outer_fri_config();
        let fri_config_basefold = sc_outer_fri_config();
        let pcs = SCOuterPcs::new(27, dft, val_mmcs.clone(), fri_config_pcs);
        let mlpcs = SCOuterMlpcs::from_config(val_mmcs, outer_whir_config(fri_config_basefold));
        Self { perm, pcs, mlpcs }
    }

    pub fn new_with_log_blowup(log_blowup: usize) -> Self {
        let perm = outer_perm();
        let hash = SCOuterHash::new(perm.clone()).unwrap();
        let compress = SCOuterCompress::new(perm.clone());
        let val_mmcs = SCOuterValMmcs::new(hash, compress);
        let dft = OuterDft {};
        let fri_config_pcs = sc_outer_fri_config_with_blowup(log_blowup);
        let fri_config_basefold = sc_outer_fri_config_with_blowup(log_blowup);
        let pcs = SCOuterPcs::new(27, dft, val_mmcs.clone(), fri_config_pcs);
        let mlpcs = SCOuterMlpcs::from_config(val_mmcs, outer_whir_config(fri_config_basefold));
        Self { pcs, perm, mlpcs }
    }

    pub fn cross_round_log_foldings(&self) -> Vec<usize> {
        self.mlpcs.cross_round_log_foldings()
    }
}

impl Default for SCKoalaBearPoseidon2Outer {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for SCKoalaBearPoseidon2Outer {
    fn clone(&self) -> Self {
        Self::new()
    }
}

impl Serialize for SCKoalaBearPoseidon2Outer {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        std::marker::PhantomData::<SCKoalaBearPoseidon2Outer>.serialize(serializer)
    }
}

impl From<std::marker::PhantomData<SCKoalaBearPoseidon2Outer>> for SCKoalaBearPoseidon2Outer {
    fn from(_: std::marker::PhantomData<SCKoalaBearPoseidon2Outer>) -> Self {
        Self::new()
    }
}

impl StarkGenericConfig for SCKoalaBearPoseidon2Outer {
    type Val = SCOuterVal;
    type Challenge = SCOuterChallenge;
    type Domain = <SCOuterPcs as p3_commit::Pcs<SCOuterChallenge, SCOuterChallenger>>::Domain;
    type Pcs = SCOuterPcs;
    type Challenger = SCOuterChallenger;

    fn pcs(&self) -> &Self::Pcs {
        &self.pcs
    }

    fn challenger(&self) -> Self::Challenger {
        SCOuterChallenger::new(self.perm.clone()).unwrap()
    }
}

impl SCStarkGenericConfig for SCKoalaBearPoseidon2Outer {
    type Mlpcs = SCOuterMlpcs;
    type MlChallenge = SCOuterChallenge;
    type MlPcsProverData = <SCOuterMlpcs as MlPCS>::ProverData;
    type MlChallenger = SCOuterChallenger;

    fn mlpcs(&self) -> &Self::Mlpcs {
        &self.mlpcs
    }

    fn mlpcs_commit_options(&self) -> MlCommitOptions {
        MlCommitOptions::auto_stacking()
    }

    fn mlpcs_stack_log_height_hint(&self) -> Option<usize> {
        outer_stack_log_height_from_env()
    }

    fn mlchallenger(&self) -> Self::MlChallenger {
        SCOuterChallenger::new(self.perm.clone()).unwrap()
    }
}

impl ZeroCommitment<SCKoalaBearPoseidon2Outer> for SCOuterPcs {
    fn zero_commitment(&self) -> Com<SCKoalaBearPoseidon2Outer> {
        SCOuterDigestHash::from([Bn254Fr::zero(); DIGEST_SIZE])
    }
}

impl ZeroCommitment<SCKoalaBearPoseidon2Outer> for SCOuterMlpcs {
    fn zero_commitment(&self) -> MlCom<SCKoalaBearPoseidon2Outer> {
        SCOuterDigestHash::from([Bn254Fr::zero(); DIGEST_SIZE])
    }
}
