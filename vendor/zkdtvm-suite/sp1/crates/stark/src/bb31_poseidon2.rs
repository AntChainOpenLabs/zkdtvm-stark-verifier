#![allow(missing_docs)]
use crate::config::{Com, StarkGenericConfig, ZeroCommitment};
use dt_primitives::poseidon2_init;
use p3_baby_bear::{BabyBear, DiffusionMatrixBabyBear};
use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::{extension::BinomialExtensionField, AbstractField, Field};
use p3_fri::{
    BatchOpening, CommitPhaseProofStep, FriConfig, FriProof, QueryProof, TwoAdicFriPcs,
    TwoAdicFriPcsProof,
};
use p3_merkle_tree::FieldMerkleTreeMmcs;
use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
use p3_symmetric::{Hash, PaddingFreeSponge, TruncatedPermutation};
use pcs::basefold::basefold_pcs::{BasefoldInputProof, BasefoldProof};
use serde::{Deserialize, Serialize};

pub const DIGEST_SIZE: usize = 8;

// ──── Runtime JSON Configuration (BabyBear) ────

use crate::koalabear_poseidon2::WhirJsonConfig;

static BB_CONFIG: std::sync::OnceLock<WhirJsonConfig> = std::sync::OnceLock::new();

pub fn babybear_config() -> &'static WhirJsonConfig {
    BB_CONFIG.get_or_init(|| {
        // Select the WHIR parameter file matching the active challenge
        // extension degree. WHIR_CONFIG_PATH can still override this.
        #[cfg(not(feature = "ext5"))]
        let default_name = "whir_config_babybear_ext4.json";
        #[cfg(feature = "ext5")]
        let default_name = "whir_config_babybear_ext5.json";
        let path = std::env::var("WHIR_CONFIG_PATH").unwrap_or_else(|_| {
            if let Ok(mut dir) = std::env::current_dir() {
                loop {
                    let candidate = dir.join(default_name);
                    if candidate.is_file() {
                        return candidate.to_string_lossy().into_owned();
                    }
                    if !dir.pop() {
                        break;
                    }
                }
            }
            default_name.to_string()
        });
        match std::fs::read_to_string(&path) {
            Ok(contents) => {
                let cfg: WhirJsonConfig = serde_json::from_str(&contents)
                    .unwrap_or_else(|e| panic!("Failed to parse {path}: {e}"));
                tracing::info!("Loaded BabyBear WHIR config from {path}");
                cfg
            }
            Err(_) => WhirJsonConfig::default(),
        }
    })
}

/// A configuration for inner recursion.
pub type InnerVal = BabyBear;
pub type InnerChallenge = BinomialExtensionField<InnerVal, 4>;
pub type InnerPerm =
    Poseidon2<InnerVal, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBabyBear, 16, 7>;
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
pub type InnerBasefoldProof = BasefoldProof<
    InnerChallenge,
    InnerChallengeMmcs,
    InnerVal,
    BasefoldInputProof<InnerVal, InnerValMmcs>,
>;

/// The permutation for inner recursion.
#[must_use]
pub fn inner_perm() -> InnerPerm {
    poseidon2_init()
}

/// The FRI config for dt proofs.
#[must_use]
pub fn dt_fri_config() -> FriConfig<InnerChallengeMmcs> {
    let cfg = babybear_config().stage("core");
    let perm = inner_perm();
    let hash = InnerHash::new(perm.clone());
    let compress = InnerCompress::new(perm.clone());
    let challenge_mmcs = InnerChallengeMmcs::new(InnerValMmcs::new(hash, compress));
    let num_queries = match std::env::var("FRI_QUERIES") {
        Ok(value) => value.parse().unwrap(),
        Err(_) => cfg.num_queries.unwrap_or(100),
    };
    FriConfig {
        log_blowup: cfg.log_blowup.unwrap_or(1),
        num_queries,
        grinding_bits_query: cfg.grinding_bits_query.unwrap_or(24),
        grinding_bits_batching: cfg.grinding_bits_batching.unwrap_or(10),
        grinding_bits_folding: cfg.grinding_bits_folding.unwrap_or(0),
        log_final_poly_len: cfg.log_final_poly_len.unwrap_or(5),
        cross_round_log_foldings: Vec::new(),
        num_committed_groups: cfg.num_committed_groups.map(Some).unwrap_or(None),
        mmcs: challenge_mmcs,
    }
}

/// The FRI config for inner recursion.
#[must_use]
pub fn inner_fri_config() -> FriConfig<InnerChallengeMmcs> {
    let cfg = babybear_config().stage("compress");
    let perm = inner_perm();
    let hash = InnerHash::new(perm.clone());
    let compress = InnerCompress::new(perm.clone());
    let challenge_mmcs = InnerChallengeMmcs::new(InnerValMmcs::new(hash, compress));
    let num_queries = match std::env::var("FRI_QUERIES") {
        Ok(value) => value.parse().unwrap(),
        Err(_) => cfg.num_queries.unwrap_or(100),
    };
    FriConfig {
        log_blowup: cfg.log_blowup.unwrap_or(1),
        num_queries,
        grinding_bits_query: cfg.grinding_bits_query.unwrap_or(24),
        grinding_bits_batching: cfg.grinding_bits_batching.unwrap_or(10),
        grinding_bits_folding: cfg.grinding_bits_folding.unwrap_or(0),
        log_final_poly_len: cfg.log_final_poly_len.unwrap_or(5),
        cross_round_log_foldings: Vec::new(),
        num_committed_groups: cfg.num_committed_groups.map(Some).unwrap_or(None),
        mmcs: challenge_mmcs,
    }
}

/// The recursion config used for recursive reduce circuit.
#[derive(Deserialize)]
#[serde(from = "std::marker::PhantomData<BabyBearPoseidon2Inner>")]
pub struct BabyBearPoseidon2Inner {
    pub perm: InnerPerm,
    pub pcs: InnerPcs,
}

impl Clone for BabyBearPoseidon2Inner {
    fn clone(&self) -> Self {
        Self::new()
    }
}

impl Serialize for BabyBearPoseidon2Inner {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        std::marker::PhantomData::<BabyBearPoseidon2Inner>.serialize(serializer)
    }
}

impl From<std::marker::PhantomData<BabyBearPoseidon2Inner>> for BabyBearPoseidon2Inner {
    fn from(_: std::marker::PhantomData<BabyBearPoseidon2Inner>) -> Self {
        Self::new()
    }
}

impl BabyBearPoseidon2Inner {
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

impl Default for BabyBearPoseidon2Inner {
    fn default() -> Self {
        Self::new()
    }
}

impl StarkGenericConfig for BabyBearPoseidon2Inner {
    type Val = InnerVal;
    type Domain = <InnerPcs as p3_commit::Pcs<InnerChallenge, InnerChallenger>>::Domain;
    type Pcs = InnerPcs;
    type Challenge = InnerChallenge;
    type Challenger = InnerChallenger;

    fn pcs(&self) -> &Self::Pcs {
        &self.pcs
    }

    fn challenger(&self) -> Self::Challenger {
        InnerChallenger::new(self.perm.clone())
    }
}

impl ZeroCommitment<BabyBearPoseidon2Inner> for InnerPcs {
    fn zero_commitment(&self) -> Com<BabyBearPoseidon2Inner> {
        InnerDigestHash::from([InnerVal::zero(); DIGEST_SIZE])
    }
}

pub mod baby_bear_poseidon2 {

    use crate::sumcheck::config::{MlCom, SCStarkGenericConfig};
    use dt_primitives::RC_16_30;
    use p3_baby_bear::{BabyBear, DiffusionMatrixBabyBear};
    use p3_challenger::DuplexChallenger;
    use p3_commit::ExtensionMmcs;
    use p3_dft::Radix2DitParallel;
    use p3_field::{extension::BinomialExtensionField, AbstractField, Field};
    use p3_fri::{FriConfig, TwoAdicFriPcs};
    use p3_merkle_tree::FieldMerkleTreeMmcs;
    use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
    use p3_symmetric::{Hash, PaddingFreeSponge, TruncatedPermutation};
    use pcs::basefold::{
        basefold_pcs::{BaseFoldPcs, BasefoldConfig},
        mlpcs::{MlCommitOptions, MlPCS},
    };
    use serde::{Deserialize, Serialize};

    use crate::{
        config::{Com, StarkGenericConfig, ZeroCommitment},
        DIGEST_SIZE,
    };
    pub type Val = BabyBear;
    pub type Challenge = BinomialExtensionField<Val, 4>;

    pub type Perm = Poseidon2<Val, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBabyBear, 16, 7>;
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
        const ROUNDS_P: usize = 13;
        let mut round_constants = RC_16_30.to_vec();
        let internal_start = ROUNDS_F / 2;
        let internal_end = (ROUNDS_F / 2) + ROUNDS_P;
        let internal_round_constants = round_constants
            .drain(internal_start..internal_end)
            .map(|vec| vec[0])
            .collect::<Vec<_>>();
        let external_round_constants = round_constants;
        Perm::new(
            ROUNDS_F,
            external_round_constants,
            Poseidon2ExternalMatrixGeneral,
            ROUNDS_P,
            internal_round_constants,
            DiffusionMatrixBabyBear,
        )
    }

    /// Number of committed IOPP groups for the WHIR cross-round schedule.
    /// BabyBear mirrors the KoalaBear layout: 4 groups for core/compress,
    /// 3 for shrink/root-shrink.
    pub const WHIR_NUM_COMMITTED_GROUPS: usize = 4;
    pub const WHIR_SHRINK_NUM_COMMITTED_GROUPS: usize = 3;
    pub const WHIR_ROOT_SHRINK_NUM_COMMITTED_GROUPS: usize = 3;

    /// Per-round WHIR query counts (defaults; overridable via JSON config).
    pub const WHIR_CORE_ROUND_QUERY_COUNTS: &[usize] = &[100, 100, 100, 100];
    pub const WHIR_COMPRESS_ROUND_QUERY_COUNTS: &[usize] = &[50, 50, 50, 50];
    pub const WHIR_SHRINK_ROUND_QUERY_COUNTS: &[usize] = &[33, 33, 33];
    pub const WHIR_ROOT_SHRINK_ROUND_QUERY_COUNTS: &[usize] = &[25, 25, 25];

    /// Force `log_final_poly_len = 0` for non-stacking (Jagged) stages so the
    /// native WHIR config and the recursion circuit verifier
    /// (`machine.config().fri_config()`) agree: non-stacking emits no FRI
    /// early-stop final polynomial.
    fn apply_non_stacking_fri(
        mut fri_config: FriConfig<ChallengeMmcs>,
        stage: &str,
    ) -> FriConfig<ChallengeMmcs> {
        if !super::babybear_config().stacking_enabled(stage) {
            fri_config.log_final_poly_len = 0;
        }
        fri_config
    }

    fn whir_config_from_fri(
        fri_config: FriConfig<ChallengeMmcs>,
        stage: &str,
        default_queries: &[usize],
    ) -> BasefoldConfig<ChallengeMmcs> {
        let path_pruning = super::babybear_config().path_pruning_enabled_for_stage(stage);

        // Non-stacking (Jagged) path: single global arity-2 query phase driven
        // by `num_queries`, no early-stop. Force `log_final_poly_len = 0` and
        // leave the per-round query schedule unset so the stacked-only
        // parameters are ignored. Early-stop is a stacking-only optimization.
        if !super::babybear_config().stacking_enabled(stage) {
            let mut fri_config = fri_config;
            fri_config.log_final_poly_len = 0;
            return BasefoldConfig::new(fri_config).with_path_pruning(path_pruning);
        }

        let cfg = super::babybear_config().stage(stage);
        let queries = cfg.round_query_counts.clone().unwrap_or_else(|| default_queries.to_vec());
        BasefoldConfig::new(fri_config)
            .with_round_query_counts(queries)
            .with_path_pruning(path_pruning)
    }

    #[must_use]
    pub fn default_fri_config() -> FriConfig<ChallengeMmcs> {
        let perm = my_perm();
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());
        let challenge_mmcs = ChallengeMmcs::new(ValMmcs::new(hash, compress));
        let num_queries = match std::env::var("FRI_QUERIES") {
            Ok(value) => value.parse().unwrap(),
            Err(_) => 100,
        };
        apply_non_stacking_fri(
            FriConfig {
                log_blowup: 1,
                num_queries,
                grinding_bits_query: 24,
                grinding_bits_batching: 10,
                grinding_bits_folding: 0,
                log_final_poly_len: 5,
                cross_round_log_foldings: Vec::new(),
                num_committed_groups: Some(WHIR_NUM_COMMITTED_GROUPS),
                mmcs: challenge_mmcs,
            },
            "core",
        )
    }

    #[must_use]
    pub fn compressed_fri_config() -> FriConfig<ChallengeMmcs> {
        let perm = my_perm();
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());
        let challenge_mmcs = ChallengeMmcs::new(ValMmcs::new(hash, compress));
        let num_queries = match std::env::var("FRI_QUERIES") {
            Ok(value) => value.parse().unwrap(),
            Err(_) => 50,
        };
        apply_non_stacking_fri(
            FriConfig {
                log_blowup: 2,
                num_queries,
                grinding_bits_query: 24,
                grinding_bits_batching: 10,
                grinding_bits_folding: 0,
                log_final_poly_len: 5,
                cross_round_log_foldings: Vec::new(),
                num_committed_groups: Some(WHIR_NUM_COMMITTED_GROUPS),
                mmcs: challenge_mmcs,
            },
            "compress",
        )
    }

    #[must_use]
    pub fn shrink_fri_config() -> FriConfig<ChallengeMmcs> {
        let cfg = super::babybear_config().stage("shrink");
        let perm = my_perm();
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());
        let challenge_mmcs = ChallengeMmcs::new(ValMmcs::new(hash, compress));
        let num_queries = match std::env::var("FRI_QUERIES") {
            Ok(value) => value.parse().unwrap(),
            Err(_) => cfg.num_queries.unwrap_or(33),
        };
        apply_non_stacking_fri(
            FriConfig {
                log_blowup: cfg.log_blowup.unwrap_or(3),
                num_queries,
                grinding_bits_query: cfg.grinding_bits_query.unwrap_or(20),
                grinding_bits_batching: cfg.grinding_bits_batching.unwrap_or(10),
                grinding_bits_folding: cfg.grinding_bits_folding.unwrap_or(0),
                log_final_poly_len: cfg.log_final_poly_len.unwrap_or(5),
                cross_round_log_foldings: Vec::new(),
                num_committed_groups: Some(
                    cfg.num_committed_groups.unwrap_or(WHIR_SHRINK_NUM_COMMITTED_GROUPS),
                ),
                mmcs: challenge_mmcs,
            },
            "shrink",
        )
    }

    #[must_use]
    pub fn root_shrink_fri_config() -> FriConfig<ChallengeMmcs> {
        let cfg = super::babybear_config().stage("root_shrink");
        let perm = my_perm();
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());
        let challenge_mmcs = ChallengeMmcs::new(ValMmcs::new(hash, compress));
        let num_queries = match std::env::var("FRI_QUERIES") {
            Ok(value) => value.parse().unwrap(),
            Err(_) => cfg.num_queries.unwrap_or(25),
        };
        apply_non_stacking_fri(
            FriConfig {
                log_blowup: cfg.log_blowup.unwrap_or(4),
                num_queries,
                grinding_bits_query: cfg.grinding_bits_query.unwrap_or(20),
                grinding_bits_batching: cfg.grinding_bits_batching.unwrap_or(16),
                grinding_bits_folding: cfg.grinding_bits_folding.unwrap_or(0),
                log_final_poly_len: cfg.log_final_poly_len.unwrap_or(5),
                cross_round_log_foldings: Vec::new(),
                num_committed_groups: Some(
                    cfg.num_committed_groups.unwrap_or(WHIR_ROOT_SHRINK_NUM_COMMITTED_GROUPS),
                ),
                mmcs: challenge_mmcs,
            },
            "root_shrink",
        )
    }

    #[must_use]
    pub fn ultra_compressed_fri_config() -> FriConfig<ChallengeMmcs> {
        let perm = my_perm();
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());
        let challenge_mmcs = ChallengeMmcs::new(ValMmcs::new(hash, compress));
        let num_queries = match std::env::var("FRI_QUERIES") {
            Ok(value) => value.parse().unwrap(),
            Err(_) => 33,
        };
        apply_non_stacking_fri(
            FriConfig {
                log_blowup: 3,
                num_queries,
                grinding_bits_query: 20,
                grinding_bits_batching: 6,
                grinding_bits_folding: 0,
                log_final_poly_len: 5,
                cross_round_log_foldings: Vec::new(),
                num_committed_groups: None,
                mmcs: challenge_mmcs,
            },
            "compress",
        )
    }

    #[derive(Clone, Copy)]
    enum BabyBearPoseidon2Type {
        Default,
        Compressed,
        Shrink,
        RootShrink,
    }

    #[derive(Deserialize)]
    #[serde(from = "std::marker::PhantomData<BabyBearPoseidon2>")]
    pub struct BabyBearPoseidon2 {
        pub perm: Perm,
        pcs: Pcs,
        config_type: BabyBearPoseidon2Type,
    }

    impl BabyBearPoseidon2 {
        #[must_use]
        pub fn new() -> Self {
            let perm = my_perm();
            let hash = MyHash::new(perm.clone());
            let compress = MyCompress::new(perm.clone());
            let val_mmcs = ValMmcs::new(hash, compress);
            let dft = Dft {};
            let fri_config = default_fri_config();
            let pcs = Pcs::new(27, dft, val_mmcs, fri_config);
            Self { pcs, perm, config_type: BabyBearPoseidon2Type::Default }
        }

        #[must_use]
        pub fn compressed() -> Self {
            let perm = my_perm();
            let hash = MyHash::new(perm.clone());
            let compress = MyCompress::new(perm.clone());
            let val_mmcs = ValMmcs::new(hash, compress);
            let dft = Dft {};
            let fri_config = compressed_fri_config();
            let pcs = Pcs::new(27, dft, val_mmcs, fri_config);
            Self { pcs, perm, config_type: BabyBearPoseidon2Type::Compressed }
        }

        #[must_use]
        pub fn shrink() -> Self {
            let perm = my_perm();
            let hash = MyHash::new(perm.clone());
            let compress = MyCompress::new(perm.clone());
            let val_mmcs = ValMmcs::new(hash, compress);
            let dft = Dft {};
            let fri_config = shrink_fri_config();
            let pcs = Pcs::new(27, dft, val_mmcs, fri_config);
            Self { pcs, perm, config_type: BabyBearPoseidon2Type::Shrink }
        }

        #[must_use]
        pub fn root_shrink() -> Self {
            let perm = my_perm();
            let hash = MyHash::new(perm.clone());
            let compress = MyCompress::new(perm.clone());
            let val_mmcs = ValMmcs::new(hash, compress);
            let dft = Dft {};
            let fri_config = root_shrink_fri_config();
            let pcs = Pcs::new(27, dft, val_mmcs, fri_config);
            Self { pcs, perm, config_type: BabyBearPoseidon2Type::RootShrink }
        }

        #[must_use]
        pub fn ultra_compressed() -> Self {
            let perm = my_perm();
            let hash = MyHash::new(perm.clone());
            let compress = MyCompress::new(perm.clone());
            let val_mmcs = ValMmcs::new(hash, compress);
            let dft = Dft {};
            let fri_config = ultra_compressed_fri_config();
            let pcs = Pcs::new(27, dft, val_mmcs, fri_config);
            Self { pcs, perm, config_type: BabyBearPoseidon2Type::Compressed }
        }
    }

    impl Clone for BabyBearPoseidon2 {
        fn clone(&self) -> Self {
            match self.config_type {
                BabyBearPoseidon2Type::Default => Self::new(),
                BabyBearPoseidon2Type::Compressed => Self::compressed(),
                BabyBearPoseidon2Type::Shrink => Self::shrink(),
                BabyBearPoseidon2Type::RootShrink => Self::root_shrink(),
            }
        }
    }

    impl Default for BabyBearPoseidon2 {
        fn default() -> Self {
            Self::new()
        }
    }

    /// Implement serialization manually instead of using serde to avoid cloing the config.
    impl Serialize for BabyBearPoseidon2 {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: serde::Serializer,
        {
            std::marker::PhantomData::<BabyBearPoseidon2>.serialize(serializer)
        }
    }

    impl From<std::marker::PhantomData<BabyBearPoseidon2>> for BabyBearPoseidon2 {
        fn from(_: std::marker::PhantomData<BabyBearPoseidon2>) -> Self {
            Self::new()
        }
    }

    impl StarkGenericConfig for BabyBearPoseidon2 {
        type Val = BabyBear;
        type Domain = <Pcs as p3_commit::Pcs<Challenge, Challenger>>::Domain;
        type Pcs = Pcs;
        type Challenge = Challenge;
        type Challenger = Challenger;

        fn pcs(&self) -> &Self::Pcs {
            &self.pcs
        }

        fn challenger(&self) -> Self::Challenger {
            Challenger::new(self.perm.clone())
        }
    }

    impl ZeroCommitment<BabyBearPoseidon2> for Pcs {
        fn zero_commitment(&self) -> Com<BabyBearPoseidon2> {
            DigestHash::from([Val::zero(); DIGEST_SIZE])
        }
    }

    #[derive(Deserialize)]
    #[serde(from = "std::marker::PhantomData<SCBabyBearPoseidon2>")]
    pub struct SCBabyBearPoseidon2 {
        pub perm: Perm,
        pcs: Pcs,
        mlpcs: Mlpcs,
        config_type: BabyBearPoseidon2Type,
    }

    impl SCBabyBearPoseidon2 {
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
            let mlpcs = Mlpcs::from_config(
                val_mmcs,
                whir_config_from_fri(fri_config1, "core", WHIR_CORE_ROUND_QUERY_COUNTS),
            );
            Self { pcs, mlpcs, perm, config_type: BabyBearPoseidon2Type::Default }
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
            let mlpcs = Mlpcs::from_config(
                val_mmcs,
                whir_config_from_fri(fri_config1, "compress", WHIR_COMPRESS_ROUND_QUERY_COUNTS),
            );
            Self { pcs, mlpcs, perm, config_type: BabyBearPoseidon2Type::Compressed }
        }

        #[must_use]
        pub fn shrink() -> Self {
            let perm = my_perm();
            let hash = MyHash::new(perm.clone());
            let compress = MyCompress::new(perm.clone());
            let val_mmcs = ValMmcs::new(hash, compress);
            let dft = Dft {};
            let fri_config = shrink_fri_config();
            let fri_config1 = shrink_fri_config();

            let pcs = Pcs::new(27, dft, val_mmcs.clone(), fri_config);
            let mlpcs = Mlpcs::from_config(
                val_mmcs,
                whir_config_from_fri(fri_config1, "shrink", WHIR_SHRINK_ROUND_QUERY_COUNTS),
            );
            Self { pcs, mlpcs, perm, config_type: BabyBearPoseidon2Type::Shrink }
        }

        #[must_use]
        pub fn root_shrink() -> Self {
            let perm = my_perm();
            let hash = MyHash::new(perm.clone());
            let compress = MyCompress::new(perm.clone());
            let val_mmcs = ValMmcs::new(hash, compress);
            let dft = Dft {};
            let fri_config = root_shrink_fri_config();
            let fri_config1 = root_shrink_fri_config();

            let pcs = Pcs::new(27, dft, val_mmcs.clone(), fri_config);
            let mlpcs = Mlpcs::from_config(
                val_mmcs,
                whir_config_from_fri(
                    fri_config1,
                    "root_shrink",
                    WHIR_ROOT_SHRINK_ROUND_QUERY_COUNTS,
                ),
            );
            Self { pcs, mlpcs, perm, config_type: BabyBearPoseidon2Type::RootShrink }
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
            let mlpcs = Mlpcs::from_config(
                val_mmcs,
                whir_config_from_fri(fri_config1, "compress", WHIR_COMPRESS_ROUND_QUERY_COUNTS),
            );
            Self { pcs, mlpcs, perm, config_type: BabyBearPoseidon2Type::Compressed }
        }

        /// The JSON/stage name for this configuration's pipeline stage.
        fn stage_name(&self) -> &'static str {
            match self.config_type {
                BabyBearPoseidon2Type::Default => "core",
                BabyBearPoseidon2Type::Compressed => "compress",
                BabyBearPoseidon2Type::Shrink => "shrink",
                BabyBearPoseidon2Type::RootShrink => "root_shrink",
            }
        }

        /// Whether this stage commits via the WHIR stacking path.
        pub fn whir_stacking_enabled(&self) -> bool {
            super::babybear_config().stacking_enabled(self.stage_name())
        }
    }

    impl Clone for SCBabyBearPoseidon2 {
        fn clone(&self) -> Self {
            match self.config_type {
                BabyBearPoseidon2Type::Default => Self::new(),
                BabyBearPoseidon2Type::Compressed => Self::compressed(),
                BabyBearPoseidon2Type::Shrink => Self::shrink(),
                BabyBearPoseidon2Type::RootShrink => Self::root_shrink(),
            }
        }
    }

    impl Default for SCBabyBearPoseidon2 {
        fn default() -> Self {
            Self::new()
        }
    }

    /// Implement serialization manually instead of using serde to avoid cloing the config.
    impl Serialize for SCBabyBearPoseidon2 {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: serde::Serializer,
        {
            std::marker::PhantomData::<SCBabyBearPoseidon2>.serialize(serializer)
        }
    }

    impl From<std::marker::PhantomData<SCBabyBearPoseidon2>> for SCBabyBearPoseidon2 {
        fn from(_: std::marker::PhantomData<SCBabyBearPoseidon2>) -> Self {
            Self::new()
        }
    }

    impl StarkGenericConfig for SCBabyBearPoseidon2 {
        type Val = BabyBear;
        type Domain = <Pcs as p3_commit::Pcs<Challenge, Challenger>>::Domain;
        type Pcs = Pcs;
        type Challenge = Challenge;
        type Challenger = Challenger;

        fn pcs(&self) -> &Self::Pcs {
            &self.pcs
        }

        fn challenger(&self) -> Self::Challenger {
            Challenger::new(self.perm.clone())
        }
    }

    impl ZeroCommitment<SCBabyBearPoseidon2> for Pcs {
        fn zero_commitment(&self) -> Com<SCBabyBearPoseidon2> {
            DigestHash::from([Val::zero(); DIGEST_SIZE])
        }
    }

    impl SCStarkGenericConfig for SCBabyBearPoseidon2 {
        type Mlpcs = Mlpcs;
        type MlChallenge = <Mlpcs as MlPCS>::ExtensionField;
        type MlPcsProverData = <Mlpcs as MlPCS>::ProverData;
        type MlChallenger = <Mlpcs as MlPCS>::Challenger;

        fn mlpcs(&self) -> &Self::Mlpcs {
            &self.mlpcs
        }

        fn mlpcs_commit_options(&self) -> MlCommitOptions {
            if self.whir_stacking_enabled() {
                MlCommitOptions::auto_stacking()
            } else {
                MlCommitOptions::no_stacking()
            }
        }

        fn mlchallenger(&self) -> Self::MlChallenger {
            Challenger::new(self.perm.clone())
        }
    }

    impl ZeroCommitment<SCBabyBearPoseidon2> for Mlpcs {
        fn zero_commitment(&self) -> MlCom<SCBabyBearPoseidon2> {
            DigestHash::from([Val::zero(); DIGEST_SIZE])
        }
    }
}
