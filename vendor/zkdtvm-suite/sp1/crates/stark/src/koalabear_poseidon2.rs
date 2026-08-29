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
use pcs::basefold::basefold_pcs::{BasefoldInputProof, BasefoldProof};
use serde::{Deserialize, Serialize};

pub const DIGEST_SIZE: usize = 8;

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

fn log_final_poly_len_from_env(default: usize) -> usize {
    log_final_poly_len_from_envs(&["WHIR_LOG_FINAL_POLY_LEN"], default)
}

fn log_final_poly_len_from_envs(vars: &[&str], default: usize) -> usize {
    vars.iter()
        .find_map(|var| {
            env_var(var).ok().map(|value| {
                value.parse::<usize>().unwrap_or_else(|_| panic!("{var} must be a usize"))
            })
        })
        .unwrap_or(default)
}

fn usize_from_envs(vars: &[&str], default: usize) -> usize {
    vars.iter()
        .find_map(|var| {
            env_var(var).ok().map(|value| {
                value.parse::<usize>().unwrap_or_else(|_| panic!("{var} must be a usize"))
            })
        })
        .unwrap_or(default)
}

// ──── Runtime JSON Configuration ────

#[derive(Debug, Deserialize, Default)]
pub struct StageJsonConfig {
    pub log_blowup: Option<usize>,
    pub num_queries: Option<usize>,
    pub grinding_bits_query: Option<usize>,
    pub grinding_bits_batching: Option<usize>,
    pub grinding_bits_folding: Option<usize>,
    pub log_final_poly_len: Option<usize>,
    pub num_committed_groups: Option<usize>,
    pub round_query_counts: Option<Vec<usize>>,
    /// WHIR commit-local stacking log height (None = auto-stack to the tallest
    /// matrix). The `WHIR_*_STACK_LOG_HEIGHT` env vars override this.
    pub stack_log_height: Option<usize>,
    /// Whether this stage uses the commit-local stacking WHIR path.
    ///
    /// `Some(true)` (or absent → defaults to `true`) keeps the stacked path:
    /// every commit batch is reduced to one stacked matrix and the per-round
    /// query schedule (`round_query_counts` / `num_committed_groups`) applies.
    /// `Some(false)` selects the legacy non-stacking path, which injects
    /// trace groups tallest-first via `merge_beta` and uses a single global
    /// arity-2 query phase controlled by `num_queries` (the per-round
    /// schedule is ignored on that path). The `WHIR_*_STACKING` env vars
    /// override this.
    pub stacking: Option<bool>,
    /// Per-stage WHIR path-pruning switch (Merkle path sharing across
    /// queries). `None` (absent) falls back to `false`. The
    /// `WHIR_<STAGE>_PATH_PRUNING` / `DT_USE_PATH_PRUNING` env vars override
    /// the JSON.
    pub path_pruning: Option<bool>,
}

#[derive(Debug, Deserialize, Default)]
pub struct WhirJsonConfig {
    pub num_skip_rounds: Option<usize>,
    pub chip_log_height_threshold: Option<usize>,
    pub use_algebraic_decomp: Option<bool>,
    pub core: Option<StageJsonConfig>,
    pub compress: Option<StageJsonConfig>,
    pub shrink: Option<StageJsonConfig>,
    pub root_shrink: Option<StageJsonConfig>,
}

impl WhirJsonConfig {
    /// Resolve the effective path-pruning flag for a single stage with priority
    /// `WHIR_<STAGE>_PATH_PRUNING` / `DT_USE_PATH_PRUNING` env var >
    /// per-stage JSON `path_pruning` > `false`.
    ///
    /// Accepted env values: `1`/`true`/`on`/`yes` enable, `0`/`false`/`off`/`no`
    /// disable (case-insensitive). The per-stage env var takes precedence over
    /// the global `DT_USE_PATH_PRUNING`.
    pub fn path_pruning_enabled_for_stage(&self, stage: &str) -> bool {
        let stage_env = match stage {
            "core" => "WHIR_CORE_PATH_PRUNING",
            "compress" => "WHIR_COMPRESS_PATH_PRUNING",
            "shrink" => "WHIR_SHRINK_PATH_PRUNING",
            "root_shrink" => "WHIR_ROOT_SHRINK_PATH_PRUNING",
            _ => "",
        };
        for var in [stage_env, "DT_USE_PATH_PRUNING"] {
            if var.is_empty() {
                continue;
            }
            if let Ok(value) = env_var(var) {
                match value.trim().to_ascii_lowercase().as_str() {
                    "1" | "true" | "on" | "yes" => return true,
                    "0" | "false" | "off" | "no" => return false,
                    _ => {}
                }
            }
        }
        self.stage(stage).path_pruning.unwrap_or(false)
    }

    /// Resolve the effective stacking flag for a stage with priority
    /// `WHIR_<STAGE>_STACKING` / `WHIR_STACKING` env var > JSON `stacking` >
    /// `true` (stacking is the default WHIR path).
    ///
    /// Accepted env values: `1`/`true`/`on`/`yes` enable, `0`/`false`/`off`/`no`
    /// disable (case-insensitive). The per-stage env var takes precedence over
    /// the global `WHIR_STACKING`.
    pub fn stacking_enabled(&self, stage: &str) -> bool {
        let stage_env = match stage {
            "core" => "WHIR_CORE_STACKING",
            "compress" => "WHIR_COMPRESS_STACKING",
            "shrink" => "WHIR_SHRINK_STACKING",
            "root_shrink" => "WHIR_ROOT_SHRINK_STACKING",
            _ => "",
        };
        for var in [stage_env, "WHIR_STACKING"] {
            if var.is_empty() {
                continue;
            }
            if let Ok(value) = env_var(var) {
                match value.trim().to_ascii_lowercase().as_str() {
                    "1" | "true" | "on" | "yes" => return true,
                    "0" | "false" | "off" | "no" => return false,
                    _ => {}
                }
            }
        }
        self.stage(stage).stacking.unwrap_or(true)
    }

    pub fn stage(&self, stage: &str) -> &StageJsonConfig {
        static EMPTY: StageJsonConfig = StageJsonConfig {
            log_blowup: None,
            num_queries: None,
            grinding_bits_query: None,
            grinding_bits_batching: None,
            grinding_bits_folding: None,
            log_final_poly_len: None,
            num_committed_groups: None,
            round_query_counts: None,
            stack_log_height: None,
            stacking: None,
            path_pruning: None,
        };
        match stage {
            "core" => self.core.as_ref().unwrap_or(&EMPTY),
            "compress" => self.compress.as_ref().unwrap_or(&EMPTY),
            "shrink" => self.shrink.as_ref().unwrap_or(&EMPTY),
            "root_shrink" => self.root_shrink.as_ref().unwrap_or(&EMPTY),
            _ => &EMPTY,
        }
    }

    pub fn num_skip_rounds(&self) -> usize {
        self.num_skip_rounds.unwrap_or(1)
    }

    pub fn chip_log_height_threshold(&self) -> usize {
        self.chip_log_height_threshold.unwrap_or(0)
    }

    pub fn use_algebraic_decomp(&self) -> bool {
        self.use_algebraic_decomp.unwrap_or(true)
    }
}

static WHIR_CONFIG: std::sync::OnceLock<WhirJsonConfig> = std::sync::OnceLock::new();

/// The verifier-authoritative v0.8.0 KoalaBear/ext5 parameter profile.
pub const WHIR_CONFIG_FILE: &str = "whir_config_koalabear_ext5.json";
pub const WHIR_CONFIG_JSON: &str =
    include_str!("../../../../../../whir_config_koalabear_ext5.json");

pub fn whir_config() -> &'static WhirJsonConfig {
    WHIR_CONFIG.get_or_init(|| {
        serde_json::from_str(WHIR_CONFIG_JSON).expect("embedded WHIR config must be valid")
    })
}

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
pub type InnerBasefoldProof = BasefoldProof<
    InnerChallenge,
    InnerChallengeMmcs,
    InnerVal,
    BasefoldInputProof<InnerVal, InnerValMmcs>,
>;

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
    let num_queries = match env_var("FRI_QUERIES") {
        Ok(value) => value.parse().unwrap(),
        Err(_) => 193,
    };
    FriConfig {
        log_blowup: 1,
        num_queries,
        grinding_bits_query: 20,
        grinding_bits_batching: 10,
        grinding_bits_folding: 0,
        log_final_poly_len: log_final_poly_len_from_env(6),
        cross_round_log_foldings: Vec::new(),
        num_committed_groups: None,
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
    let num_queries = match env_var("FRI_QUERIES") {
        Ok(value) => value.parse().unwrap(),
        Err(_) => 193,
    };
    FriConfig {
        log_blowup: 1,
        num_queries,
        grinding_bits_query: 20,
        grinding_bits_batching: 10,
        grinding_bits_folding: 0,
        log_final_poly_len: log_final_poly_len_from_env(6),
        cross_round_log_foldings: Vec::new(),
        num_committed_groups: None,
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
    type Challenge = InnerChallenge;
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
    use p3_challenger::{DuplexChallenger, HashChallenger, SerializingChallenger32};
    use p3_commit::ExtensionMmcs;
    use p3_dft::Radix2DitParallel;
    #[cfg(not(feature = "ext5"))]
    use p3_field::extension::BinomialExtensionField;
    #[cfg(feature = "ext5")]
    use p3_field::extension::QuinticTrinomialExtensionField;
    use p3_field::{AbstractField, Field};
    use p3_fri::{FriConfig, TwoAdicFriPcs};
    use p3_koala_bear::{DiffusionMatrixKoalaBear, KoalaBear};
    use p3_merkle_tree::FieldMerkleTreeMmcs;
    use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
    use p3_sha256::{Sha256, Sha256Compress};
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
    pub type Val = KoalaBear;
    #[cfg(not(feature = "ext5"))]
    pub type Challenge = BinomialExtensionField<Val, 4>;
    #[cfg(feature = "ext5")]
    pub type Challenge = QuinticTrinomialExtensionField<Val>;

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

    pub type RootHash = Sha256;
    pub type RootFieldHash = p3_symmetric::SerializingHasher32<RootHash>;
    pub type RootDigestHash = Hash<Val, u8, 32>;
    pub type RootCompress = Sha256Compress;
    pub type RootValMmcs = FieldMerkleTreeMmcs<Val, u8, RootFieldHash, RootCompress, 32>;
    pub type RootChallengeMmcs = ExtensionMmcs<Val, Challenge, RootValMmcs>;
    pub type RootChallenger = SerializingChallenger32<Val, HashChallenger<u8, RootHash, 32>>;
    type RootPcs = TwoAdicFriPcs<Val, Dft, RootValMmcs, RootChallengeMmcs>;
    pub type RootMlpcs =
        BaseFoldPcs<Val, RootValMmcs, RootChallengeMmcs, Challenge, RootChallenger>;

    /// Number of committed IOPP groups for the WHIR cross-round schedule, per stage.
    ///
    /// When set, `FriConfig::num_committed_groups = Some(N)` causes the WHIR PCS
    /// to uniformly distribute `active_rounds = num_vars - log_final_poly_len`
    /// sumcheck variables across exactly N committed IOPP groups, with larger
    /// groups first if not evenly divisible.
    ///
    /// Core/Compress use 4 groups (e.g. active=16 → [4,4,4,4]).
    /// Shrink/Root-shrink use 3 groups (e.g. active=14 → [5,5,4]) for better
    /// per-group folding depth under JBR(m=2).
    pub const WHIR_CORE_NUM_COMMITTED_GROUPS: usize = 4;
    pub const WHIR_COMPRESS_NUM_COMMITTED_GROUPS: usize = 4;
    pub const WHIR_SHRINK_NUM_COMMITTED_GROUPS: usize = 3;
    pub const WHIR_ROOT_SHRINK_NUM_COMMITTED_GROUPS: usize = 3;

    /// Per-round WHIR query counts, calibrated for 100-bit security.
    ///
    /// The security regime is chosen per circuit to minimize proof size while
    /// meeting the 100-bit target (verified via soundcalc/WHIR PCS model):
    ///
    /// - **Core** (rate=1/2): **UDR** (Unique Decoding Regime). JBR is invalid at rate=1/2 because
    ///   δ_JBR < 0 for m=1 and δ_JBR < δ_UDR for m=2. UDR gives δ=0.25, requiring ~193 queries on
    ///   iteration 0 (where -log2(1-δ) ≈ 0.415) plus 20 bits grinding per round.
    ///   `grinding_bits_folding = 0` (no folding PoW needed under UDR).
    ///
    /// - **Compress** (rate=1/4): **JBR(m=2)** (Johnson Bound Regime, multiplicity=2). JBR gives
    ///   larger δ on later iterations (δ_1≈0.78 vs UDR 0.48), allowing far fewer queries there
    ///   [37,22,16] vs [84,81,81]. The trade-off: JBR's list size L=(m+0.5)/√ρ ≈ 5.0 inflates the
    ///   folding error d·L/|F|+ε_powers, requiring `grinding_bits_folding = 18` to compensate. Net
    ///   effect: proof size ~93 KiB (vs 175 KiB under UDR), -47%.
    ///
    /// - **Shrink / Root-shrink** (rate=1/8 or 1/16): **JBR(m=2)**. Same reasoning as Compress but
    ///   with even better δ (δ_0≈0.56 vs UDR 0.44). `grinding_bits_folding = 20`. Net effect: proof
    ///   size ~69 KiB (vs 164 KiB under UDR), -58%.
    ///
    /// Only the first N entries are used (N = per-stage `WHIR_*_NUM_COMMITTED_GROUPS`,
    /// the number of committed IOPP groups computed by the uniform schedule).
    pub const WHIR_CORE_ROUND_QUERY_COUNTS: &[usize] = &[193, 88, 81, 81];
    pub const WHIR_COMPRESS_ROUND_QUERY_COUNTS: &[usize] = &[118, 37, 22, 16];
    pub const WHIR_SHRINK_ROUND_QUERY_COUNTS: &[usize] = &[68, 30, 20];
    pub const WHIR_ROOT_SHRINK_ROUND_QUERY_COUNTS: &[usize] = &[57, 28, 19];

    fn parse_cross_round_log_foldings(var: &str, value: &str) -> Vec<usize> {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            return Vec::new();
        }

        trimmed
            .split(',')
            .map(|part| {
                let folding = part
                    .trim()
                    .parse::<usize>()
                    .unwrap_or_else(|_| panic!("{var} must be comma-separated usize values"));
                assert!(folding > 0, "{var} entries must be positive");
                folding
            })
            .collect()
    }

    #[allow(dead_code)]
    fn cross_round_log_foldings_from_env(vars: &[&str], default: &[usize]) -> Vec<usize> {
        vars.iter()
            .find_map(|var| {
                super::env_var(var).ok().map(|value| parse_cross_round_log_foldings(var, &value))
            })
            .unwrap_or_else(|| default.to_vec())
    }

    fn round_query_counts_from_env(vars: &[&str], default: &[usize]) -> Vec<usize> {
        vars.iter()
            .find_map(|var| {
                super::env_var(var).ok().map(|value| parse_cross_round_log_foldings(var, &value))
            })
            .unwrap_or_else(|| default.to_vec())
    }

    fn whir_config_from_fri<FriMmcs>(
        fri_config: FriConfig<FriMmcs>,
        stage: &str,
        vars: &[&str],
        default_queries: &[usize],
    ) -> BasefoldConfig<FriMmcs> {
        let path_pruning = super::whir_config().path_pruning_enabled_for_stage(stage);

        // Non-stacking (Jagged) path: the legacy WHIR opening uses a single
        // global arity-2 query phase driven by `num_queries`, with no
        // early-stop. Force `log_final_poly_len = 0` and leave the per-round
        // query schedule unset (`round_queries = None`) so the stacked-only
        // parameters (`round_query_counts` / `num_committed_groups`) are
        // ignored. Early-stop is a stacking-only optimization.
        if !super::whir_config().stacking_enabled(stage) {
            let mut fri_config = fri_config;
            fri_config.log_final_poly_len = 0;
            return BasefoldConfig::new(fri_config).with_path_pruning(path_pruning);
        }

        let cfg = super::whir_config().stage(stage);
        let json_queries = cfg.round_query_counts.as_deref();
        let effective_default = json_queries.unwrap_or(default_queries);
        BasefoldConfig::new(fri_config)
            .with_round_query_counts(round_query_counts_from_env(vars, effective_default))
            .with_path_pruning(path_pruning)
    }

    /// The MLPCS verifier configuration shared by core proving and native replay recording.
    pub fn core_mlpcs_config() -> BasefoldConfig<ChallengeMmcs> {
        whir_config_from_fri(
            default_fri_config(),
            "core",
            &["WHIR_CORE_ROUND_QUERIES", "WHIR_ROUND_QUERIES"],
            WHIR_CORE_ROUND_QUERY_COUNTS,
        )
    }

    /// The MLPCS verifier configuration shared by compress proving and native replay recording.
    pub fn compress_mlpcs_config() -> BasefoldConfig<ChallengeMmcs> {
        whir_config_from_fri(
            compressed_fri_config(),
            "compress",
            &["WHIR_COMPRESS_ROUND_QUERIES", "WHIR_ROUND_QUERIES"],
            WHIR_COMPRESS_ROUND_QUERY_COUNTS,
        )
    }

    /// The MLPCS verifier configuration shared by shrink proving and native replay recording.
    pub fn shrink_mlpcs_config() -> BasefoldConfig<ChallengeMmcs> {
        whir_config_from_fri(
            shrink_fri_config(),
            "shrink",
            &["WHIR_SHRINK_ROUND_QUERIES", "WHIR_ROUND_QUERIES"],
            WHIR_SHRINK_ROUND_QUERY_COUNTS,
        )
    }

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

    fn build_fri_config_with_mmcs<FriMmcs>(
        stage: &str,
        default_log_blowup: usize,
        default_num_queries: usize,
        query_pow_env_vars: &[&str],
        default_grinding_bits_query: usize,
        default_grinding_bits_batching: usize,
        default_grinding_bits_folding: usize,
        final_poly_env_vars: &[&str],
        default_log_final_poly_len: usize,
        default_num_committed_groups: usize,
        challenge_mmcs: FriMmcs,
    ) -> FriConfig<FriMmcs> {
        let cfg = super::whir_config().stage(stage);

        let num_queries = match super::env_var("FRI_QUERIES") {
            Ok(value) => value.parse().unwrap(),
            Err(_) => cfg.num_queries.unwrap_or(default_num_queries),
        };

        // Non-stacking (Jagged) stages have no FRI early-stop. Force
        // `log_final_poly_len = 0` here so BOTH the native WHIR config
        // (`whir_config_from_fri`) and the recursion circuit verifier
        // (`machine.config().fri_config()`) see the same value; otherwise the
        // circuit expects a `2^log_final_poly_len` final polynomial that the
        // non-stacking prover never emits.
        let stacking = super::whir_config().stacking_enabled(stage);
        let log_final_poly_len = if stacking {
            super::log_final_poly_len_from_envs(
                final_poly_env_vars,
                cfg.log_final_poly_len.unwrap_or(default_log_final_poly_len),
            )
        } else {
            0
        };

        FriConfig {
            log_blowup: cfg.log_blowup.unwrap_or(default_log_blowup),
            num_queries,
            grinding_bits_query: super::usize_from_envs(
                query_pow_env_vars,
                cfg.grinding_bits_query.unwrap_or(default_grinding_bits_query),
            ),
            grinding_bits_batching: cfg
                .grinding_bits_batching
                .unwrap_or(default_grinding_bits_batching),
            grinding_bits_folding: cfg
                .grinding_bits_folding
                .unwrap_or(default_grinding_bits_folding),
            log_final_poly_len,
            cross_round_log_foldings: Vec::new(),
            num_committed_groups: Some(
                cfg.num_committed_groups.unwrap_or(default_num_committed_groups),
            ),
            mmcs: challenge_mmcs,
        }
    }

    fn build_fri_config(
        stage: &str,
        default_log_blowup: usize,
        default_num_queries: usize,
        query_pow_env_vars: &[&str],
        default_grinding_bits_query: usize,
        default_grinding_bits_batching: usize,
        default_grinding_bits_folding: usize,
        final_poly_env_vars: &[&str],
        default_log_final_poly_len: usize,
        default_num_committed_groups: usize,
    ) -> FriConfig<ChallengeMmcs> {
        let perm = my_perm();
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());
        let challenge_mmcs = ChallengeMmcs::new(ValMmcs::new(hash, compress));
        build_fri_config_with_mmcs(
            stage,
            default_log_blowup,
            default_num_queries,
            query_pow_env_vars,
            default_grinding_bits_query,
            default_grinding_bits_batching,
            default_grinding_bits_folding,
            final_poly_env_vars,
            default_log_final_poly_len,
            default_num_committed_groups,
            challenge_mmcs,
        )
    }

    fn build_root_sha256_fri_config() -> FriConfig<RootChallengeMmcs> {
        let field_hash = RootFieldHash::new(Sha256);
        let compress = Sha256Compress;
        let challenge_mmcs = RootChallengeMmcs::new(RootValMmcs::new(field_hash, compress));
        build_fri_config_with_mmcs(
            "root_shrink",
            4,
            57,
            &[
                "WHIR_ROOT_SHRINK_QUERY_POW_BITS",
                "WHIR_SHRINK_QUERY_POW_BITS",
                "WHIR_QUERY_POW_BITS",
            ],
            20,
            20,
            20,
            &[
                "WHIR_ROOT_SHRINK_LOG_FINAL_POLY_LEN",
                "WHIR_SHRINK_LOG_FINAL_POLY_LEN",
                "WHIR_LOG_FINAL_POLY_LEN",
            ],
            6,
            WHIR_ROOT_SHRINK_NUM_COMMITTED_GROUPS,
            challenge_mmcs,
        )
    }

    /// Core FRI config — UDR regime, no folding PoW needed.
    #[must_use]
    pub fn default_fri_config() -> FriConfig<ChallengeMmcs> {
        build_fri_config(
            "core",
            1,   // log_blowup
            193, // num_queries
            &["WHIR_CORE_QUERY_POW_BITS", "WHIR_QUERY_POW_BITS"],
            20, // grinding_bits_query
            10, // grinding_bits_batching
            0,  // grinding_bits_folding (UDR)
            &["WHIR_CORE_LOG_FINAL_POLY_LEN", "WHIR_LOG_FINAL_POLY_LEN"],
            6, // log_final_poly_len
            WHIR_CORE_NUM_COMMITTED_GROUPS,
        )
    }

    /// Compress FRI config — JBR(m=2) regime, 18 bits folding PoW.
    #[must_use]
    pub fn compressed_fri_config() -> FriConfig<ChallengeMmcs> {
        build_fri_config(
            "compress",
            2,   // log_blowup
            118, // num_queries
            &["WHIR_COMPRESS_QUERY_POW_BITS", "WHIR_QUERY_POW_BITS"],
            20, // grinding_bits_query
            10, // grinding_bits_batching
            18, // grinding_bits_folding (JBR m=2)
            &["WHIR_COMPRESS_LOG_FINAL_POLY_LEN", "WHIR_LOG_FINAL_POLY_LEN"],
            6, // log_final_poly_len
            WHIR_COMPRESS_NUM_COMMITTED_GROUPS,
        )
    }

    /// Shrink FRI config — JBR(m=2) regime, 20 bits folding PoW.
    #[must_use]
    pub fn shrink_fri_config() -> FriConfig<ChallengeMmcs> {
        build_fri_config(
            "shrink",
            3,  // log_blowup
            97, // num_queries
            &["WHIR_SHRINK_QUERY_POW_BITS", "WHIR_QUERY_POW_BITS"],
            20, // grinding_bits_query
            10, // grinding_bits_batching
            20, // grinding_bits_folding (JBR m=2)
            &["WHIR_SHRINK_LOG_FINAL_POLY_LEN", "WHIR_LOG_FINAL_POLY_LEN"],
            6, // log_final_poly_len
            WHIR_SHRINK_NUM_COMMITTED_GROUPS,
        )
    }

    /// Root-shrink FRI config — JBR(m=2) regime, same folding PoW as shrink.
    #[must_use]
    pub fn root_shrink_fri_config() -> FriConfig<ChallengeMmcs> {
        build_fri_config(
            "root_shrink",
            4,  // log_blowup
            57, // num_queries
            &[
                "WHIR_ROOT_SHRINK_QUERY_POW_BITS",
                "WHIR_SHRINK_QUERY_POW_BITS",
                "WHIR_QUERY_POW_BITS",
            ],
            20, // grinding_bits_query
            20, // grinding_bits_batching
            20, // grinding_bits_folding (JBR m=2)
            &[
                "WHIR_ROOT_SHRINK_LOG_FINAL_POLY_LEN",
                "WHIR_SHRINK_LOG_FINAL_POLY_LEN",
                "WHIR_LOG_FINAL_POLY_LEN",
            ],
            6, // log_final_poly_len
            WHIR_ROOT_SHRINK_NUM_COMMITTED_GROUPS,
        )
    }

    #[must_use]
    pub fn ultra_compressed_fri_config() -> FriConfig<ChallengeMmcs> {
        build_fri_config(
            "compress",
            1,   // log_blowup
            193, // num_queries
            &["WHIR_COMPRESS_QUERY_POW_BITS", "WHIR_QUERY_POW_BITS"],
            20, // grinding_bits_query
            10, // grinding_bits_batching
            0,  // grinding_bits_folding
            &["WHIR_COMPRESS_LOG_FINAL_POLY_LEN", "WHIR_LOG_FINAL_POLY_LEN"],
            6, // log_final_poly_len
            WHIR_COMPRESS_NUM_COMMITTED_GROUPS,
        )
    }

    #[derive(Clone, Copy)]
    enum KoalaBearPoseidon2Type {
        Default,
        Compressed,
        Shrink,
        RootShrink,
    }

    /// Resolve the stacking log height with priority env vars > JSON
    /// `stack_log_height` (for `stage`) > None (auto-stack).
    fn stack_log_height_hint(vars: &[&str], stage: &str) -> Option<usize> {
        let from_env = vars.iter().find_map(|var| {
            super::env_var(var).ok().map(|value| {
                value.parse::<usize>().unwrap_or_else(|_| panic!("{var} must be a valid usize"))
            })
        });
        from_env.or_else(|| super::whir_config().stage(stage).stack_log_height)
    }

    #[derive(Deserialize)]
    #[serde(from = "std::marker::PhantomData<SCKoalaBearPoseidon2>")]
    pub struct SCKoalaBearPoseidon2 {
        pub perm: Perm,
        pcs: Pcs,
        mlpcs: Mlpcs,
        config_type: KoalaBearPoseidon2Type,
    }

    #[derive(Deserialize)]
    #[serde(from = "std::marker::PhantomData<SCKoalaBearSha256Root>")]
    pub struct SCKoalaBearSha256Root {
        /// Poseidon2 is still needed by the recursion VM program executed in the
        /// root_shrink stage; only this stage's final PCS commitments use SHA256.
        pub perm: Perm,
        pcs: RootPcs,
        mlpcs: RootMlpcs,
    }

    impl SCKoalaBearSha256Root {
        #[must_use]
        pub fn root_shrink() -> Self {
            let perm = my_perm();
            let field_hash = RootFieldHash::new(Sha256);
            let compress = Sha256Compress;
            let val_mmcs = RootValMmcs::new(field_hash, compress);
            let dft = Dft {};
            let fri_config = build_root_sha256_fri_config();
            let fri_config1 = build_root_sha256_fri_config();

            let pcs = RootPcs::new(27, dft, val_mmcs.clone(), fri_config);
            let mlpcs = RootMlpcs::from_config(
                val_mmcs,
                whir_config_from_fri(
                    fri_config1,
                    "root_shrink",
                    &[
                        "WHIR_ROOT_SHRINK_ROUND_QUERIES",
                        "WHIR_SHRINK_ROUND_QUERIES",
                        "WHIR_ROUND_QUERIES",
                    ],
                    WHIR_ROOT_SHRINK_ROUND_QUERY_COUNTS,
                ),
            );
            Self { pcs, mlpcs, perm }
        }

        /// Whether root_shrink commits via the WHIR stacking path.
        pub fn whir_stacking_enabled(&self) -> bool {
            super::whir_config().stacking_enabled("root_shrink")
        }

        pub fn whir_stack_log_height_hint(&self) -> Option<usize> {
            stack_log_height_hint(
                &[
                    "WHIR_ROOT_SHRINK_STACK_LOG_HEIGHT",
                    "WHIR_SHRINK_STACK_LOG_HEIGHT",
                    "WHIR_STACK_LOG_HEIGHT",
                ],
                "root_shrink",
            )
        }
    }

    impl Clone for SCKoalaBearSha256Root {
        fn clone(&self) -> Self {
            Self::root_shrink()
        }
    }

    impl Default for SCKoalaBearSha256Root {
        fn default() -> Self {
            Self::root_shrink()
        }
    }

    impl Serialize for SCKoalaBearSha256Root {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: serde::Serializer,
        {
            std::marker::PhantomData::<SCKoalaBearSha256Root>.serialize(serializer)
        }
    }

    impl From<std::marker::PhantomData<SCKoalaBearSha256Root>> for SCKoalaBearSha256Root {
        fn from(_: std::marker::PhantomData<SCKoalaBearSha256Root>) -> Self {
            Self::root_shrink()
        }
    }

    impl StarkGenericConfig for SCKoalaBearSha256Root {
        type Val = KoalaBear;
        type Domain = <RootPcs as p3_commit::Pcs<Challenge, RootChallenger>>::Domain;
        type Pcs = RootPcs;
        type Challenge = Challenge;
        type Challenger = RootChallenger;

        fn pcs(&self) -> &Self::Pcs {
            &self.pcs
        }

        fn challenger(&self) -> Self::Challenger {
            RootChallenger::from_hasher(Vec::new(), Sha256)
        }
    }

    impl ZeroCommitment<SCKoalaBearSha256Root> for RootPcs {
        fn zero_commitment(&self) -> Com<SCKoalaBearSha256Root> {
            RootDigestHash::from([0u8; 32])
        }
    }

    impl SCStarkGenericConfig for SCKoalaBearSha256Root {
        type Mlpcs = RootMlpcs;
        type MlChallenge = <RootMlpcs as MlPCS>::ExtensionField;
        type MlPcsProverData = <RootMlpcs as MlPCS>::ProverData;
        type MlChallenger = <RootMlpcs as MlPCS>::Challenger;

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

        fn mlpcs_stack_log_height_hint(&self) -> Option<usize> {
            self.whir_stack_log_height_hint()
        }

        fn mlchallenger(&self) -> Self::MlChallenger {
            RootChallenger::from_hasher(Vec::new(), Sha256)
        }
    }

    impl ZeroCommitment<SCKoalaBearSha256Root> for RootMlpcs {
        fn zero_commitment(&self) -> MlCom<SCKoalaBearSha256Root> {
            RootDigestHash::from([0u8; 32])
        }
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
            let config_type = KoalaBearPoseidon2Type::Default;

            let pcs = Pcs::new(27, dft, val_mmcs.clone(), fri_config);
            let mlpcs = Mlpcs::from_config(val_mmcs, core_mlpcs_config());
            Self { pcs, mlpcs, perm, config_type }
        }

        #[must_use]
        pub fn compressed() -> Self {
            let perm = my_perm();
            let hash = MyHash::new(perm.clone());
            let compress = MyCompress::new(perm.clone());
            let val_mmcs = ValMmcs::new(hash, compress);
            let dft = Dft {};
            let fri_config = compressed_fri_config();
            let config_type = KoalaBearPoseidon2Type::Compressed;

            let pcs = Pcs::new(27, dft, val_mmcs.clone(), fri_config);
            let mlpcs = Mlpcs::from_config(val_mmcs, compress_mlpcs_config());
            Self { pcs, mlpcs, perm, config_type }
        }

        #[must_use]
        pub fn shrink() -> Self {
            let perm = my_perm();
            let hash = MyHash::new(perm.clone());
            let compress = MyCompress::new(perm.clone());
            let val_mmcs = ValMmcs::new(hash, compress);
            let dft = Dft {};
            let fri_config = shrink_fri_config();
            let config_type = KoalaBearPoseidon2Type::Shrink;

            let pcs = Pcs::new(27, dft, val_mmcs.clone(), fri_config);
            let mlpcs = Mlpcs::from_config(val_mmcs, shrink_mlpcs_config());
            Self { pcs, mlpcs, perm, config_type }
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
            let config_type = KoalaBearPoseidon2Type::RootShrink;

            let pcs = Pcs::new(27, dft, val_mmcs.clone(), fri_config);
            let mlpcs = Mlpcs::from_config(
                val_mmcs,
                whir_config_from_fri(
                    fri_config1,
                    "root_shrink",
                    &[
                        "WHIR_ROOT_SHRINK_ROUND_QUERIES",
                        "WHIR_SHRINK_ROUND_QUERIES",
                        "WHIR_ROUND_QUERIES",
                    ],
                    WHIR_ROOT_SHRINK_ROUND_QUERY_COUNTS,
                ),
            );
            Self { pcs, mlpcs, perm, config_type }
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
            let config_type = KoalaBearPoseidon2Type::Compressed;

            let pcs = Pcs::new(27, dft, val_mmcs.clone(), fri_config);
            let mlpcs = Mlpcs::from_config(
                val_mmcs,
                whir_config_from_fri(
                    fri_config1,
                    "compress",
                    &["WHIR_COMPRESS_ROUND_QUERIES", "WHIR_ROUND_QUERIES"],
                    WHIR_COMPRESS_ROUND_QUERY_COUNTS,
                ),
            );
            Self { pcs, mlpcs, perm, config_type }
        }

        pub fn cross_round_log_foldings(&self) -> Vec<usize> {
            self.mlpcs.cross_round_log_foldings()
        }

        /// The JSON/stage name for this configuration's pipeline stage.
        pub const fn whir_stage_name(&self) -> &'static str {
            match self.config_type {
                KoalaBearPoseidon2Type::Default => "core",
                KoalaBearPoseidon2Type::Compressed => "compress",
                KoalaBearPoseidon2Type::Shrink => "shrink",
                KoalaBearPoseidon2Type::RootShrink => "root_shrink",
            }
        }

        /// Whether this stage commits via the WHIR stacking path.
        pub fn whir_stacking_enabled(&self) -> bool {
            super::whir_config().stacking_enabled(self.whir_stage_name())
        }

        pub fn whir_stack_log_height_hint(&self) -> Option<usize> {
            match self.config_type {
                KoalaBearPoseidon2Type::Default => stack_log_height_hint(
                    &["WHIR_CORE_STACK_LOG_HEIGHT", "WHIR_STACK_LOG_HEIGHT"],
                    "core",
                ),
                KoalaBearPoseidon2Type::Compressed => stack_log_height_hint(
                    &["WHIR_COMPRESS_STACK_LOG_HEIGHT", "WHIR_STACK_LOG_HEIGHT"],
                    "compress",
                ),
                KoalaBearPoseidon2Type::Shrink => stack_log_height_hint(
                    &["WHIR_SHRINK_STACK_LOG_HEIGHT", "WHIR_STACK_LOG_HEIGHT"],
                    "shrink",
                ),
                KoalaBearPoseidon2Type::RootShrink => stack_log_height_hint(
                    &[
                        "WHIR_ROOT_SHRINK_STACK_LOG_HEIGHT",
                        "WHIR_SHRINK_STACK_LOG_HEIGHT",
                        "WHIR_STACK_LOG_HEIGHT",
                    ],
                    "root_shrink",
                ),
            }
        }
    }

    impl Clone for SCKoalaBearPoseidon2 {
        fn clone(&self) -> Self {
            match self.config_type {
                KoalaBearPoseidon2Type::Default => Self::new(),
                KoalaBearPoseidon2Type::Compressed => Self::compressed(),
                KoalaBearPoseidon2Type::Shrink => Self::shrink(),
                KoalaBearPoseidon2Type::RootShrink => Self::root_shrink(),
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
        type Challenge = Challenge;
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

        fn mlpcs_commit_options(&self) -> MlCommitOptions {
            if self.whir_stacking_enabled() {
                MlCommitOptions::auto_stacking()
            } else {
                MlCommitOptions::no_stacking()
            }
        }

        fn mlpcs_stack_log_height_hint(&self) -> Option<usize> {
            self.whir_stack_log_height_hint()
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
