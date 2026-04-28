use crate::air::PublicValues;
use crate::config::Challenge;
use crate::word::Word;
use crate::{
    septic_digest::SepticDigest,
    sumcheck::{
        config::SCStarkGenericConfig,
        keys::SCStarkVerifyingKey,
    },
};
use hashbrown::HashMap;
use itertools::Itertools;
use p3_air::Air;
use p3_challenger::CanObserve;
use p3_field::{AbstractExtensionField, AbstractField};
use p3_matrix::{dense::RowMajorMatrix, Dimensions};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{borrow::Borrow, fmt::Debug, iter::once};
use tracing::instrument;

use crate::{
    air::{InteractionScope, MachineAir},
    lookup::InteractionKind,
    ShardProof, VerifierConstraintFolder,
};

use super::{Chip, MachineProof, VerificationError, Verifier};
use crate::config::{Com, Dom, PcsProverData, StarkGenericConfig, Val};

use crate::sumcheck::{
    proof::{SCMachineProof, SCShardProof},
    verifier::{SumcheckVerificationError, Verifier as SumcheckVerifier},
};

pub type MachineChip<SC, A> = Chip<Val<SC>, A>;

pub type SumcheckChip<SC, A> = Chip<Challenge<SC>, A>;

/// Compute the expected local cumulative sum contribution from dangling interactions.
pub fn compute_expected_state_imbalance<SC: StarkGenericConfig>(
    public_values: &[Val<SC>],
    permutation_challenges: &[Challenge<SC>],
    global_cumulative_sum: SepticDigest<Val<SC>>,
) -> Challenge<SC> {
    let pv: &PublicValues<Word<Val<SC>>, Val<SC>> = public_values.borrow();
    let alpha = permutation_challenges[0];
    let beta = permutation_challenges[1];
    let one = Challenge::<SC>::one();
    let mut result = Challenge::<SC>::zero();

    let start_clk = pv.start_clk;
    let exit_clk = pv.exit_clk;
    if start_clk != exit_clk {
        let beta2 = beta * beta;
        let beta3 = beta2 * beta;

        let state_kind = Challenge::<SC>::from_canonical_usize(InteractionKind::State as usize);
        let shard_term = beta * Challenge::<SC>::from_base(pv.execution_shard);

        let recv_fp = alpha
            + state_kind
            + shard_term
            + beta2 * Challenge::<SC>::from_base(start_clk)
            + beta3 * Challenge::<SC>::from_base(pv.start_pc);

        let send_fp = alpha
            + state_kind
            + shard_term
            + beta2 * Challenge::<SC>::from_base(exit_clk)
            + beta3 * Challenge::<SC>::from_base(pv.next_pc);

        result += one / send_fp - one / recv_fp;
    }

    let beta2 = beta * beta;
    let addr_kind =
        Challenge::<SC>::from_canonical_usize(InteractionKind::MemoryGlobalAddr as usize);

    let prev_init = pv.previous_init_addr;
    let last_init = pv.last_init_addr;
    if prev_init != last_init {
        let base = alpha + addr_kind;
        let recv_fp = base + beta2 * Challenge::<SC>::from_base(prev_init);
        let send_fp = base + beta2 * Challenge::<SC>::from_base(last_init);
        result += one / send_fp - one / recv_fp;
    }

    let prev_fin = pv.previous_finalize_addr;
    let last_fin = pv.last_finalize_addr;
    if prev_fin != last_fin {
        let base = alpha + addr_kind + beta;
        let recv_fp = base + beta2 * Challenge::<SC>::from_base(prev_fin);
        let send_fp = base + beta2 * Challenge::<SC>::from_base(last_fin);
        result += one / send_fp - one / recv_fp;
    }

    if !global_cumulative_sum.is_zero() {
        let zero_point = SepticDigest::<Val<SC>>::zero_for_field().0;
        let final_point = global_cumulative_sum.0;
        let chain_kind =
            Challenge::<SC>::from_canonical_usize(InteractionKind::GlobalDigestChain as usize);

        let mut recv_fp = alpha + chain_kind;
        let mut send_fp = alpha + chain_kind;
        let mut beta_pow = beta;
        for i in 0..7 {
            recv_fp += beta_pow * Challenge::<SC>::from_base(zero_point.x.0[i]);
            send_fp += beta_pow * Challenge::<SC>::from_base(final_point.x.0[i]);
            beta_pow *= beta;
        }
        for i in 0..7 {
            recv_fp += beta_pow * Challenge::<SC>::from_base(zero_point.y.0[i]);
            send_fp += beta_pow * Challenge::<SC>::from_base(final_point.y.0[i]);
            beta_pow *= beta;
        }

        result += one / send_fp - one / recv_fp;
    }

    result
}

pub struct StarkMachine<SC: StarkGenericConfig, A> {
    config: SC,
    chips: Vec<Chip<Val<SC>, A>>,
    num_pv_elts: usize,
    contains_global_bus: bool,
}

impl<SC: StarkGenericConfig, A> StarkMachine<SC, A> {
    pub const fn new(
        config: SC,
        chips: Vec<Chip<Val<SC>, A>>,
        num_pv_elts: usize,
        contains_global_bus: bool,
    ) -> Self {
        Self { config, chips, num_pv_elts, contains_global_bus }
    }
}

pub struct SCStarkMachine<SC: SCStarkGenericConfig, A, AE> {
    config: SC,
    chips: Vec<Chip<Val<SC>, A>>,
    chips_ext: Vec<Chip<Challenge<SC>, AE>>,
    num_pv_elts: usize,
    contains_global_bus: bool,
}

impl<SC: SCStarkGenericConfig, A, AE> SCStarkMachine<SC, A, AE> {
    pub const fn new(
        config: SC,
        chips: Vec<Chip<Val<SC>, A>>,
        chips_ext: Vec<Chip<Challenge<SC>, AE>>,
        num_pv_elts: usize,
        contains_global_bus: bool,
    ) -> Self {
        Self { config, chips, chips_ext, num_pv_elts, contains_global_bus }
    }

    pub const fn has_global_bus(&self) -> bool {
        self.contains_global_bus
    }
}

/// Proving key (retained for type compatibility; not used at verification time).
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "PcsProverData<SC>: Serialize"))]
#[serde(bound(deserialize = "PcsProverData<SC>: DeserializeOwned"))]
pub struct StarkProvingKey<SC: StarkGenericConfig> {
    pub commit: Com<SC>,
    pub pc_start: Val<SC>,
    pub initial_global_cumulative_sum: SepticDigest<Val<SC>>,
    pub traces: Vec<RowMajorMatrix<Val<SC>>>,
    pub data: PcsProverData<SC>,
    pub chip_ordering: HashMap<String, usize>,
    pub local_only: Vec<bool>,
    pub constraints_map: HashMap<String, usize>,
}

impl<SC: StarkGenericConfig> StarkProvingKey<SC> {
    pub fn observe_into(&self, challenger: &mut SC::Challenger) {
        challenger.observe(self.commit.clone());
        challenger.observe(self.pc_start);
        challenger.observe_slice(&self.initial_global_cumulative_sum.0.x.0);
        challenger.observe_slice(&self.initial_global_cumulative_sum.0.y.0);
        challenger.observe(Val::<SC>::zero());
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "Dom<SC>: Serialize"))]
#[serde(bound(deserialize = "Dom<SC>: DeserializeOwned"))]
pub struct StarkVerifyingKey<SC: StarkGenericConfig> {
    pub commit: Com<SC>,
    pub pc_start: Val<SC>,
    pub initial_global_cumulative_sum: SepticDigest<Val<SC>>,
    pub chip_information: Vec<(String, Dom<SC>, Dimensions)>,
    pub chip_ordering: HashMap<String, usize>,
}

impl<SC: StarkGenericConfig> StarkVerifyingKey<SC> {
    pub fn observe_into(&self, challenger: &mut SC::Challenger) {
        challenger.observe(self.commit.clone());
        challenger.observe(self.pc_start);
        challenger.observe_slice(&self.initial_global_cumulative_sum.0.x.0);
        challenger.observe_slice(&self.initial_global_cumulative_sum.0.y.0);
        challenger.observe(Val::<SC>::zero());
    }
}

impl<SC: StarkGenericConfig> Debug for StarkVerifyingKey<SC> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VerifyingKey").finish()
    }
}

impl<SC: StarkGenericConfig, A: MachineAir<Val<SC>>> StarkMachine<SC, A> {
    pub fn shard_chips_ordered<'a, 'b>(
        &'a self,
        chip_ordering: &'b HashMap<String, usize>,
    ) -> impl Iterator<Item = &'b MachineChip<SC, A>>
    where
        'a: 'b,
    {
        self.chips
            .iter()
            .filter(|chip| chip_ordering.contains_key(&chip.name()))
            .sorted_by_key(|chip| chip_ordering.get(&chip.name()))
    }

    pub const fn config(&self) -> &SC {
        &self.config
    }

    pub fn chips(&self) -> &[MachineChip<SC, A>] {
        &self.chips
    }

    pub const fn num_pv_elts(&self) -> usize {
        self.num_pv_elts
    }

    pub const fn contains_global_bus(&self) -> bool {
        self.contains_global_bus
    }

    pub fn chips_sorted_indices(&self, proof: &ShardProof<SC>) -> Vec<Option<usize>> {
        self.chips().iter().map(|chip| proof.chip_ordering.get(&chip.name()).copied()).collect()
    }

    #[instrument("verify", level = "info", skip_all)]
    #[allow(clippy::match_bool)]
    pub fn verify(
        &self,
        vk: &StarkVerifyingKey<SC>,
        proof: &MachineProof<SC>,
        challenger: &mut SC::Challenger,
    ) -> Result<(), MachineVerificationError<SC>>
    where
        SC::Challenger: Clone,
        A: for<'a> Air<VerifierConstraintFolder<'a, SC>>,
    {
        vk.observe_into(challenger);

        if proof.shard_proofs.is_empty() {
            return Err(MachineVerificationError::EmptyProof);
        }

        tracing::debug_span!("verify shard proofs").in_scope(|| {
            for (i, shard_proof) in proof.shard_proofs.iter().enumerate() {
                tracing::debug_span!("verifying shard", shard = i).in_scope(|| {
                    let chips =
                        self.shard_chips_ordered(&shard_proof.chip_ordering).collect::<Vec<_>>();
                    let mut shard_challenger = challenger.clone();
                    shard_challenger
                        .observe_slice(&shard_proof.public_values[0..self.num_pv_elts()]);
                    Verifier::verify_shard(
                        &self.config,
                        vk,
                        &chips,
                        &mut shard_challenger,
                        shard_proof,
                    )
                    .map_err(MachineVerificationError::InvalidShardProof)
                })?;
            }

            Ok(())
        })?;

        tracing::debug_span!("verify global cumulative sum is 0").in_scope(|| {
            let sum = proof
                .shard_proofs
                .iter()
                .map(ShardProof::global_cumulative_sum)
                .chain(once(vk.initial_global_cumulative_sum))
                .sum::<SepticDigest<Val<SC>>>();

            if !sum.is_zero() {
                return Err(MachineVerificationError::NonZeroCumulativeSum(
                    InteractionScope::Global,
                    0,
                ));
            }

            Ok(())
        })
    }
}

impl<SC: SCStarkGenericConfig, A: MachineAir<Val<SC>>, AE: MachineAir<Challenge<SC>>>
    SCStarkMachine<SC, A, AE>
{
    pub fn shard_chips_ordered<'a, 'b>(
        &'a self,
        chip_ordering: &'b HashMap<String, usize>,
    ) -> impl Iterator<Item = &'b MachineChip<SC, A>>
    where
        'a: 'b,
    {
        self.chips
            .iter()
            .filter(|chip| chip_ordering.contains_key(&chip.name()))
            .sorted_by_key(|chip| chip_ordering.get(&chip.name()))
    }

    pub fn shard_chips_ext_ordered<'a, 'b>(
        &'a self,
        chip_ordering: &'b HashMap<String, usize>,
    ) -> impl Iterator<Item = &'b SumcheckChip<SC, AE>>
    where
        'a: 'b,
    {
        self.chips_ext
            .iter()
            .filter(|chip| chip_ordering.contains_key(&chip.name()))
            .sorted_by_key(|chip| chip_ordering.get(&chip.name()))
    }

    pub const fn config(&self) -> &SC {
        &self.config
    }

    pub fn chips(&self) -> &[MachineChip<SC, A>] {
        &self.chips
    }

    pub fn chips_ext(&self) -> &[SumcheckChip<SC, AE>] {
        &self.chips_ext
    }

    pub const fn num_pv_elts(&self) -> usize {
        self.num_pv_elts
    }

    pub fn chips_sorted_indices(&self, proof: &ShardProof<SC>) -> Vec<Option<usize>> {
        self.chips().iter().map(|chip| proof.chip_ordering.get(&chip.name()).copied()).collect()
    }

    #[instrument("verify", level = "info", skip_all)]
    pub fn verify(
        &self,
        vk: &SCStarkVerifyingKey<SC>,
        proof: &SCMachineProof<SC>,
        challenger: &mut SC::MlChallenger,
        num_skip_rounds: usize,
        chip_log_height_threshold: usize,
    ) -> Result<(), MachineVerificationError<SC>>
    where
        SC::MlChallenger: Clone,
        A: for<'a> Air<crate::sumcheck::folder::SumcheckVerifierConstraintFolder<'a, SC>>,
    {
        tracing::info!("verify with univariate skip parameter k={}", num_skip_rounds);

        vk.observe_into(challenger);

        if proof.shard_proofs.is_empty() {
            return Err(MachineVerificationError::EmptyProof);
        }

        tracing::debug_span!("verify shard proofs v2").in_scope(|| {
            for (i, shard_proof) in proof.shard_proofs.iter().enumerate() {
                tracing::debug_span!("verifying shard v2", shard = i).in_scope(|| {
                    let chips =
                        self.shard_chips_ordered(&shard_proof.chip_ordering).collect::<Vec<_>>();
                    let mut shard_challenger = challenger.clone();
                    shard_challenger
                        .observe_slice(&shard_proof.public_values[0..self.num_pv_elts()]);
                    SumcheckVerifier::verify_shard(
                        &self.config,
                        vk,
                        &chips,
                        &mut shard_challenger,
                        shard_proof,
                        num_skip_rounds,
                        chip_log_height_threshold,
                        self.contains_global_bus,
                    )
                    .map_err(MachineVerificationError::InvalidShardProofSumcheck)
                })?;
            }

            Ok(())
        })?;

        tracing::debug_span!("verify global cumulative sum is 0").in_scope(|| {
            let sum = proof
                .shard_proofs
                .iter()
                .map(SCShardProof::global_cumulative_sum)
                .chain(once(vk.initial_global_cumulative_sum))
                .sum::<SepticDigest<Val<SC>>>();

            if !sum.is_zero() {
                return Err(MachineVerificationError::NonZeroCumulativeSum(
                    InteractionScope::Global,
                    0,
                ));
            }

            Ok(())
        })
    }
}

/// Errors that can occur during machine verification.
pub enum MachineVerificationError<SC: StarkGenericConfig> {
    InvalidShardProof(VerificationError<SC>),
    InvalidShardProofSumcheck(SumcheckVerificationError<SC>),
    InvalidGlobalProof(VerificationError<SC>),
    NonZeroCumulativeSum(InteractionScope, usize),
    InvalidPublicValuesDigest,
    DebugInteractionsFailed,
    EmptyProof,
    InvalidPublicValues(&'static str),
    TooManyShards,
    InvalidChipOccurrence(String),
    MissingCpuInFirstShard,
    CpuLogDegreeTooLarge(usize),
    InvalidVerificationKey,
}

impl<SC: StarkGenericConfig> Debug for MachineVerificationError<SC> {
    #[allow(clippy::uninlined_format_args)]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MachineVerificationError::InvalidShardProof(e) => {
                write!(f, "Invalid shard proof: {:?}", e)
            }
            MachineVerificationError::InvalidShardProofSumcheck(e) => {
                write!(f, "Invalid shard proof: {:?}", e)
            }
            MachineVerificationError::InvalidGlobalProof(e) => {
                write!(f, "Invalid global proof: {:?}", e)
            }
            MachineVerificationError::NonZeroCumulativeSum(scope, shard) => {
                write!(f, "Non-zero cumulative sum.  Scope: {}, Shard: {}", scope, shard)
            }
            MachineVerificationError::InvalidPublicValuesDigest => {
                write!(f, "Invalid public values digest")
            }
            MachineVerificationError::EmptyProof => write!(f, "Empty proof"),
            MachineVerificationError::DebugInteractionsFailed => {
                write!(f, "Debug interactions failed")
            }
            MachineVerificationError::InvalidPublicValues(s) => {
                write!(f, "Invalid public values: {}", s)
            }
            MachineVerificationError::TooManyShards => write!(f, "Too many shards"),
            MachineVerificationError::InvalidChipOccurrence(s) => {
                write!(f, "Invalid chip occurrence: {}", s)
            }
            MachineVerificationError::MissingCpuInFirstShard => {
                write!(f, "Missing CPU in first shard")
            }
            MachineVerificationError::CpuLogDegreeTooLarge(log_degree) => {
                write!(f, "CPU log degree too large: {}", log_degree)
            }
            MachineVerificationError::InvalidVerificationKey => {
                write!(f, "Invalid verification key")
            }
        }
    }
}

impl<SC: StarkGenericConfig> std::fmt::Display for MachineVerificationError<SC> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self, f)
    }
}

impl<SC: StarkGenericConfig> std::error::Error for MachineVerificationError<SC> {}
