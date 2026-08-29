#![allow(missing_docs)]
use crate::{Challenge, StarkGenericConfig, Val, ZeroCommitment};
use p3_challenger::{CanObserve, CanSample, FieldChallenger};
use p3_field::{ExtensionField, TwoAdicField};
use pcs::basefold::mlpcs::{MlCommitOptions, MlPCS};
use serde::{de::DeserializeOwned, Serialize};

// MlPCS related types
pub type MlChallenger<SC> = <SC as SCStarkGenericConfig>::MlChallenger;
pub type MlCom<SC> = <<SC as SCStarkGenericConfig>::Mlpcs as MlPCS>::Commitment;
pub type MlPcsOpeningProof<SC> = <<SC as SCStarkGenericConfig>::Mlpcs as MlPCS>::BatchProof;
pub type MlPcsProverData<SC> = <<SC as SCStarkGenericConfig>::Mlpcs as MlPCS>::ProverData;
pub type MlPcsVerificationTrace<SC> =
    <<SC as SCStarkGenericConfig>::Mlpcs as MlPCS>::VerificationTrace;

pub trait SCStarkGenericConfig:
    StarkGenericConfig + 'static + Send + Sync + Serialize + DeserializeOwned + Clone
{
    type Mlpcs: MlPCS<
            Field = Val<Self>,
            ExtensionField = Challenge<Self>,
            ProverData = Self::MlPcsProverData,
            Challenger = Self::MlChallenger,
        > + Sync
        + ZeroCommitment<Self>;

    type MlChallenge: TwoAdicField + ExtensionField<Self::Val>;

    type MlPcsProverData: Clone;

    type MlChallenger: FieldChallenger<Self::Val>
        + CanObserve<<Self::Mlpcs as MlPCS>::Commitment>
        + CanSample<Challenge<Self>>
        + Serialize
        + DeserializeOwned;

    /// Get the PCS used by this configuration.
    fn mlpcs(&self) -> &Self::Mlpcs;

    /// Commit options for each multilinear PCS commit call.
    fn mlpcs_commit_options(&self) -> MlCommitOptions {
        MlCommitOptions::default()
    }

    /// Optional circuit-level PCS stacking height hint.
    ///
    /// This is mainly needed for setup-time preprocessed commitments: those are fixed in the
    /// proving/verifying key before shard-specific main and permutation traces are known. Returning
    /// `Some(h)` forces every stacked commit for this circuit to use at least `2^h` rows.
    fn mlpcs_stack_log_height_hint(&self) -> Option<usize> {
        None
    }

    /// Resolve the stack height for one PCS commit batch.
    fn mlpcs_target_stack_log_height(&self, batch_max_log_height: Option<usize>) -> Option<usize> {
        match (self.mlpcs_stack_log_height_hint(), batch_max_log_height) {
            (Some(hint), Some(batch_max)) => Some(hint.max(batch_max)),
            (Some(hint), None) => Some(hint),
            (None, batch_max) => batch_max,
        }
    }

    /// Commit options using a resolved stack height when this PCS configuration enables stacking.
    fn mlpcs_commit_options_for_stack_log_height(
        &self,
        stack_log_height: Option<usize>,
    ) -> MlCommitOptions {
        if self.mlpcs_commit_options().stacking.is_none() {
            return MlCommitOptions::default();
        }

        stack_log_height
            .map_or_else(MlCommitOptions::auto_stacking, MlCommitOptions::stacking_log_height)
            .with_stacked_matrix_cache_from_env()
    }

    /// Initialize a new challenger.
    fn mlchallenger(&self) -> Self::MlChallenger;
}
