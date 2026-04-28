#![allow(missing_docs)]
use crate::{Challenge, StarkGenericConfig, Val, ZeroCommitment};
use basefold::basefold::mlpcs::MlPCS;
use p3_challenger::{CanObserve, CanSample, FieldChallenger};
use p3_field::{ExtensionField, TwoAdicField};
use serde::{de::DeserializeOwned, Serialize};

pub type MlChallenger<SC> = <SC as SCStarkGenericConfig>::MlChallenger;
pub type MlCom<SC> = <<SC as SCStarkGenericConfig>::Mlpcs as MlPCS>::Commitment;
pub type MlPcsOpeningProof<SC> = <<SC as SCStarkGenericConfig>::Mlpcs as MlPCS>::BatchProof;
pub type MlPcsProverData<SC> = <<SC as SCStarkGenericConfig>::Mlpcs as MlPCS>::ProverData;

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

    fn mlpcs(&self) -> &Self::Mlpcs;

    fn mlchallenger(&self) -> Self::MlChallenger;
}
