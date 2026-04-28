//! Proving and verifying key types for the sumcheck-based STARK.

use core::fmt::{Debug, Formatter};

use hashbrown::HashMap;
use p3_challenger::CanObserve;
use p3_field::AbstractField;
use p3_matrix::{dense::RowMajorMatrix, Dimensions};
use serde::{de::DeserializeOwned, Deserialize, Serialize};

use crate::config::Val;
use crate::septic_digest::SepticDigest;
use crate::sumcheck::config::{MlChallenger, MlCom, MlPcsProverData, SCStarkGenericConfig};
use crate::sumcheck::trace::CompressedMatrix;

/// Per-STARK proving key for the sumcheck protocol (stub, retained for type compatibility).
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "MlCom<SC>: Serialize, Val<SC>: Serialize, MlPcsProverData<SC>: Serialize, CompressedMatrix<Val<SC>, Val<SC>>: Serialize"
))]
#[serde(bound(
    deserialize = "MlCom<SC>: DeserializeOwned, Val<SC>: DeserializeOwned, MlPcsProverData<SC>: DeserializeOwned, CompressedMatrix<Val<SC>, Val<SC>>: DeserializeOwned"
))]
pub struct SCStarkProvingKey<SC: SCStarkGenericConfig> {
    pub commit: MlCom<SC>,
    pub pc_start: Val<SC>,
    pub initial_global_cumulative_sum: SepticDigest<Val<SC>>,
    pub traces: Vec<CompressedMatrix<Val<SC>, Val<SC>>>,
    pub data: MlPcsProverData<SC>,
    pub chip_ordering: HashMap<String, usize>,
    pub local_only: Vec<bool>,
    pub constraints_map: HashMap<String, usize>,
}

impl<SC: SCStarkGenericConfig> SCStarkProvingKey<SC> {
    pub fn get_preprocessed_trace(&self, index: usize) -> RowMajorMatrix<Val<SC>> {
        self.traces[index].decompress()
    }

    pub fn get_preprocessed_compressed_for_chips(
        &self,
        chip_names: &[String],
    ) -> Vec<Option<&CompressedMatrix<Val<SC>, Val<SC>>>> {
        chip_names
            .iter()
            .map(|name| self.chip_ordering.get(name).map(|&idx| &self.traces[idx]))
            .collect()
    }

    pub fn get_preprocessed_traces_for_open(
        &self,
        chip_names: &[String],
    ) -> Vec<Option<CompressedMatrix<Val<SC>>>> {
        chip_names
            .iter()
            .map(|name| self.chip_ordering.get(name).map(|&idx| self.traces[idx].clone()))
            .collect()
    }

    pub fn observe_into(&self, challenger: &mut SC::MlChallenger) {
        challenger.observe(self.commit.clone());
        challenger.observe(self.pc_start);
        challenger.observe_slice(&self.initial_global_cumulative_sum.0.x.0);
        challenger.observe_slice(&self.initial_global_cumulative_sum.0.y.0);
        challenger.observe(Val::<SC>::zero());
    }
}

pub trait SCMachineProvingKey<SC: SCStarkGenericConfig>: Send + Sync {
    fn preprocessed_commit(&self) -> MlCom<SC>;
    fn pc_start(&self) -> Val<SC>;
    fn initial_global_cumulative_sum(&self) -> SepticDigest<Val<SC>>;
    fn observe_into(&self, challenger: &mut MlChallenger<SC>);
}

impl<SC> SCMachineProvingKey<SC> for SCStarkProvingKey<SC>
where
    SC: 'static + SCStarkGenericConfig,
    MlPcsProverData<SC>: Send + Sync,
    MlCom<SC>: Send + Sync,
{
    fn preprocessed_commit(&self) -> MlCom<SC> {
        self.commit.clone()
    }

    fn pc_start(&self) -> Val<SC> {
        self.pc_start
    }

    fn initial_global_cumulative_sum(&self) -> SepticDigest<Val<SC>> {
        self.initial_global_cumulative_sum
    }

    fn observe_into(&self, challenger: &mut MlChallenger<SC>) {
        challenger.observe(self.commit.clone());
        challenger.observe(self.pc_start);
        challenger.observe_slice(&self.initial_global_cumulative_sum.0.x.0);
        challenger.observe_slice(&self.initial_global_cumulative_sum.0.y.0);
        challenger.observe(Val::<SC>::zero());
    }
}

/// Per-STARK verifying key for the sumcheck protocol.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "MlCom<SC>: Serialize"))]
#[serde(bound(deserialize = "MlCom<SC>: DeserializeOwned"))]
pub struct SCStarkVerifyingKey<SC: SCStarkGenericConfig> {
    pub commit: MlCom<SC>,
    pub pc_start: Val<SC>,
    pub initial_global_cumulative_sum: SepticDigest<Val<SC>>,
    pub chip_information: Vec<(String, Dimensions)>,
    pub chip_ordering: HashMap<String, usize>,
    pub constraints_map: HashMap<String, usize>,
}

impl<SC: SCStarkGenericConfig> Debug for SCStarkVerifyingKey<SC> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VerifyingKey").finish()
    }
}

impl<SC: SCStarkGenericConfig> SCStarkVerifyingKey<SC> {
    pub fn observe_into(&self, challenger: &mut SC::MlChallenger) {
        challenger.observe(self.commit.clone());
        challenger.observe(self.pc_start);
        challenger.observe_slice(&self.initial_global_cumulative_sum.0.x.0);
        challenger.observe_slice(&self.initial_global_cumulative_sum.0.y.0);
        challenger.observe(Val::<SC>::zero());
    }
}
