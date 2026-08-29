//! Proving and verifying key types for the sumcheck-based STARK.
//!
//! - [`SCStarkProvingKey`]: per-STARK proving key holding preprocessed compressed traces.
//! - [`SCStarkVerifyingKey`]: per-STARK verifying key with chip metadata and commitments.
//! - [`SCMachineProvingKey`]: trait abstracting machine-level proving key operations.

use core::fmt::{Debug, Formatter};

use hashbrown::HashMap;
use p3_challenger::CanObserve;
use p3_field::AbstractField;
use p3_matrix::{dense::RowMajorMatrix, Dimensions};
use serde::{de::DeserializeOwned, Deserialize, Serialize};

use crate::{
    config::Val,
    global_d11::{
        canonical_program_boundary_fields_v1, observe_program_global_metadata_v2,
        validate_global146_identity, BoundaryOwnerRegistryV2, ProgramImageBoundaryV1,
    },
    sumcheck::{
        config::{MlChallenger, MlCom, MlPcsProverData, SCStarkGenericConfig},
        trace::CompressedMatrix,
    },
};

/// Per-STARK proving key for the sumcheck protocol.
///
/// Stores preprocessed compressed traces and PCS commitment data needed by the prover.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "MlCom<SC>: Serialize, Val<SC>: Serialize, MlPcsProverData<SC>: Serialize, CompressedMatrix<Val<SC>, Val<SC>>: Serialize"
))]
#[serde(bound(
    deserialize = "MlCom<SC>: DeserializeOwned, Val<SC>: DeserializeOwned, MlPcsProverData<SC>: DeserializeOwned, CompressedMatrix<Val<SC>, Val<SC>>: DeserializeOwned"
))]
pub struct SCStarkProvingKey<SC: SCStarkGenericConfig> {
    /// PCS commitment to the preprocessed traces.
    pub commit: MlCom<SC>,
    /// Program counter start value.
    pub pc_start: Val<SC>,
    /// Program-image Global boundary.
    pub program_boundary: ProgramImageBoundaryV1<u32>,
    /// Ordered Global owner inventory.
    pub owner_registry: BoundaryOwnerRegistryV2,
    /// Product-local Global146 protocol identity.
    pub global146_identity: [u8; 32],
    /// Compressed preprocessed traces, one per chip.
    pub traces: Vec<CompressedMatrix<Val<SC>, Val<SC>>>,
    /// PCS prover data for the preprocessed commitment.
    pub data: MlPcsProverData<SC>,
    /// Stack height used by the preprocessed PCS commitment, when commit-time stacking is enabled.
    #[serde(default)]
    pub preprocessed_pcs_stack_log_height: Option<usize>,
    /// Map from chip name to its index in the ordering.
    pub chip_ordering: HashMap<String, usize>,
    /// Whether each chip uses only local interactions.
    pub local_only: Vec<bool>,
    /// Map from chip name to its constraint count.
    pub constraints_map: HashMap<String, usize>,
}

impl<SC: SCStarkGenericConfig> SCStarkProvingKey<SC> {
    /// Decompress and return the preprocessed trace at the given chip index.
    pub fn get_preprocessed_trace(&self, index: usize) -> RowMajorMatrix<Val<SC>> {
        self.traces[index].decompress()
    }

    /// Look up compressed preprocessed traces for the given chip names.
    ///
    /// Returns `None` for chips that have no preprocessed trace.
    pub fn get_preprocessed_compressed_for_chips(
        &self,
        chip_names: &[String],
    ) -> Vec<Option<&CompressedMatrix<Val<SC>, Val<SC>>>> {
        chip_names
            .iter()
            .map(|name| self.chip_ordering.get(name).map(|&idx| &self.traces[idx]))
            .collect()
    }

    /// Return compressed preprocessed traces for the given chip names (used during PCS opening).
    pub fn get_preprocessed_traces_for_open(
        &self,
        chip_names: &[String],
    ) -> Vec<Option<CompressedMatrix<Val<SC>>>> {
        chip_names
            .iter()
            .map(|name| self.chip_ordering.get(name).map(|&idx| self.traces[idx].clone()))
            .collect()
    }

    /// Observe the proving key into the Fiat-Shamir challenger.
    pub fn observe_into(&self, challenger: &mut SC::MlChallenger) {
        validate_global146_identity(&self.global146_identity)
            .expect("admitted proving key has the current Global146 identity");
        challenger.observe(Val::<SC>::from_canonical_u32(0x3156_4b47));
        challenger.observe(Val::<SC>::from_canonical_u32(1));
        challenger.observe(self.commit.clone());
        observe_program_global_metadata_v2::<Val<SC>, _>(
            challenger,
            self.pc_start,
            &self.program_boundary,
            &self.owner_registry,
        )
        .expect("validated proving key has canonical Global metadata");
    }
}

/// Trait abstracting machine-level proving key operations.
pub trait SCMachineProvingKey<SC: SCStarkGenericConfig>: Send + Sync {
    /// Return the PCS commitment to preprocessed traces.
    fn preprocessed_commit(&self) -> MlCom<SC>;
    /// Return the program counter start value.
    fn pc_start(&self) -> Val<SC>;
    /// Observe the key into the Fiat-Shamir challenger.
    fn observe_into(&self, challenger: &mut MlChallenger<SC>);
    /// Return the setup-time preprocessed PCS stack height, if any.
    fn preprocessed_pcs_stack_log_height(&self) -> Option<usize>;
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

    fn observe_into(&self, challenger: &mut MlChallenger<SC>) {
        SCStarkProvingKey::observe_into(self, challenger);
    }

    fn preprocessed_pcs_stack_log_height(&self) -> Option<usize> {
        self.preprocessed_pcs_stack_log_height
    }
}

/// Per-STARK verifying key for the sumcheck protocol.
///
/// Contains the preprocessed commitment, chip metadata, and constraint counts
/// needed by the verifier.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "MlCom<SC>: Serialize"))]
#[serde(bound(deserialize = "MlCom<SC>: DeserializeOwned"))]
pub struct SCStarkVerifyingKey<SC: SCStarkGenericConfig> {
    /// PCS commitment to the preprocessed traces.
    pub commit: MlCom<SC>,
    /// Program counter start value.
    pub pc_start: Val<SC>,
    /// Program-image Global boundary.
    pub program_boundary: ProgramImageBoundaryV1<u32>,
    /// Ordered Global owner inventory.
    pub owner_registry: BoundaryOwnerRegistryV2,
    /// Product-local Global146 protocol identity.
    pub global146_identity: [u8; 32],
    /// Per-chip name and dimensions (width × height).
    pub chip_information: Vec<(String, Dimensions)>,
    /// Map from chip name to its index in the ordering.
    pub chip_ordering: HashMap<String, usize>,
    /// Map from chip name to its constraint count.
    pub constraints_map: HashMap<String, usize>,
}

impl<SC: SCStarkGenericConfig> Debug for SCStarkVerifyingKey<SC> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VerifyingKey").finish()
    }
}

impl<SC: SCStarkGenericConfig> SCStarkVerifyingKey<SC> {
    /// Observe the verifying key into the Fiat-Shamir challenger.
    pub fn observe_into(&self, challenger: &mut SC::MlChallenger) {
        validate_global146_identity(&self.global146_identity)
            .expect("admitted verifying key has the current Global146 identity");
        challenger.observe(Val::<SC>::from_canonical_u32(0x3156_4b47));
        challenger.observe(Val::<SC>::from_canonical_u32(1));
        challenger.observe(self.commit.clone());
        observe_program_global_metadata_v2::<Val<SC>, _>(
            challenger,
            self.pc_start,
            &self.program_boundary,
            &self.owner_registry,
        )
        .expect("validated verifying key has canonical Global metadata");
    }

    /// Canonical host hash input matching `SCVerifyingKeyVariable::hash`.
    pub fn canonical_hash_inputs(&self) -> Vec<Val<SC>>
    where
        MlCom<SC>: AsRef<[Val<SC>; crate::DIGEST_SIZE]>,
        Val<SC>: p3_field::PrimeField32,
    {
        validate_global146_identity(&self.global146_identity)
            .expect("hashed SC VK has the current Global146 identity");
        self.owner_registry.validate().expect("hashed SC VK has a canonical owner registry");
        let mut inputs = Vec::new();
        inputs.push(Val::<SC>::from_canonical_u32(0x3156_4b47));
        inputs.push(Val::<SC>::one());
        inputs.extend_from_slice(self.commit.as_ref());
        inputs.push(self.pc_start);
        inputs.extend(
            canonical_program_boundary_fields_v1::<Val<SC>>(&self.program_boundary)
                .expect("hashed SC VK has a canonical program boundary"),
        );
        inputs.extend(self.owner_registry.digest.map(Val::<SC>::from_canonical_u8));
        inputs.extend(self.global146_identity.map(Val::<SC>::from_canonical_u8));
        for (name, dimensions) in &self.chip_information {
            inputs.push(Val::<SC>::from_canonical_usize(dimensions.width));
            inputs.push(Val::<SC>::from_canonical_usize(dimensions.height));
            inputs.push(Val::<SC>::from_canonical_usize(name.len()));
            inputs.extend(name.bytes().map(Val::<SC>::from_canonical_u8));
        }
        inputs
    }
}
