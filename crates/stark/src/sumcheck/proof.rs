//! Proof-related types for the sumcheck path: sumcheck proof, shard data, commitment,
//! chip/shard opened values, shard proof, machine proof.

use core::fmt::Debug;
use hashbrown::HashMap;
use itertools::Itertools;
use serde::{Deserialize, Serialize};

use crate::shape::OrderedShape;
use crate::sumcheck::config::{MlCom, MlPcsOpeningProof, SCStarkGenericConfig};
use crate::sumcheck::types::UniPolyEvals;
use crate::StarkGenericConfig;
use crate::{
    config::{Challenge, Val},
    septic_digest::SepticDigest,
    SCAirOpenedValues,
};
use p3_matrix::Dimensions;

// ---------- Sumcheck proof ----------

#[derive(Serialize, Deserialize, Clone)]
#[serde(bound = "")]
pub struct SumcheckProof<SC: StarkGenericConfig> {
    pub unipolys: Vec<UniPolyEvals<Challenge<SC>>>,
}

// ---------- Shard data ----------

/// Shard data: holds compressed main traces and PCS commitment data.
#[allow(clippy::type_complexity)]
pub struct SCShardMainData<SC: SCStarkGenericConfig, M, P> {
    pub compressed_traces: Vec<(String, M)>,
    pub main_commit: MlCom<SC>,
    pub main_data: P,
    pub chip_ordering: HashMap<String, usize>,
    pub public_values: Vec<SC::Val>,
}

// ---------- Commitment ----------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SCShardCommitment<C> {
    pub main_commit: C,
    pub permutation_commit: Option<C>,
}

// ---------- Chip / shard opened values ----------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize, EF: Serialize"))]
#[serde(bound(deserialize = "F: Deserialize<'de>, EF: Deserialize<'de>"))]
pub struct SCChipOpenedValues<F, EF> {
    pub preprocessed: SCAirOpenedValues<EF>,
    pub main: SCAirOpenedValues<EF>,
    pub permutation: SCAirOpenedValues<EF>,
    pub global_cumulative_sum: SepticDigest<F>,
    pub local_cumulative_sum: EF,
    pub log_height: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(bound(serialize = "F: Serialize, EF: Serialize"))]
#[serde(bound(deserialize = "F: Deserialize<'de>, EF: Deserialize<'de>"))]
pub struct SCShardOpenedValues<F, EF> {
    pub chips: Vec<SCChipOpenedValues<F, EF>>,
}

// ---------- Shard proof ----------

#[derive(Serialize, Deserialize, Clone)]
#[serde(bound = "")]
pub struct SCShardProof<SC: SCStarkGenericConfig> {
    pub commitment: SCShardCommitment<MlCom<SC>>,
    pub opened_values: SCShardOpenedValues<Val<SC>, Challenge<SC>>,
    pub opening_proof: MlPcsOpeningProof<SC>,
    pub sumcheck_proof: SumcheckProof<SC>,
    pub dimensions: Vec<Vec<Dimensions>>,
    pub chip_ordering: HashMap<String, usize>,
    pub public_values: Vec<Val<SC>>,
}

impl<SC: SCStarkGenericConfig> Debug for SCShardProof<SC> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SCShardProof").finish()
    }
}

impl<SC: SCStarkGenericConfig> SCShardProof<SC> {
    #[must_use]
    pub fn local_cumulative_sum(&self) -> Challenge<SC> {
        self.opened_values.chips.iter().map(|c| c.local_cumulative_sum).sum()
    }

    #[must_use]
    pub fn global_cumulative_sum(&self) -> SepticDigest<Val<SC>> {
        self.opened_values.chips.iter().map(|c| c.global_cumulative_sum).sum()
    }

    #[must_use]
    pub fn shape(&self) -> OrderedShape {
        OrderedShape {
            inner: self
                .chip_ordering
                .iter()
                .sorted_by_key(|(_, idx)| *idx)
                .zip(self.opened_values.chips.iter())
                .map(|((name, _), chip)| (name.clone(), chip.log_height))
                .collect(),
        }
    }

    /// Check whether this shard contains execution (CPU) events.
    ///
    /// After the chip-split refactor, there is no single "Cpu" chip. Instead, we
    /// detect execution shards by examining the public values: if `start_clk != exit_clk`,
    /// the shard contains CPU instruction events.
    #[must_use]
    pub fn contains_cpu(&self) -> bool {
        use crate::air::PublicValues;
        use crate::Word;
        use std::borrow::Borrow;
        let pv: &PublicValues<Word<Val<SC>>, Val<SC>> = self.public_values.as_slice().borrow();
        pv.start_clk != pv.exit_clk
    }

    #[must_use]
    pub fn log_degree_cpu(&self) -> usize {
        // After the chip-split refactor, there is no single "Cpu" chip.
        // Return the max log height among all instruction chips as a proxy.
        self.opened_values.chips.iter().map(|c| c.log_height).max().unwrap_or(0)
    }

    /// Whether this shard proof includes the `MemoryGlobalInit` chip.
    #[must_use]
    pub fn contains_global_memory_init(&self) -> bool {
        self.chip_ordering.contains_key("MemoryGlobalInit")
    }

    /// Whether this shard proof includes the `MemoryGlobalFinalize` chip.
    #[must_use]
    pub fn contains_global_memory_finalize(&self) -> bool {
        self.chip_ordering.contains_key("MemoryGlobalFinalize")
    }
}

// ---------- Machine proof ----------

#[derive(Serialize, Deserialize, Clone)]
#[serde(bound = "")]
pub struct SCMachineProof<SC: SCStarkGenericConfig> {
    pub shard_proofs: Vec<SCShardProof<SC>>,
}

impl<SC: SCStarkGenericConfig> Debug for SCMachineProof<SC> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SCMachineProof").field("shard_proofs", &self.shard_proofs.len()).finish()
    }
}
