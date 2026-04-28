use dt_primitives::{sc_poseidon2_hash, SCField};
use dt_stark::koalabear_poseidon2::koala_bear_poseidon2::SCKoalaBearPoseidon2;
use dt_stark::sumcheck::config::SCStarkGenericConfig;
use dt_stark::sumcheck::keys::SCStarkVerifyingKey;
use dt_stark::sumcheck::proof::SCShardProof;
use dt_stark::DIGEST_SIZE;
use p3_field::{AbstractField, PrimeField32};
use serde::{Deserialize, Serialize};

use basefold::basefold::mlpcs::MlPCS;

pub type InnerSC = SCKoalaBearPoseidon2;

pub type CoreSC = InnerSC;

#[derive(Clone, Serialize, Deserialize)]
pub struct DTVerifyingKey {
    pub vk: SCStarkVerifyingKey<CoreSC>,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "SCShardProof<SC>: Serialize"))]
#[serde(bound(deserialize = "SCShardProof<SC>: Deserialize<'de>"))]
pub struct DTReduceProof<SC: SCStarkGenericConfig> {
    pub vk: SCStarkVerifyingKey<SC>,
    pub proof: SCShardProof<SC>,
}

impl<SC: SCStarkGenericConfig> std::fmt::Debug for DTReduceProof<SC> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DTReduceProof").finish_non_exhaustive()
    }
}

pub trait HashableKey {
    fn hash_babybear(&self) -> [SCField; DIGEST_SIZE];
    fn hash_u32(&self) -> [u32; DIGEST_SIZE];
}

impl HashableKey for DTVerifyingKey {
    fn hash_babybear(&self) -> [SCField; DIGEST_SIZE] {
        self.vk.hash_babybear()
    }

    fn hash_u32(&self) -> [u32; DIGEST_SIZE] {
        self.vk.hash_u32()
    }
}

impl<SC: SCStarkGenericConfig<Val = SCField>> HashableKey for SCStarkVerifyingKey<SC>
where
    <SC::Mlpcs as MlPCS>::Commitment: AsRef<[SCField; DIGEST_SIZE]>,
{
    fn hash_babybear(&self) -> [SCField; DIGEST_SIZE] {
        let mut num_inputs = DIGEST_SIZE + 1 + 14 + (7 * self.chip_information.len());
        for (name, _) in self.chip_information.iter() {
            num_inputs += name.len();
        }
        let mut inputs = Vec::with_capacity(num_inputs);
        inputs.extend(self.commit.as_ref());
        inputs.push(self.pc_start);
        inputs.extend(self.initial_global_cumulative_sum.0.x.0);
        inputs.extend(self.initial_global_cumulative_sum.0.y.0);
        for (name, dimension) in self.chip_information.iter() {
            inputs.push(SCField::from_canonical_usize(dimension.width));
            inputs.push(SCField::from_canonical_usize(dimension.height));
            inputs.push(SCField::from_canonical_usize(name.len()));
            for byte in name.as_bytes() {
                inputs.push(SCField::from_canonical_u8(*byte));
            }
        }

        sc_poseidon2_hash(inputs)
    }

    fn hash_u32(&self) -> [u32; DIGEST_SIZE] {
        self.hash_babybear()
            .into_iter()
            .map(|n| n.as_canonical_u32())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }
}
