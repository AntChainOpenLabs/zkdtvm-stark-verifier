use super::{
    SCDTCompressWitnessValues, SCDTCompressWitnessVariable, SCDTMerkleProofWitnessValues,
    SCDTMerkleProofWitnessVariable, SCDTRecursionWitnessValues, SCDTRecursionWitnessVariable,
};
use crate::{
    hash::FieldHasher,
    sc_machine::SCDTDeferredWitnessValues,
    sumcheck::{
        types::{BasefoldProofVariable, SCVerifyingKeyVariable, SumcheckProofVariable},
        SCBabyBearFriConfigVariable,
    },
    witness::{WitnessWriter, Witnessable},
    Builder, CircuitConfig, FieldHasherVariable,
};
use dt_recursion_compiler::prelude::Felt;

use crate::{sc_machine::SCDTDeferredWitnessVariable, sumcheck::types::SCShardProofVariable};
#[cfg(feature = "koalabear")]
use dt_primitives::SCField;
#[cfg(feature = "koalabear")]
use dt_stark::koalabear_poseidon2::koala_bear_poseidon2::SCKoalaBearPoseidon2;
use dt_stark::{
    baby_bear_poseidon2::SCBabyBearPoseidon2,
    sumcheck::{
        config::{MlCom, MlPcsOpeningProof},
        keys::SCStarkVerifyingKey,
        proof::{SCShardProof, SumcheckProof},
    },
    Challenge, InnerChallenge, InnerVal, Val,
};
use p3_field::{AbstractField, PrimeField32};
impl<
        C: CircuitConfig<F = Val<SC>, EF = Challenge<SC>>,
        SC: SCBabyBearFriConfigVariable<C> + dt_stark::StarkGenericConfig,
    > Witnessable<C> for SCStarkVerifyingKey<SC>
where
    MlCom<SC>: Witnessable<C, WitnessVariable = <SC as FieldHasherVariable<C>>::DigestVariable>,
    MlPcsOpeningProof<SC>: Witnessable<C, WitnessVariable = BasefoldProofVariable<C, SC>>,
    SumcheckProof<SC>: Witnessable<C, WitnessVariable = SumcheckProofVariable<C>>,
    Val<SC>: Witnessable<C, WitnessVariable = Felt<C::F>>,
{
    type WitnessVariable = SCVerifyingKeyVariable<C, SC>;
    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        dt_stark::global_d11::validate_global146_identity(&self.global146_identity)
            .expect("SC VK witness has the current Global146 identity");
        let commitment = self.commit.read(builder);
        let pc_start = self.pc_start.read(builder);
        self.owner_registry.validate().expect("validated VK has a canonical owner registry");
        let seed = dt_stark::global_d11::program_global_seed::<Val<SC>>(&self.program_boundary)
            .expect("validated VK has a canonical ProgramGlobalSeed");
        let program_global_seed =
            [*seed.x.coefficients(), *seed.y.coefficients(), *seed.z.coefficients()].read(builder);
        let program_global_digest =
            self.owner_registry.digest.map(Val::<SC>::from_canonical_u8).read(builder);
        let has_global_owner = !self.owner_registry.owners.is_empty();
        let chip_information = self.chip_information.clone();
        let chip_ordering = self.chip_ordering.clone();
        let constraints_map = self.constraints_map.clone();
        SCVerifyingKeyVariable {
            commitment,
            pc_start,
            program_global_seed,
            program_global_digest,
            has_global_owner,
            chip_information,
            chip_ordering,
            constraints_map,
        }
    }
    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        dt_stark::global_d11::validate_global146_identity(&self.global146_identity)
            .expect("SC VK witness has the current Global146 identity");
        self.commit.write(witness);
        self.pc_start.write(witness);
        self.owner_registry.validate().expect("validated VK has a canonical owner registry");
        let seed = dt_stark::global_d11::program_global_seed::<Val<SC>>(&self.program_boundary)
            .expect("validated VK has a canonical ProgramGlobalSeed");
        [*seed.x.coefficients(), *seed.y.coefficients(), *seed.z.coefficients()].write(witness);
        self.owner_registry.digest.map(Val::<SC>::from_canonical_u8).write(witness);
    }
}
impl<
        C: CircuitConfig<F = Val<SC>, EF = Challenge<SC>>,
        SC: SCBabyBearFriConfigVariable<C> + dt_stark::StarkGenericConfig,
    > Witnessable<C> for SCDTCompressWitnessValues<SC>
where
    MlCom<SC>: Witnessable<C, WitnessVariable = <SC as FieldHasherVariable<C>>::DigestVariable>,
    MlPcsOpeningProof<SC>: Witnessable<C, WitnessVariable = BasefoldProofVariable<C, SC>>,
    Val<SC>: Witnessable<C, WitnessVariable = Felt<C::F>>,
    SumcheckProof<SC>: Witnessable<C, WitnessVariable = SumcheckProofVariable<C>>,
    SCShardProof<SC>: Witnessable<C, WitnessVariable = SCShardProofVariable<C, SC>>,
    SCStarkVerifyingKey<SC>: Witnessable<C, WitnessVariable = SCVerifyingKeyVariable<C, SC>>,
{
    type WitnessVariable = SCDTCompressWitnessVariable<C, SC>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let vks_and_proofs = self.vks_and_proofs.read(builder);
        let is_complete = Val::<SC>::from_bool(self.is_complete).read(builder);

        SCDTCompressWitnessVariable { vks_and_proofs, is_complete }
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        self.vks_and_proofs.write(witness);
        Val::<SC>::from_bool(self.is_complete).write(witness);
    }
}
impl<C: CircuitConfig<F = SC::Val>, SC: SCBabyBearFriConfigVariable<C>> Witnessable<C>
    for SCDTMerkleProofWitnessValues<SC>
where
    // This trait bound is redundant, but Rust-Analyzer is not able to infer it.
    SC: FieldHasher<Val<SC>>,
    <SC as FieldHasher<Val<SC>>>::Digest: Witnessable<C, WitnessVariable = SC::DigestVariable>,
{
    type WitnessVariable = SCDTMerkleProofWitnessVariable<C, SC>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        SCDTMerkleProofWitnessVariable {
            vk_merkle_proofs: self.vk_merkle_proofs.read(builder),
            values: self.values.read(builder),
            root: self.root.read(builder),
        }
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        self.vk_merkle_proofs.write(witness);
        self.values.write(witness);
        self.root.write(witness);
    }
}

#[cfg(feature = "babybear")]
impl<C> Witnessable<C> for SCDTRecursionWitnessValues<SCBabyBearPoseidon2>
where
    C: CircuitConfig<F = InnerVal, EF = InnerChallenge, Bit = Felt<InnerVal>>,
{
    type WitnessVariable = SCDTRecursionWitnessVariable<C, SCBabyBearPoseidon2>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let vk = self.vk.read(builder);
        let shard_proofs = self.shard_proofs.read(builder);
        let reconstruct_deferred_digest = self.reconstruct_deferred_digest.read(builder);
        let is_complete = InnerVal::from_bool(self.is_complete).read(builder);
        let is_first_shard = InnerVal::from_bool(self.is_first_shard).read(builder);
        let vk_root = self.vk_root.read(builder);
        SCDTRecursionWitnessVariable {
            vk,
            shard_proofs,
            is_complete,
            is_first_shard,
            reconstruct_deferred_digest,
            vk_root,
        }
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        self.vk.write(witness);
        self.shard_proofs.write(witness);
        self.reconstruct_deferred_digest.write(witness);
        self.is_complete.write(witness);
        self.is_first_shard.write(witness);
        self.vk_root.write(witness);
    }
}
#[cfg(feature = "babybear")]
impl<C> Witnessable<C> for SCDTDeferredWitnessValues<SCBabyBearPoseidon2>
where
    C: CircuitConfig<F = InnerVal, EF = InnerChallenge, Bit = Felt<InnerVal>>,
{
    type WitnessVariable = SCDTDeferredWitnessVariable<C, SCBabyBearPoseidon2>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let vks_and_proofs = self.vks_and_proofs.read(builder);
        let vk_merkle_data = self.vk_merkle_data.read(builder);
        let start_reconstruct_deferred_digest =
            self.start_reconstruct_deferred_digest.read(builder);
        let dt_vk_digest = self.dt_vk_digest.read(builder);
        let committed_value_digest = self.committed_value_digest.read(builder);
        let deferred_proofs_digest = self.deferred_proofs_digest.read(builder);
        let end_pc = self.end_pc.read(builder);
        let end_shard = self.end_shard.read(builder);
        let end_execution_shard = self.end_execution_shard.read(builder);
        let init_addr = self.init_addr.read(builder);
        let finalize_addr = self.finalize_addr.read(builder);
        let is_complete = InnerVal::from_bool(self.is_complete).read(builder);

        SCDTDeferredWitnessVariable {
            vks_and_proofs,
            vk_merkle_data,
            start_reconstruct_deferred_digest,
            dt_vk_digest,
            committed_value_digest,
            deferred_proofs_digest,
            end_pc,
            end_shard,
            end_execution_shard,
            init_addr,
            finalize_addr,
            is_complete,
        }
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        self.vks_and_proofs.write(witness);
        self.vk_merkle_data.write(witness);
        self.start_reconstruct_deferred_digest.write(witness);
        self.dt_vk_digest.write(witness);
        self.committed_value_digest.write(witness);
        self.deferred_proofs_digest.write(witness);
        self.end_pc.write(witness);
        self.end_shard.write(witness);
        self.end_execution_shard.write(witness);
        self.init_addr.write(witness);
        self.finalize_addr.write(witness);
        self.is_complete.write(witness);
    }
}

#[cfg(feature = "koalabear")]
impl<C> Witnessable<C> for SCDTRecursionWitnessValues<SCKoalaBearPoseidon2>
where
    C: CircuitConfig<F = SCField, EF = Challenge<SCKoalaBearPoseidon2>, Bit = Felt<SCField>>,
{
    type WitnessVariable = SCDTRecursionWitnessVariable<C, SCKoalaBearPoseidon2>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let vk = self.vk.read(builder);
        let shard_proofs = self.shard_proofs.read(builder);
        let reconstruct_deferred_digest = self.reconstruct_deferred_digest.read(builder);
        let is_complete = SCField::from_bool(self.is_complete).read(builder);
        let is_first_shard = SCField::from_bool(self.is_first_shard).read(builder);
        let vk_root = self.vk_root.read(builder);
        SCDTRecursionWitnessVariable {
            vk,
            shard_proofs,
            is_complete,
            is_first_shard,
            reconstruct_deferred_digest,
            vk_root,
        }
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        self.vk.write(witness);
        self.shard_proofs.write(witness);
        self.reconstruct_deferred_digest.write(witness);
        self.is_complete.write(witness);
        self.is_first_shard.write(witness);
        self.vk_root.write(witness);
    }
}

#[cfg(feature = "koalabear")]
impl<C> Witnessable<C> for SCDTDeferredWitnessValues<SCKoalaBearPoseidon2>
where
    C: CircuitConfig<F = SCField, EF = Challenge<SCKoalaBearPoseidon2>, Bit = Felt<SCField>>,
{
    type WitnessVariable = SCDTDeferredWitnessVariable<C, SCKoalaBearPoseidon2>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let vks_and_proofs = self.vks_and_proofs.read(builder);
        let vk_merkle_data = self.vk_merkle_data.read(builder);
        let start_reconstruct_deferred_digest =
            self.start_reconstruct_deferred_digest.read(builder);
        let dt_vk_digest = self.dt_vk_digest.read(builder);
        let committed_value_digest = self.committed_value_digest.read(builder);
        let deferred_proofs_digest = self.deferred_proofs_digest.read(builder);
        let end_pc = self.end_pc.read(builder);
        let end_shard = self.end_shard.read(builder);
        let end_execution_shard = self.end_execution_shard.read(builder);
        let init_addr = self.init_addr.read(builder);
        let finalize_addr = self.finalize_addr.read(builder);
        let is_complete = SCField::from_bool(self.is_complete).read(builder);

        SCDTDeferredWitnessVariable {
            vks_and_proofs,
            vk_merkle_data,
            start_reconstruct_deferred_digest,
            dt_vk_digest,
            committed_value_digest,
            deferred_proofs_digest,
            end_pc,
            end_shard,
            end_execution_shard,
            init_addr,
            finalize_addr,
            is_complete,
        }
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        self.vks_and_proofs.write(witness);
        self.vk_merkle_data.write(witness);
        self.start_reconstruct_deferred_digest.write(witness);
        self.dt_vk_digest.write(witness);
        self.committed_value_digest.write(witness);
        self.deferred_proofs_digest.write(witness);
        self.end_pc.write(witness);
        self.end_shard.write(witness);
        self.end_execution_shard.write(witness);
        self.init_addr.write(witness);
        self.finalize_addr.write(witness);
        self.is_complete.write(witness);
    }
}
