use std::borrow::Borrow;

use p3_baby_bear::BabyBear;
use p3_challenger::DuplexChallenger;
use p3_symmetric::Hash;

use dt_recursion_compiler::ir::Builder;
use dt_stark::{
    baby_bear_poseidon2::BabyBearPoseidon2, Com, InnerChallenge, InnerPerm, InnerVal, OpeningProof,
    StarkVerifyingKey, Word,
};
use p3_field::{AbstractField, PrimeField32};

use dt_recursion_compiler::ir::Felt;

use crate::{
    challenger::DuplexChallengerVariable,
    hash::{FieldHasher, FieldHasherVariable},
    merkle_tree::MerkleProof,
    stark::MerkleProofVariable,
    witness::{WitnessWriter, Witnessable},
    BabyBearFriConfigVariable, CircuitConfig, TwoAdicPcsProofVariable, VerifyingKeyVariable,
};

use super::{
    DTCompressWitnessValues, DTCompressWitnessVariable, DTDeferredWitnessValues,
    DTDeferredWitnessVariable, DTMerkleProofWitnessValues, DTMerkleProofWitnessVariable,
    DTRecursionWitnessValues, DTRecursionWitnessVariable,
};

impl<C: CircuitConfig, T: Witnessable<C>> Witnessable<C> for Word<T> {
    type WitnessVariable = Word<T::WitnessVariable>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        Word(self.0.read(builder))
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        self.0.write(witness);
    }
}

impl<C> Witnessable<C> for DuplexChallenger<InnerVal, InnerPerm, 16, 8>
where
    C: CircuitConfig<F = InnerVal, EF = InnerChallenge>,
{
    type WitnessVariable = DuplexChallengerVariable<C>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let sponge_state = self.sponge_state.read(builder);
        let input_buffer = self.input_buffer.read(builder);
        let output_buffer = self.output_buffer.read(builder);
        DuplexChallengerVariable { sponge_state, input_buffer, output_buffer }
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        self.sponge_state.write(witness);
        self.input_buffer.write(witness);
        self.output_buffer.write(witness);
    }
}

impl<C, F, W, const DIGEST_ELEMENTS: usize> Witnessable<C> for Hash<F, W, DIGEST_ELEMENTS>
where
    C: CircuitConfig,
    W: Witnessable<C>,
{
    type WitnessVariable = [W::WitnessVariable; DIGEST_ELEMENTS];

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let array: &[W; DIGEST_ELEMENTS] = self.borrow();
        array.read(builder)
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        let array: &[W; DIGEST_ELEMENTS] = self.borrow();
        array.write(witness);
    }
}

impl<C: CircuitConfig<F = InnerVal, EF = InnerChallenge>, SC: BabyBearFriConfigVariable<C>>
    Witnessable<C> for StarkVerifyingKey<SC>
where
    Com<SC>: Witnessable<C, WitnessVariable = <SC as FieldHasherVariable<C>>::DigestVariable>,
    OpeningProof<SC>: Witnessable<C, WitnessVariable = TwoAdicPcsProofVariable<C, SC>>,
{
    type WitnessVariable = VerifyingKeyVariable<C, SC>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        dt_stark::global_d11::validate_global146_identity(&self.global146_identity)
            .expect("direct VK witness has the current Global146 identity");
        let commitment = self.commit.read(builder);
        let pc_start = self.pc_start.read(builder);
        self.owner_registry.validate().expect("validated VK has a canonical owner registry");
        let (presence, xy) = match self.program_boundary {
            dt_stark::global_d11::ProgramImageBoundaryV1::Infinity => {
                (InnerVal::zero(), [[InnerVal::zero(); 11]; 2])
            }
            dt_stark::global_d11::ProgramImageBoundaryV1::Affine { x, y } => (
                InnerVal::one(),
                [x.map(InnerVal::from_canonical_u32), y.map(InnerVal::from_canonical_u32)],
            ),
        };
        let program_boundary_presence = presence.read(builder);
        let program_boundary_xy = xy.read(builder);
        let owner_entries = self
            .owner_registry
            .owners
            .iter()
            .map(|entry| {
                [
                    InnerVal::from_canonical_u32(entry.owner.0),
                    InnerVal::from_canonical_u8(entry.kind as u8),
                ]
            })
            .collect::<Vec<_>>();
        let owner_registry_entries = owner_entries.read(builder);
        let owner_registry_digest =
            self.owner_registry.digest.map(InnerVal::from_canonical_u8).read(builder);
        let seed = dt_stark::global_d11::program_global_seed::<InnerVal>(&self.program_boundary)
            .expect("validated VK has a canonical ProgramGlobalSeed");
        let program_global_seed =
            [*seed.x.coefficients(), *seed.y.coefficients(), *seed.z.coefficients()].read(builder);
        let chip_information = self.chip_information.clone();
        let chip_ordering = self.chip_ordering.clone();
        VerifyingKeyVariable {
            commitment,
            pc_start,
            program_boundary_presence,
            program_boundary_xy,
            owner_registry_entries,
            owner_registry_digest,
            program_global_seed,
            chip_information,
            chip_ordering,
        }
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        dt_stark::global_d11::validate_global146_identity(&self.global146_identity)
            .expect("direct VK witness has the current Global146 identity");
        self.commit.write(witness);
        self.pc_start.write(witness);
        self.owner_registry.validate().expect("validated VK has a canonical owner registry");
        let (presence, xy) = match self.program_boundary {
            dt_stark::global_d11::ProgramImageBoundaryV1::Infinity => {
                (InnerVal::zero(), [[InnerVal::zero(); 11]; 2])
            }
            dt_stark::global_d11::ProgramImageBoundaryV1::Affine { x, y } => (
                InnerVal::one(),
                [x.map(InnerVal::from_canonical_u32), y.map(InnerVal::from_canonical_u32)],
            ),
        };
        presence.write(witness);
        xy.write(witness);
        self.owner_registry
            .owners
            .iter()
            .map(|entry| {
                [
                    InnerVal::from_canonical_u32(entry.owner.0),
                    InnerVal::from_canonical_u8(entry.kind as u8),
                ]
            })
            .collect::<Vec<_>>()
            .write(witness);
        self.owner_registry.digest.map(InnerVal::from_canonical_u8).write(witness);
        let seed = dt_stark::global_d11::program_global_seed::<InnerVal>(&self.program_boundary)
            .expect("validated VK has a canonical ProgramGlobalSeed");
        [*seed.x.coefficients(), *seed.y.coefficients(), *seed.z.coefficients()].write(witness);
    }
}

impl<C> Witnessable<C> for DTRecursionWitnessValues<BabyBearPoseidon2>
where
    C: CircuitConfig<F = InnerVal, EF = InnerChallenge, Bit = Felt<InnerVal>>,
{
    type WitnessVariable = DTRecursionWitnessVariable<C, BabyBearPoseidon2>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let vk = self.vk.read(builder);
        let shard_proofs = self.shard_proofs.read(builder);
        let reconstruct_deferred_digest = self.reconstruct_deferred_digest.read(builder);
        let is_complete = InnerVal::from_bool(self.is_complete).read(builder);
        let is_first_shard = InnerVal::from_bool(self.is_first_shard).read(builder);
        let vk_root = self.vk_root.read(builder);
        DTRecursionWitnessVariable {
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

impl<C: CircuitConfig<F = InnerVal, EF = InnerChallenge>, SC: BabyBearFriConfigVariable<C>>
    Witnessable<C> for DTCompressWitnessValues<SC>
where
    Com<SC>: Witnessable<C, WitnessVariable = <SC as FieldHasherVariable<C>>::DigestVariable>,
    OpeningProof<SC>: Witnessable<C, WitnessVariable = TwoAdicPcsProofVariable<C, SC>>,
{
    type WitnessVariable = DTCompressWitnessVariable<C, SC>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let vks_and_proofs = self.vks_and_proofs.read(builder);
        let is_complete = InnerVal::from_bool(self.is_complete).read(builder);

        DTCompressWitnessVariable { vks_and_proofs, is_complete }
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        self.vks_and_proofs.write(witness);
        InnerVal::from_bool(self.is_complete).write(witness);
    }
}

impl<C> Witnessable<C> for DTDeferredWitnessValues<BabyBearPoseidon2>
where
    C: CircuitConfig<F = InnerVal, EF = InnerChallenge, Bit = Felt<InnerVal>>,
{
    type WitnessVariable = DTDeferredWitnessVariable<C, BabyBearPoseidon2>;

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

        DTDeferredWitnessVariable {
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

impl<C: CircuitConfig, HV: FieldHasherVariable<C>> Witnessable<C> for MerkleProof<C::F, HV>
where
    HV::Digest: Witnessable<C, WitnessVariable = HV::DigestVariable>,
{
    type WitnessVariable = MerkleProofVariable<C, HV>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let mut bits = vec![];
        let mut index = self.index;
        for _ in 0..self.path.len() {
            bits.push(index % 2 == 1);
            index >>= 1;
        }
        let index_bits = bits.read(builder);
        let path = self.path.read(builder);

        MerkleProofVariable { index: index_bits, path }
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        let mut index = self.index;
        for _ in 0..self.path.len() {
            (index % 2 == 1).write(witness);
            index >>= 1;
        }
        self.path.write(witness);
    }
}

impl<C: CircuitConfig<F = BabyBear>, SC: BabyBearFriConfigVariable<C>> Witnessable<C>
    for DTMerkleProofWitnessValues<SC>
where
    // This trait bound is redundant, but Rust-Analyzer is not able to infer it.
    SC: FieldHasher<BabyBear>,
    <SC as FieldHasher<BabyBear>>::Digest: Witnessable<C, WitnessVariable = SC::DigestVariable>,
{
    type WitnessVariable = DTMerkleProofWitnessVariable<C, SC>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        DTMerkleProofWitnessVariable {
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
