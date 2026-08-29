use std::{
    array,
    borrow::{Borrow, BorrowMut},
};

use dt_primitives::consts::WORD_SIZE;
use dt_recursion_compiler::{
    circuit::CircuitV2Builder,
    ir::{Builder, Felt},
};
use dt_stark::{
    air::{FullAir, MachineAir, PolyAirExtendable, POSEIDON_NUM_WORDS},
    baby_bear_poseidon2::SCBabyBearPoseidon2,
    sumcheck::{keys::SCStarkVerifyingKey, proof::SCShardProof},
    Challenge, SCStarkMachine, Val, Word,
};
use p3_air::Air;
use p3_baby_bear::BabyBear;
use p3_commit::Mmcs;
use p3_field::{extension::BinomialExtensionField, AbstractField};
use p3_matrix::dense::RowMajorMatrix;
use serde::{Deserialize, Serialize};

use dt_recursion_core::{
    air::{RecursionPublicValues, PV_DIGEST_NUM_WORDS, RECURSIVE_PROOF_NUM_PV_ELTS},
    DIGEST_SIZE,
};

use super::public_values::{
    sc_assert_recursion_public_values_valid, sc_recursion_public_values_digest,
};
use crate::{
    challenger::{CanObserveVariable, DuplexChallengerVariable},
    hash::{FieldHasher, FieldHasherVariable},
    machine::{assert_complete, DTDeferredShape},
    sc_machine::{
        SCDTCompressWitnessValues, SCDTMerkleProofVerifier, SCDTMerkleProofWitnessValues,
        SCDTMerkleProofWitnessVariable,
    },
    sumcheck::{
        polyair_folder::RecursivePolyAirConstraintFolder,
        polyair_precompute::RecursivePolyAirPrecomputeRowBuilder,
        polyair_verifier::PolyAirSumcheckVerifier,
        types::{SCShardProofVariable, SCVerifyingKeyVariable},
        SCBabyBearFriConfig, SCBabyBearFriConfigVariable,
    },
    CircuitConfig,
};
use p3_uni_stark::SymbolicAirBuilder;
use polyair::SCStarkMachine as PolyAirStarkMachine;

pub struct SCDTDeferredVerifier<C, SC, A, const D: usize> {
    _phantom: std::marker::PhantomData<(C, SC, A)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "SC::Digest: Serialize"))]
#[serde(bound(deserialize = "SC::Digest: Deserialize<'de>"))]
pub struct SCDTDeferredWitnessValues<SC: SCBabyBearFriConfig + FieldHasher<Val<SC>>> {
    pub vks_and_proofs: Vec<(SCStarkVerifyingKey<SC>, SCShardProof<SC>)>,
    pub vk_merkle_data: SCDTMerkleProofWitnessValues<SC>,
    pub start_reconstruct_deferred_digest: [SC::Val; POSEIDON_NUM_WORDS],
    pub dt_vk_digest: [SC::Val; DIGEST_SIZE],
    pub committed_value_digest: [Word<SC::Val>; PV_DIGEST_NUM_WORDS],
    pub deferred_proofs_digest: [SC::Val; POSEIDON_NUM_WORDS],
    pub end_pc: SC::Val,
    pub end_shard: SC::Val,
    pub end_execution_shard: SC::Val,
    pub init_addr: SC::Val,
    pub finalize_addr: SC::Val,
    pub is_complete: bool,
}

pub struct SCDTDeferredWitnessVariable<
    C: CircuitConfig<F = SC::Val>,
    SC: FieldHasherVariable<C> + SCBabyBearFriConfigVariable<C>,
> {
    pub vks_and_proofs: Vec<(SCVerifyingKeyVariable<C, SC>, SCShardProofVariable<C, SC>)>,
    pub vk_merkle_data: SCDTMerkleProofWitnessVariable<C, SC>,
    pub start_reconstruct_deferred_digest: [Felt<C::F>; POSEIDON_NUM_WORDS],
    pub dt_vk_digest: [Felt<C::F>; DIGEST_SIZE],
    pub committed_value_digest: [Word<Felt<C::F>>; PV_DIGEST_NUM_WORDS],
    pub deferred_proofs_digest: [Felt<C::F>; POSEIDON_NUM_WORDS],
    pub end_pc: Felt<C::F>,
    pub end_shard: Felt<C::F>,
    pub end_execution_shard: Felt<C::F>,
    pub init_addr: Felt<C::F>,
    pub finalize_addr: Felt<C::F>,
    pub is_complete: Felt<C::F>,
}

impl<C, SC, A, const D: usize> SCDTDeferredVerifier<C, SC, A, D>
where
    SC: SCBabyBearFriConfigVariable<C, FriChallengerVariable = DuplexChallengerVariable<C>>
        + FieldHasherVariable<C, DigestVariable = [Felt<C::F>; DIGEST_SIZE]>,
    C: CircuitConfig<F = SC::Val, EF = Challenge<SC>>,
    SC::ValMmcs: Mmcs<SC::Val, ProverData<RowMajorMatrix<SC::Val>>: Clone>,
    A: MachineAir<SC::Val>,
    Val<SC>: PolyAirExtendable<D>,
    Builder<C>: CircuitV2Builder<C>,
{
    /// Verify a batch of deferred proofs.
    ///
    /// Each deferred proof is a recursive proof representing some computation. Namely, every such
    /// proof represents a recursively verified program.
    /// verifier:
    /// - Asserts that each of these proofs is valid as a `compress` proof.
    /// - Asserts that each of these proofs is complete by checking the `is_complete` flag in the
    ///   proof's public values.
    /// - Aggregates the proof information into the accumulated deferred digest.
    pub fn verify(
        builder: &mut Builder<C>,
        machine: &PolyAirStarkMachine<SC, A, D>,
        input: SCDTDeferredWitnessVariable<C, SC>,
        value_assertions: bool,
    ) where
        A: for<'a> FullAir<RecursivePolyAirConstraintFolder<'a, C>>,
        A: for<'a> FullAir<RecursivePolyAirPrecomputeRowBuilder<'a, C>>,
    {
        let SCDTDeferredWitnessVariable {
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
        } = input;

        // First, verify the merkle tree proofs.
        let vk_root = vk_merkle_data.root;
        let values = vks_and_proofs.iter().map(|(vk, _)| vk.hash(builder)).collect::<Vec<_>>();
        SCDTMerkleProofVerifier::verify(builder, values, vk_merkle_data, value_assertions);

        let mut deferred_public_values_stream: Vec<Felt<C::F>> =
            (0..RECURSIVE_PROOF_NUM_PV_ELTS).map(|_| builder.uninit()).collect();
        let deferred_public_values: &mut RecursionPublicValues<_> =
            deferred_public_values_stream.as_mut_slice().borrow_mut();

        // Initialize the start of deferred digests.
        deferred_public_values.start_reconstruct_deferred_digest =
            start_reconstruct_deferred_digest;

        // Initialize the consistency check variable.
        let mut reconstruct_deferred_digest: [Felt<C::F>; POSEIDON_NUM_WORDS] =
            start_reconstruct_deferred_digest;

        for (vk, shard_proof) in vks_and_proofs {
            // Initialize a challenger.
            let mut challenger = machine.config().challenger_variable(builder);
            vk.observe_into(builder, &mut challenger);

            // Observe the and public values.
            challenger.observe_slice(
                builder,
                shard_proof.public_values[0..machine.num_pv_elts()].iter().copied(),
            );

            let chips = machine.shard_chips_ordered(&shard_proof.chip_ordering).collect::<Vec<_>>();
            PolyAirSumcheckVerifier::<C, SC, A, D>::verify_shard(
                builder,
                &vk,
                &chips,
                &mut challenger,
                &shard_proof,
            );

            // Get the current public values.
            let current_public_values: &RecursionPublicValues<Felt<C::F>> =
                shard_proof.public_values.as_slice().borrow();
            // Assert that the `vk_root` is the same as the witnessed one.
            for (elem, expected) in current_public_values.vk_root.iter().zip(vk_root.iter()) {
                builder.assert_felt_eq(*elem, *expected);
            }
            // Assert that the public values are valid.
            sc_assert_recursion_public_values_valid::<C, SC>(builder, current_public_values);

            // Assert that the proof is complete.
            builder.assert_felt_eq(current_public_values.is_complete, C::F::one());

            // Update deferred proof digest
            // poseidon2( current_digest[..8] || pv.dt_vk_digest[..8] ||
            // pv.committed_value_digest[..32] )
            let mut inputs: [Felt<C::F>; 48] = array::from_fn(|_| builder.uninit());
            inputs[0..DIGEST_SIZE].copy_from_slice(&reconstruct_deferred_digest);

            inputs[DIGEST_SIZE..DIGEST_SIZE + DIGEST_SIZE]
                .copy_from_slice(&current_public_values.dt_vk_digest);

            for j in 0..PV_DIGEST_NUM_WORDS {
                for k in 0..WORD_SIZE {
                    let element = current_public_values.committed_value_digest[j][k];
                    inputs[j * WORD_SIZE + k + 16] = element;
                }
            }
            reconstruct_deferred_digest = SC::hash(builder, &inputs);
        }

        // Set the public values.

        // Set initial_pc, end_pc, initial_shard, and end_shard to be the hitned values.
        deferred_public_values.start_pc = end_pc;
        deferred_public_values.next_pc = end_pc;
        deferred_public_values.start_shard = end_shard;
        deferred_public_values.next_shard = end_shard;
        deferred_public_values.start_execution_shard = end_execution_shard;
        deferred_public_values.next_execution_shard = end_execution_shard;
        // Set the init and finalize addresses to be the hinted values.
        deferred_public_values.previous_init_addr = init_addr;
        deferred_public_values.last_init_addr = init_addr;
        deferred_public_values.previous_finalize_addr = finalize_addr;
        deferred_public_values.last_finalize_addr = finalize_addr;

        // Set the dt_vk_digest to be the hitned value.
        deferred_public_values.dt_vk_digest = dt_vk_digest;

        // Set the committed value digest to be the hitned value.
        deferred_public_values.committed_value_digest = committed_value_digest;
        // Set the deferred proof digest to be the hitned value.
        deferred_public_values.deferred_proofs_digest = deferred_proofs_digest;

        // Set the exit code to be zero for now.
        deferred_public_values.exit_code = builder.eval(C::F::zero());
        // Assign the deferred proof digests.
        deferred_public_values.end_reconstruct_deferred_digest = reconstruct_deferred_digest;
        // Set the is_complete flag.
        deferred_public_values.is_complete = is_complete;
        // Set the `contains_execution_shard` flag.
        deferred_public_values.contains_execution_shard = builder.eval(C::F::zero());
        let zero = builder.eval(C::F::zero());
        deferred_public_values.global_has_opening = zero;
        deferred_public_values.global_count = zero;
        let mut global_identity = [[zero; 11]; 3];
        global_identity[1][0] = builder.eval(C::F::one());
        deferred_public_values.global_interval_start = global_identity;
        deferred_public_values.global_interval_end = global_identity;
        // Set the vk root from the witness.
        deferred_public_values.vk_root = vk_root;
        // Set the digest according to the previous values.
        deferred_public_values.digest =
            sc_recursion_public_values_digest::<C, SC>(builder, deferred_public_values);

        assert_complete(builder, deferred_public_values, is_complete);
        builder.assert_felt_eq(is_complete, C::F::zero());

        SC::commit_recursion_public_values(builder, *deferred_public_values);
    }
}

impl SCDTDeferredWitnessValues<SCBabyBearPoseidon2> {
    pub fn dummy<
        A: MachineAir<BabyBear> + for<'a> Air<SymbolicAirBuilder<BabyBear>>,
        AE: MachineAir<BinomialExtensionField<BabyBear, 4>>,
    >(
        machine: &SCStarkMachine<SCBabyBearPoseidon2, A, AE>,
        shape: &DTDeferredShape,
    ) -> Self {
        let inner_witness =
            SCDTCompressWitnessValues::<SCBabyBearPoseidon2>::dummy(machine, &shape.inner);
        let vks_and_proofs = inner_witness.vks_and_proofs;

        let vk_merkle_data = SCDTMerkleProofWitnessValues::<SCBabyBearPoseidon2>::dummy(
            vks_and_proofs.len(),
            shape.height,
        );

        Self {
            vks_and_proofs,
            vk_merkle_data,
            is_complete: true,
            dt_vk_digest: [BabyBear::zero(); DIGEST_SIZE],
            start_reconstruct_deferred_digest: [BabyBear::zero(); POSEIDON_NUM_WORDS],
            committed_value_digest: [Word::default(); PV_DIGEST_NUM_WORDS],
            deferred_proofs_digest: [BabyBear::zero(); POSEIDON_NUM_WORDS],
            end_pc: BabyBear::zero(),
            end_shard: BabyBear::zero(),
            end_execution_shard: BabyBear::zero(),
            init_addr: BabyBear::zero(),
            finalize_addr: BabyBear::zero(),
        }
    }

    pub fn dummy_polyair<A: MachineAir<BabyBear>, const D: usize>(
        machine: &PolyAirStarkMachine<SCBabyBearPoseidon2, A, D>,
        shape: &DTDeferredShape,
    ) -> Self
    where
        BabyBear: PolyAirExtendable<D>,
    {
        let inner_witness =
            SCDTCompressWitnessValues::<SCBabyBearPoseidon2>::dummy_polyair(machine, &shape.inner);
        let vks_and_proofs = inner_witness.vks_and_proofs;

        let vk_merkle_data = SCDTMerkleProofWitnessValues::<SCBabyBearPoseidon2>::dummy(
            vks_and_proofs.len(),
            shape.height,
        );

        Self {
            vks_and_proofs,
            vk_merkle_data,
            is_complete: true,
            dt_vk_digest: [BabyBear::zero(); DIGEST_SIZE],
            start_reconstruct_deferred_digest: [BabyBear::zero(); POSEIDON_NUM_WORDS],
            committed_value_digest: [Word::default(); PV_DIGEST_NUM_WORDS],
            deferred_proofs_digest: [BabyBear::zero(); POSEIDON_NUM_WORDS],
            end_pc: BabyBear::zero(),
            end_shard: BabyBear::zero(),
            end_execution_shard: BabyBear::zero(),
            init_addr: BabyBear::zero(),
            finalize_addr: BabyBear::zero(),
        }
    }
}

#[cfg(feature = "koalabear")]
impl
    SCDTDeferredWitnessValues<
        dt_stark::koalabear_poseidon2::koala_bear_poseidon2::SCKoalaBearPoseidon2,
    >
{
    pub fn dummy<
        A: MachineAir<p3_koala_bear::KoalaBear>
            + for<'a> Air<SymbolicAirBuilder<p3_koala_bear::KoalaBear>>,
        AE: MachineAir<
            Challenge<dt_stark::koalabear_poseidon2::koala_bear_poseidon2::SCKoalaBearPoseidon2>,
        >,
    >(
        machine: &SCStarkMachine<
            dt_stark::koalabear_poseidon2::koala_bear_poseidon2::SCKoalaBearPoseidon2,
            A,
            AE,
        >,
        shape: &DTDeferredShape,
    ) -> Self {
        use p3_koala_bear::KoalaBear;
        type KBConfig = dt_stark::koalabear_poseidon2::koala_bear_poseidon2::SCKoalaBearPoseidon2;

        let inner_witness = SCDTCompressWitnessValues::<KBConfig>::dummy(machine, &shape.inner);
        let vks_and_proofs = inner_witness.vks_and_proofs;

        let vk_merkle_data =
            SCDTMerkleProofWitnessValues::<KBConfig>::dummy(vks_and_proofs.len(), shape.height);

        Self {
            vks_and_proofs,
            vk_merkle_data,
            is_complete: true,
            dt_vk_digest: [KoalaBear::zero(); DIGEST_SIZE],
            start_reconstruct_deferred_digest: [KoalaBear::zero(); POSEIDON_NUM_WORDS],
            committed_value_digest: [Word::default(); PV_DIGEST_NUM_WORDS],
            deferred_proofs_digest: [KoalaBear::zero(); POSEIDON_NUM_WORDS],
            end_pc: KoalaBear::zero(),
            end_shard: KoalaBear::zero(),
            end_execution_shard: KoalaBear::zero(),
            init_addr: KoalaBear::zero(),
            finalize_addr: KoalaBear::zero(),
        }
    }

    pub fn dummy_polyair<A: MachineAir<p3_koala_bear::KoalaBear>, const D: usize>(
        machine: &PolyAirStarkMachine<
            dt_stark::koalabear_poseidon2::koala_bear_poseidon2::SCKoalaBearPoseidon2,
            A,
            D,
        >,
        shape: &DTDeferredShape,
    ) -> Self
    where
        p3_koala_bear::KoalaBear: PolyAirExtendable<D>,
    {
        use p3_koala_bear::KoalaBear;
        type KBConfig = dt_stark::koalabear_poseidon2::koala_bear_poseidon2::SCKoalaBearPoseidon2;

        let inner_witness =
            SCDTCompressWitnessValues::<KBConfig>::dummy_polyair(machine, &shape.inner);
        let vks_and_proofs = inner_witness.vks_and_proofs;

        let vk_merkle_data =
            SCDTMerkleProofWitnessValues::<KBConfig>::dummy(vks_and_proofs.len(), shape.height);

        Self {
            vks_and_proofs,
            vk_merkle_data,
            is_complete: true,
            dt_vk_digest: [KoalaBear::zero(); DIGEST_SIZE],
            start_reconstruct_deferred_digest: [KoalaBear::zero(); POSEIDON_NUM_WORDS],
            committed_value_digest: [Word::default(); PV_DIGEST_NUM_WORDS],
            deferred_proofs_digest: [KoalaBear::zero(); POSEIDON_NUM_WORDS],
            end_pc: KoalaBear::zero(),
            end_shard: KoalaBear::zero(),
            end_execution_shard: KoalaBear::zero(),
            init_addr: KoalaBear::zero(),
            finalize_addr: KoalaBear::zero(),
        }
    }
}
