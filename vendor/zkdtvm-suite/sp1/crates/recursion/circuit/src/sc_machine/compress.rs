use itertools::{izip, Itertools};
use p3_uni_stark::SymbolicAirBuilder;
use std::{
    array,
    borrow::{Borrow, BorrowMut},
    marker::PhantomData,
    mem::MaybeUninit,
};

use dt_recursion_compiler::ir::{Builder, Felt, SymbolicFelt};
use p3_air::Air;
use p3_baby_bear::BabyBear;
use p3_commit::Mmcs;
use p3_field::{extension::BinomialExtensionField, AbstractField};
use p3_matrix::dense::RowMajorMatrix;
use serde::{de::DeserializeOwned, Deserialize, Serialize};

use dt_recursion_core::air::{RecursionPublicValues, RECURSIVE_PROOF_NUM_PV_ELTS};

use super::public_values::{
    sc_assert_recursion_public_values_valid, sc_recursion_public_values_digest,
    sc_root_public_values_digest,
};
use crate::{
    challenger::CanObserveVariable,
    global_claim::{assert_canonical_shard_interval, global_identity},
    machine::assert_complete,
    // stark::{dummy_vk_and_shard_proof, ShardProofVariable, StarkVerifier},
    sumcheck::types::{SCShardProofVariable, SCVerifyingKeyVariable},
    sumcheck::{
        polyair_folder::RecursivePolyAirConstraintFolder,
        polyair_precompute::RecursivePolyAirPrecomputeRowBuilder,
        polyair_sc_dummy_vk_and_shard_proof, polyair_verifier::PolyAirSumcheckVerifier,
        sc_dummy_vk_and_shard_proof, SCBabyBearFriConfigVariable,
    },
    CircuitConfig,
};
use crate::{
    machine::{DTCompressShape, PublicValuesOutputDigest},
    sumcheck::SCBabyBearFriConfig,
};
use dt_recursion_compiler::circuit::CircuitV2Builder;
use dt_stark::{
    air::{FullAir, MachineAir, PolyAirExtendable, POSEIDON_NUM_WORDS, PV_DIGEST_NUM_WORDS},
    baby_bear_poseidon2::SCBabyBearPoseidon2,
    sumcheck::{config::SCStarkGenericConfig, keys::SCStarkVerifyingKey, proof::SCShardProof},
    Challenge, Dom, SCStarkMachine, Val, Word, DIGEST_SIZE,
};
use polyair::SCStarkMachine as PolyAirStarkMachine;
/// A program to verify a batch of recursive proofs and aggregate their public values.
#[derive(Debug, Clone, Copy)]
pub struct SCDTCompressVerifier<C, SC, A, const D: usize> {
    _phantom: PhantomData<(C, SC, A)>,
}

// pub enum PublicValuesOutputDigest {
//     Reduce,
//     Root,
// }

/// Witness layout for the compress stage verifier.
pub struct SCDTCompressWitnessVariable<
    C: CircuitConfig<F = SC::Val>,
    SC: SCBabyBearFriConfigVariable<C>,
> {
    /// The shard proofs to verify.
    pub vks_and_proofs: Vec<(SCVerifyingKeyVariable<C, SC>, SCShardProofVariable<C, SC>)>,
    pub is_complete: Felt<C::F>,
}

/// An input layout for the reduce verifier.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "SCShardProof<SC>: Serialize, Dom<SC>: Serialize"))]
#[serde(bound(deserialize = "SCShardProof<SC>: Deserialize<'de>, Dom<SC>: DeserializeOwned"))]
pub struct SCDTCompressWitnessValues<SC: SCStarkGenericConfig> {
    pub vks_and_proofs: Vec<(SCStarkVerifyingKey<SC>, SCShardProof<SC>)>,
    pub is_complete: bool,
}

// #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
// pub struct DTCompressShape {
//     proof_shapes: Vec<OrderedShape>,
// }

impl<C, SC, A, const D: usize> SCDTCompressVerifier<C, SC, A, D>
where
    SC: SCBabyBearFriConfigVariable<C>,
    C: CircuitConfig<F = SC::Val, EF = Challenge<SC>>,
    SC::ValMmcs: Mmcs<SC::Val, ProverData<RowMajorMatrix<SC::Val>>: Clone>,
    A: MachineAir<SC::Val>,
    Val<SC>: PolyAirExtendable<D>,
    Builder<C>: CircuitV2Builder<C>,
{
    /// Verify a batch of recursive proofs and aggregate their public values.
    ///
    /// The compression verifier can aggregate proofs of different kinds:
    /// - Core proofs: proofs which are recursive proof of a batch of zkDTVM shard proofs. The
    ///   implementation in this function assumes a fixed recursive verifier specified by
    ///   `recursive_vk`.
    /// - Deferred proofs: proofs which are recursive proof of a batch of deferred proofs. The
    ///   implementation in this function assumes a fixed deferred verification program specified by
    ///   `deferred_vk`.
    /// - Compress proofs: these are proofs which refer to a prove of this program. The key for it
    ///   is part of public values will be propagated across all levels of recursion and will be
    ///   checked against itself as in the prover or as in [super::DTRootVerifier].
    pub fn verify(
        builder: &mut Builder<C>,
        machine: &PolyAirStarkMachine<SC, A, D>,
        input: SCDTCompressWitnessVariable<C, SC>,
        vk_root: [Felt<C::F>; DIGEST_SIZE],
        kind: PublicValuesOutputDigest,
    ) where
        A: for<'a> FullAir<RecursivePolyAirConstraintFolder<'a, C>>,
        A: for<'a> FullAir<RecursivePolyAirPrecomputeRowBuilder<'a, C>>,
    {
        // Read input.
        let SCDTCompressWitnessVariable { vks_and_proofs, is_complete } = input;

        // Initialize the values for the aggregated public output.

        let mut reduce_public_values_stream: Vec<Felt<_>> = (0..RECURSIVE_PROOF_NUM_PV_ELTS)
            .map(|_| unsafe { MaybeUninit::zeroed().assume_init() })
            .collect();
        let compress_public_values: &mut RecursionPublicValues<_> =
            reduce_public_values_stream.as_mut_slice().borrow_mut();

        // Make sure there is at least one proof.
        assert!(!vks_and_proofs.is_empty());

        // Initialize the consistency check variables.
        let mut dt_vk_digest: [Felt<_>; DIGEST_SIZE] =
            array::from_fn(|_| unsafe { MaybeUninit::zeroed().assume_init() });
        let mut pc: Felt<_> = unsafe { MaybeUninit::zeroed().assume_init() };
        let mut shard: Felt<_> = unsafe { MaybeUninit::zeroed().assume_init() };

        let mut exit_code: Felt<_> = builder.uninit();

        let mut execution_shard: Felt<_> = unsafe { MaybeUninit::zeroed().assume_init() };
        let mut committed_value_digest: [Word<Felt<_>>; PV_DIGEST_NUM_WORDS] =
            array::from_fn(|_| {
                Word(array::from_fn(|_| unsafe { MaybeUninit::zeroed().assume_init() }))
            });
        let mut deferred_proofs_digest: [Felt<_>; POSEIDON_NUM_WORDS] =
            array::from_fn(|_| unsafe { MaybeUninit::zeroed().assume_init() });
        let mut reconstruct_deferred_digest: [Felt<_>; POSEIDON_NUM_WORDS] =
            core::array::from_fn(|_| unsafe { MaybeUninit::zeroed().assume_init() });
        let mut global_has_opening: Felt<_> = builder.eval(C::F::zero());
        let mut global_count: Felt<_> = builder.eval(C::F::zero());
        let mut global_interval_start = global_identity(builder);
        let mut global_interval_end = global_interval_start;
        let mut global_interval_seen = false;
        let mut program_global_seed = global_interval_start;
        let mut init_addr: Felt<_> = unsafe { MaybeUninit::zeroed().assume_init() };
        let mut finalize_addr: Felt<_> = unsafe { MaybeUninit::zeroed().assume_init() };

        // Initialize a flag to denote if the any of the recursive proofs represents a shard range
        // where at least once of the shards is an execution shard (i.e. contains cpu).
        let mut contains_execution_shard: Felt<_> = builder.eval(C::F::zero());

        // Verify proofs, check consistency, and aggregate public values.
        for (i, (vk, shard_proof)) in vks_and_proofs.into_iter().enumerate() {
            // Verify the shard proof.

            // Prepare a challenger.
            let mut challenger = machine.config().challenger_variable(builder);

            vk.observe_into(builder, &mut challenger);

            // Observe the public values.
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
            // Assert that the public values are valid.
            sc_assert_recursion_public_values_valid::<C, SC>(builder, current_public_values);
            // Assert that the vk root is the same as the witnessed one.
            for (expected, actual) in vk_root.iter().zip(current_public_values.vk_root.iter()) {
                builder.assert_felt_eq(*expected, *actual);
            }

            // Set the exit code, it is already constrained to be zero in the previous proof.
            exit_code = current_public_values.exit_code;

            if i == 0 {
                // Initialize global and accumulated values.

                // Initialize the start of deferred digests.
                for (digest, current_digest, global_digest) in izip!(
                    reconstruct_deferred_digest.iter_mut(),
                    current_public_values.start_reconstruct_deferred_digest.iter(),
                    compress_public_values.start_reconstruct_deferred_digest.iter_mut()
                ) {
                    *digest = *current_digest;
                    *global_digest = *current_digest;
                }

                // Initialize the dt_vk digest
                for (digest, first_digest) in
                    dt_vk_digest.iter_mut().zip(current_public_values.dt_vk_digest)
                {
                    *digest = first_digest;
                }

                // Initiallize start pc.
                compress_public_values.start_pc = current_public_values.start_pc;
                pc = current_public_values.start_pc;

                // Initialize start shard.
                compress_public_values.start_shard = current_public_values.start_shard;
                shard = current_public_values.start_shard;

                // Initialize start execution shard.
                compress_public_values.start_execution_shard =
                    current_public_values.start_execution_shard;
                execution_shard = current_public_values.start_execution_shard;

                // Initialize the MemoryInitialize address.
                init_addr = current_public_values.previous_init_addr;
                compress_public_values.previous_init_addr =
                    current_public_values.previous_init_addr;

                // Initialize the MemoryFinalize address.
                finalize_addr = current_public_values.previous_finalize_addr;
                compress_public_values.previous_finalize_addr =
                    current_public_values.previous_finalize_addr;

                // Assign the committed values and deferred proof digests.
                for (word, current_word) in committed_value_digest
                    .iter_mut()
                    .zip_eq(current_public_values.committed_value_digest.iter())
                {
                    for (byte, current_byte) in word.0.iter_mut().zip_eq(current_word.0.iter()) {
                        *byte = *current_byte;
                    }
                }

                for (digest, current_digest) in deferred_proofs_digest
                    .iter_mut()
                    .zip_eq(current_public_values.deferred_proofs_digest.iter())
                {
                    *digest = *current_digest;
                }

                program_global_seed = vk.program_global_seed;
            }

            for coordinate in 0..3 {
                for limb in 0..11 {
                    builder.assert_felt_eq(
                        program_global_seed[coordinate][limb],
                        vk.program_global_seed[coordinate][limb],
                    );
                }
            }
            assert_canonical_shard_interval(
                builder,
                current_public_values.global_has_opening,
                current_public_values.global_count,
                current_public_values.global_interval_start,
                current_public_values.global_interval_end,
            );
            if global_interval_seen {
                for coordinate in 0..3 {
                    for limb in 0..11 {
                        builder.assert_felt_eq(
                            global_interval_end[coordinate][limb],
                            current_public_values.global_interval_start[coordinate][limb],
                        );
                    }
                }
            } else {
                global_interval_start = current_public_values.global_interval_start;
                global_interval_seen = true;
            }
            global_interval_end = current_public_values.global_interval_end;
            global_count = builder.eval(global_count + current_public_values.global_count);
            global_has_opening = builder.eval(
                global_has_opening + current_public_values.global_has_opening -
                    global_has_opening * current_public_values.global_has_opening,
            );

            // Assert that the current values match the accumulated values.

            // Assert that the start deferred digest is equal to the current deferred digest.
            for (digest, current_digest) in reconstruct_deferred_digest
                .iter()
                .zip_eq(current_public_values.start_reconstruct_deferred_digest.iter())
            {
                builder.assert_felt_eq(*digest, *current_digest);
            }

            // // Consistency checks for all accumulated values.

            // Assert that the dt_vk digest is always the same.
            for (digest, current) in dt_vk_digest.iter().zip(current_public_values.dt_vk_digest) {
                builder.assert_felt_eq(*digest, current);
            }

            // Assert that the start pc is equal to the current pc.
            builder.assert_felt_eq(pc, current_public_values.start_pc);

            // Verify that the shard is equal to the current shard.
            builder.assert_felt_eq(shard, current_public_values.start_shard);

            // Execution shard constraints.
            {
                // Assert that `contains_execution_shard` is boolean.
                builder.assert_felt_eq(
                    current_public_values.contains_execution_shard *
                        (SymbolicFelt::one() - current_public_values.contains_execution_shard),
                    C::F::zero(),
                );
                // A flag to indicate whether the first execution shard has been seen. We have:
                // - `is_first_execution_shard_seen`  = current_contains_execution_shard &&
                //   !execution_shard_seen_before.
                // Since `contains_execution_shard` is the boolean flag used to denote if we have
                // seen an execution shard, we can use it to denote if we have seen an execution
                // shard before.
                let is_first_execution_shard_seen: Felt<_> = builder.eval(
                    current_public_values.contains_execution_shard *
                        (SymbolicFelt::one() - contains_execution_shard),
                );

                // If this is the first execution shard, then we update the start execution shard
                // and the `execution_shard` values.
                compress_public_values.start_execution_shard = builder.eval(
                    current_public_values.start_execution_shard * is_first_execution_shard_seen +
                        compress_public_values.start_execution_shard *
                            (SymbolicFelt::one() - is_first_execution_shard_seen),
                );
                execution_shard = builder.eval(
                    current_public_values.start_execution_shard * is_first_execution_shard_seen +
                        execution_shard * (SymbolicFelt::one() - is_first_execution_shard_seen),
                );

                // If this is an execution shard, make the assertion that the value is consistent.
                builder.assert_felt_eq(
                    current_public_values.contains_execution_shard *
                        (execution_shard - current_public_values.start_execution_shard),
                    C::F::zero(),
                );
            }

            // Assert that the MemoryInitialize address is the same.
            builder.assert_felt_eq(init_addr, current_public_values.previous_init_addr);

            // Assert that the MemoryFinalize address is the same.
            builder.assert_felt_eq(finalize_addr, current_public_values.previous_finalize_addr);

            // Digest constraints.
            {
                // If `committed_value_digest` is not zero, then
                // `public_values.committed_value_digest should be the current.

                // Set a flags to indicate whether `committed_value_digest` is non-zero. The flags
                // are given by the elements of the array, and they will be used as filters to
                // constrain the equality.
                let mut is_non_zero_flags = vec![];
                for word in committed_value_digest {
                    for byte in word {
                        is_non_zero_flags.push(byte);
                    }
                }

                // Using the flags, we can constrain the equality.
                for is_non_zero in is_non_zero_flags {
                    for (word_current, word_public) in committed_value_digest
                        .into_iter()
                        .zip(current_public_values.committed_value_digest)
                    {
                        for (byte_current, byte_public) in word_current.into_iter().zip(word_public)
                        {
                            builder.assert_felt_eq(
                                is_non_zero * (byte_current - byte_public),
                                C::F::zero(),
                            );
                        }
                    }
                }

                // Update the committed value digest.
                for (word, current_word) in committed_value_digest
                    .iter_mut()
                    .zip_eq(current_public_values.committed_value_digest.iter())
                {
                    for (byte, current_byte) in word.0.iter_mut().zip_eq(current_word.0.iter()) {
                        *byte = *current_byte;
                    }
                }

                //  If `deferred_proofs_digest` is not zero, then the current value should be
                // `public_values.deferred_proofs_digest`. We will use a similar approach as above.
                let mut is_non_zero_flags = vec![];
                for element in deferred_proofs_digest {
                    is_non_zero_flags.push(element);
                }

                for is_non_zero in is_non_zero_flags {
                    for (digest_current, digest_public) in deferred_proofs_digest
                        .into_iter()
                        .zip(current_public_values.deferred_proofs_digest)
                    {
                        builder.assert_felt_eq(
                            is_non_zero * (digest_current - digest_public),
                            C::F::zero(),
                        );
                    }
                }

                // Update the deferred proofs digest.
                for (digest, current_digest) in deferred_proofs_digest
                    .iter_mut()
                    .zip_eq(current_public_values.deferred_proofs_digest.iter())
                {
                    *digest = *current_digest;
                }
            }

            // Update the accumulated values.

            // If the current shard has an execution shard, then we update the flag in case it was
            // not already set. That is:
            // - If the current shard has an execution shard and the flag is set to zero, it will be
            //   set to one.
            // - If the current shard has an execution shard and the flag is set to one, it will
            //   remain set to one.
            contains_execution_shard = builder.eval(
                contains_execution_shard +
                    current_public_values.contains_execution_shard *
                        (SymbolicFelt::one() - contains_execution_shard),
            );

            // If this proof contains an execution shard, we update the execution shard value.
            execution_shard = builder.eval(
                current_public_values.next_execution_shard *
                    current_public_values.contains_execution_shard +
                    execution_shard *
                        (SymbolicFelt::one() - current_public_values.contains_execution_shard),
            );

            // Update the reconstruct deferred proof digest.
            for (digest, current_digest) in reconstruct_deferred_digest
                .iter_mut()
                .zip_eq(current_public_values.end_reconstruct_deferred_digest.iter())
            {
                *digest = *current_digest;
            }

            // Update pc to be the next pc.
            pc = current_public_values.next_pc;

            // Update the shard to be the next shard.
            shard = current_public_values.next_shard;

            // Update the MemoryInitialize address.
            init_addr = current_public_values.last_init_addr;

            // Update the MemoryFinalize address.
            finalize_addr = current_public_values.last_finalize_addr;

        }

        // Update the global values from the last accumulated values.
        // Set dt_vk digest to the one from the proof values.
        compress_public_values.dt_vk_digest = dt_vk_digest;
        // Set next_pc to be the last pc (which is the same as accumulated pc)
        compress_public_values.next_pc = pc;
        // Set next shard to be the last shard
        compress_public_values.next_shard = shard;
        // Set next execution shard to be the last execution shard
        compress_public_values.next_execution_shard = execution_shard;
        // Set the MemoryInitialize address to be the last MemoryInitialize address.
        compress_public_values.last_init_addr = init_addr;
        // Set the MemoryFinalize address to be the last MemoryFinalize address.
        compress_public_values.last_finalize_addr = finalize_addr;
        // Set the start reconstruct deferred digest to be the last reconstruct deferred digest.
        compress_public_values.end_reconstruct_deferred_digest = reconstruct_deferred_digest;
        // Assign the deferred proof digests.
        compress_public_values.deferred_proofs_digest = deferred_proofs_digest;
        // Assign the committed value digests.
        compress_public_values.committed_value_digest = committed_value_digest;
        compress_public_values.global_has_opening = global_has_opening;
        compress_public_values.global_count = global_count;
        compress_public_values.global_interval_start = global_interval_start;
        compress_public_values.global_interval_end = global_interval_end;
        let identity = global_identity(builder);
        for coordinate in 0..3 {
            for limb in 0..11 {
                builder.assert_felt_eq(
                    is_complete *
                        (global_interval_start[coordinate][limb] -
                            program_global_seed[coordinate][limb]),
                    C::F::zero(),
                );
                builder.assert_felt_eq(
                    is_complete *
                        (global_interval_end[coordinate][limb] - identity[coordinate][limb]),
                    C::F::zero(),
                );
            }
        }
        // Assign the `is_complete` flag.
        compress_public_values.is_complete = is_complete;
        // Set the contains an execution shard flag.
        compress_public_values.contains_execution_shard = contains_execution_shard;
        // Set the exit code.
        compress_public_values.exit_code = exit_code;
        // Reflect the vk root.
        compress_public_values.vk_root = vk_root;
        // Set the digest according to the previous values.
        compress_public_values.digest = match kind {
            PublicValuesOutputDigest::Reduce => {
                sc_recursion_public_values_digest::<C, SC>(builder, compress_public_values)
            }
            PublicValuesOutputDigest::Root => {
                sc_root_public_values_digest::<C, SC>(builder, compress_public_values)
            }
        };

        // If the proof is complete, make completeness assertions.
        assert_complete(builder, compress_public_values, is_complete);

        SC::commit_recursion_public_values(builder, *compress_public_values);
    }
}

impl<SC: SCBabyBearFriConfig> SCDTCompressWitnessValues<SC> {
    pub fn shape(&self) -> DTCompressShape {
        let proof_shapes = self.vks_and_proofs.iter().map(|(_, proof)| proof.shape()).collect();
        DTCompressShape { proof_shapes }
    }
}

impl SCDTCompressWitnessValues<SCBabyBearPoseidon2> {
    pub fn dummy<
        A: MachineAir<BabyBear> + for<'a> Air<SymbolicAirBuilder<BabyBear>>,
        AE: MachineAir<BinomialExtensionField<BabyBear, 4>>,
    >(
        machine: &SCStarkMachine<SCBabyBearPoseidon2, A, AE>,
        shape: &DTCompressShape,
    ) -> Self {
        let vks_and_proofs = shape
            .proof_shapes
            .iter()
            .map(|proof_shape| {
                let (vk, proof) = sc_dummy_vk_and_shard_proof(machine, proof_shape);
                (vk, proof)
            })
            .collect();

        Self { vks_and_proofs, is_complete: false }
    }

    pub fn dummy_polyair<A: MachineAir<BabyBear>, const D: usize>(
        machine: &PolyAirStarkMachine<SCBabyBearPoseidon2, A, D>,
        shape: &DTCompressShape,
    ) -> Self
    where
        BabyBear: PolyAirExtendable<D>,
    {
        let vks_and_proofs = shape
            .proof_shapes
            .iter()
            .map(|proof_shape| {
                let (vk, proof) = polyair_sc_dummy_vk_and_shard_proof(machine, proof_shape);
                (vk, proof)
            })
            .collect();

        Self { vks_and_proofs, is_complete: false }
    }
}

#[cfg(feature = "koalabear")]
impl
    SCDTCompressWitnessValues<
        dt_stark::koalabear_poseidon2::koala_bear_poseidon2::SCKoalaBearPoseidon2,
    >
{
    pub fn dummy<
        A: MachineAir<p3_koala_bear::KoalaBear>,
        AE: MachineAir<
            Challenge<dt_stark::koalabear_poseidon2::koala_bear_poseidon2::SCKoalaBearPoseidon2>,
        >,
    >(
        machine: &SCStarkMachine<
            dt_stark::koalabear_poseidon2::koala_bear_poseidon2::SCKoalaBearPoseidon2,
            A,
            AE,
        >,
        shape: &DTCompressShape,
    ) -> Self
    where
        A: for<'a> Air<SymbolicAirBuilder<p3_koala_bear::KoalaBear>>,
    {
        use crate::sumcheck::kb_sc_dummy_vk_and_shard_proof;
        let vks_and_proofs = shape
            .proof_shapes
            .iter()
            .map(|proof_shape| {
                let (vk, proof) = kb_sc_dummy_vk_and_shard_proof(machine, proof_shape);
                (vk, proof)
            })
            .collect();

        Self { vks_and_proofs, is_complete: false }
    }

    pub fn dummy_polyair<A: MachineAir<p3_koala_bear::KoalaBear>, const D: usize>(
        machine: &PolyAirStarkMachine<
            dt_stark::koalabear_poseidon2::koala_bear_poseidon2::SCKoalaBearPoseidon2,
            A,
            D,
        >,
        shape: &DTCompressShape,
    ) -> Self
    where
        p3_koala_bear::KoalaBear: PolyAirExtendable<D>,
    {
        use crate::sumcheck::kb_polyair_sc_dummy_vk_and_shard_proof;
        let vks_and_proofs = shape
            .proof_shapes
            .iter()
            .map(|proof_shape| {
                let (vk, proof) = kb_polyair_sc_dummy_vk_and_shard_proof(machine, proof_shape);
                (vk, proof)
            })
            .collect();

        Self { vks_and_proofs, is_complete: false }
    }
}
