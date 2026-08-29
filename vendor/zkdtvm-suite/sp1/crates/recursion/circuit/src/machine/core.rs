use std::{
    array,
    borrow::{Borrow, BorrowMut},
    marker::PhantomData,
    mem::MaybeUninit,
};

use itertools::Itertools;
use p3_baby_bear::BabyBear;
use p3_commit::Mmcs;
use p3_field::AbstractField;
use p3_matrix::dense::RowMajorMatrix;

use dt_core_machine::riscv::{RiscvAir, MAX_CPU_LOG_DEGREE, MAX_LOG_NUMBER_OF_SHARDS};
use serde::{de::DeserializeOwned, Deserialize, Serialize};

use dt_recursion_core::air::PV_DIGEST_NUM_WORDS;
use dt_stark::{
    air::{MachineAir, PublicValues, POSEIDON_NUM_WORDS},
    baby_bear_poseidon2::BabyBearPoseidon2,
    shape::OrderedShape,
    sumcheck::{config::SCStarkGenericConfig, keys::SCStarkVerifyingKey, proof::SCShardProof},
    Challenge, Dom, StarkMachine, Word,
};

use dt_stark::{ShardProof, StarkGenericConfig, StarkVerifyingKey};

use dt_recursion_compiler::{
    circuit::CircuitV2Builder,
    ir::{Builder, Config, Felt, SymbolicFelt},
};

use dt_recursion_core::{
    air::{RecursionPublicValues, RECURSIVE_PROOF_NUM_PV_ELTS},
    DIGEST_SIZE,
};

use crate::{
    challenger::{CanObserveVariable, DuplexChallengerVariable},
    global_claim::{assert_canonical_shard_interval, global_identity},
    machine::{assert_complete, recursion_public_values_digest},
    stark::{dummy_vk_and_shard_proof, ShardProofVariable, StarkVerifier},
    sumcheck::{
        types::{SCShardProofVariable, SCVerifyingKeyVariable},
        SCBabyBearFriConfigVariable,
    },
    BabyBearFriConfig, BabyBearFriConfigVariable, CircuitConfig, VerifyingKeyVariable,
};

pub struct DTRecursionWitnessVariable<
    C: CircuitConfig<F = BabyBear>,
    SC: BabyBearFriConfigVariable<C>,
> {
    pub vk: VerifyingKeyVariable<C, SC>,
    pub shard_proofs: Vec<ShardProofVariable<C, SC>>,
    pub reconstruct_deferred_digest: [Felt<C::F>; DIGEST_SIZE],
    pub is_complete: Felt<C::F>,
    pub is_first_shard: Felt<C::F>,
    pub vk_root: [Felt<C::F>; DIGEST_SIZE],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "ShardProof<SC>: Serialize, Dom<SC>: Serialize"))]
#[serde(bound(deserialize = "ShardProof<SC>: Deserialize<'de>, Dom<SC>: DeserializeOwned"))]
pub struct DTRecursionWitnessValues<SC: StarkGenericConfig> {
    pub vk: StarkVerifyingKey<SC>,
    pub shard_proofs: Vec<ShardProof<SC>>,
    pub is_complete: bool,
    pub is_first_shard: bool,
    pub vk_root: [SC::Val; DIGEST_SIZE],
    pub reconstruct_deferred_digest: [SC::Val; 8],
}

pub struct SCDTRecursionWitnessVariable<
    C: CircuitConfig<F = SC::Val>,
    SC: SCBabyBearFriConfigVariable<C>,
> {
    pub vk: SCVerifyingKeyVariable<C, SC>,
    pub shard_proofs: Vec<SCShardProofVariable<C, SC>>,
    pub reconstruct_deferred_digest: [Felt<C::F>; DIGEST_SIZE],
    pub is_complete: Felt<C::F>,
    pub is_first_shard: Felt<C::F>,
    pub vk_root: [Felt<C::F>; DIGEST_SIZE],
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "ShardProof<SC>: Serialize, Dom<SC>: Serialize"))]
#[serde(bound(deserialize = "ShardProof<SC>: Deserialize<'de>, Dom<SC>: DeserializeOwned"))]
pub struct SCDTRecursionWitnessValues<SC: SCStarkGenericConfig> {
    pub vk: SCStarkVerifyingKey<SC>,
    pub shard_proofs: Vec<SCShardProof<SC>>,
    pub is_complete: bool,
    pub is_first_shard: bool,
    pub vk_root: [SC::Val; DIGEST_SIZE],
    pub reconstruct_deferred_digest: [SC::Val; 8],
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DTRecursionShape {
    pub proof_shapes: Vec<OrderedShape>,
    pub is_complete: bool,
}

/// A program for recursively verifying a batch of zkDTVM proofs.
#[derive(Debug, Clone, Copy)]
pub struct DTRecursiveVerifier<C: Config, SC: BabyBearFriConfig> {
    _phantom: PhantomData<(C, SC)>,
}

impl<C, SC> DTRecursiveVerifier<C, SC>
where
    SC: BabyBearFriConfigVariable<
        C,
        FriChallengerVariable = DuplexChallengerVariable<C>,
        DigestVariable = [Felt<BabyBear>; DIGEST_SIZE],
    >,
    C: CircuitConfig<F = SC::Val, EF = Challenge<SC>, Bit = Felt<BabyBear>>,
    <SC::ValMmcs as Mmcs<BabyBear>>::ProverData<RowMajorMatrix<BabyBear>>: Clone,
{
    /// Verify a batch of zkDTVM shard proofs and aggregate their public values.
    ///
    /// This program represents a first recursive step in the verification of a zkDTVM proof
    /// consisting of one or more shards. Each shard proof is verified and its public values are
    /// aggregated into a single set representing the start and end state of the program execution
    /// across all shards.
    ///
    /// # Constraints
    ///
    /// ## Verifying the STARK proofs.
    /// For each shard, the verifier asserts the correctness of the STARK proof which is composed
    /// of verifying the FRI proof for openings and verifying the constraints.
    ///
    /// ## Aggregating the shard public values.
    /// See [DTProver::verify] for the verification algorithm of a complete zkDTVM proof. In this
    /// function, we are aggregating several shard proofs and attesting to an aggregated state which
    /// represents all the shards.
    ///
    /// ## The leaf challenger.
    /// A key difference between the recursive tree verification and the complete one in
    /// [DTProver::verify] is that the recursive verifier has no way of reconstructing the
    /// chanllenger only from a part of the shard proof. Therefore, the value of the leaf challenger
    /// is witnessed in the program and the verifier asserts correctness given this challenger.
    /// In the course of the recursive verification, the challenger is reconstructed by observing
    /// the commitments one by one, and in the final step, the challenger is asserted to be the same
    /// as the one witnessed here.
    pub fn verify(
        builder: &mut Builder<C>,
        machine: &StarkMachine<SC, RiscvAir<SC::Val>>,
        input: DTRecursionWitnessVariable<C, SC>,
    ) {
        // Read input.
        let DTRecursionWitnessVariable {
            vk,
            shard_proofs,
            is_complete,
            is_first_shard,
            vk_root,
            reconstruct_deferred_digest,
        } = input;

        // Initialize shard variables.
        let mut initial_shard: Felt<_> = unsafe { MaybeUninit::zeroed().assume_init() };
        let mut current_shard: Felt<_> = unsafe { MaybeUninit::zeroed().assume_init() };

        // Initialize execution shard variables.
        let mut initial_execution_shard: Felt<_> = unsafe { MaybeUninit::zeroed().assume_init() };
        let mut current_execution_shard: Felt<_> = unsafe { MaybeUninit::zeroed().assume_init() };

        // Initialize program counter variables.
        let mut start_pc: Felt<_> = unsafe { MaybeUninit::zeroed().assume_init() };
        let mut current_pc: Felt<_> = unsafe { MaybeUninit::zeroed().assume_init() };

        // Initialize memory initialization and finalization variables.
        let mut initial_previous_init_addr: Felt<_> =
            unsafe { MaybeUninit::zeroed().assume_init() };
        let mut initial_previous_finalize_addr: Felt<_> =
            unsafe { MaybeUninit::zeroed().assume_init() };
        let mut current_init_addr: Felt<_> = unsafe { MaybeUninit::zeroed().assume_init() };
        let mut current_finalize_addr: Felt<_> = unsafe { MaybeUninit::zeroed().assume_init() };

        // Initialize the exit code variable.
        let mut exit_code: Felt<_> = unsafe { MaybeUninit::zeroed().assume_init() };

        // Initialize the public values digest.
        let mut committed_value_digest: [Word<Felt<_>>; PV_DIGEST_NUM_WORDS] =
            array::from_fn(|_| Word(array::from_fn(|_| builder.uninit())));

        // Initialize the deferred proofs digest.
        let mut deferred_proofs_digest: [Felt<_>; POSEIDON_NUM_WORDS] =
            array::from_fn(|_| builder.uninit());

        let mut global_has_opening: Felt<_> = builder.eval(C::F::zero());
        let mut global_count: Felt<_> = builder.eval(C::F::zero());
        let mut global_interval_start = global_identity(builder);
        let mut global_interval_end = global_interval_start;
        let mut global_interval_seen = false;

        // Assert that the number of proofs is not zero.
        assert!(!shard_proofs.is_empty());

        // Initialize a flag to denote the first (if any) CPU shard.
        let mut cpu_shard_seen = false;

        // Verify proofs.
        for (i, shard_proof) in shard_proofs.into_iter().enumerate() {
            let contains_cpu = shard_proof.contains_cpu();
            let contains_memory_init = shard_proof.contains_memory_init();
            let contains_memory_finalize = shard_proof.contains_memory_finalize();

            // Get the public values.
            let public_values: &PublicValues<Word<Felt<_>>, Felt<_>> =
                shard_proof.public_values.as_slice().borrow();

            // If this is the first proof in the batch, initialize the variables.
            if i == 0 {
                // Shard.
                initial_shard = public_values.shard;
                current_shard = public_values.shard;

                // Execution shard.
                initial_execution_shard = public_values.execution_shard;
                current_execution_shard = public_values.execution_shard;

                // Program counter.
                start_pc = public_values.start_pc;
                current_pc = public_values.start_pc;

                // Memory initialization & finalization.
                current_init_addr = public_values.previous_init_addr;
                initial_previous_init_addr = public_values.previous_init_addr;
                current_finalize_addr = public_values.previous_finalize_addr;
                initial_previous_finalize_addr = public_values.previous_finalize_addr;

                // Exit code.
                exit_code = public_values.exit_code;

                // Committed public values digests.
                for (word, first_word) in committed_value_digest
                    .iter_mut()
                    .zip_eq(public_values.committed_value_digest.iter())
                {
                    for (byte, first_byte) in word.0.iter_mut().zip_eq(first_word.0.iter()) {
                        *byte = *first_byte;
                    }
                }

                // Deferred proofs digests.
                for (digest, first_digest) in deferred_proofs_digest
                    .iter_mut()
                    .zip_eq(public_values.deferred_proofs_digest.iter())
                {
                    *digest = *first_digest;
                }

                // First shard constraints. We verify the validity of the `is_first_shard` boolean
                // flag, and make assertions for that are specific to the first shard using that
                // flag.

                // We assert that `is_first_shard == (initial_shard == 1)` in three steps.

                // First, we assert that the `is_first_shard` flag is boolean.
                builder
                    .assert_felt_eq(is_first_shard * (is_first_shard - C::F::one()), C::F::zero());
                // Assert that if `is_first_shard == 1`, then `initial_shard == 1`.
                builder
                    .assert_felt_eq(is_first_shard * (initial_shard - C::F::one()), C::F::zero());
                // Assert that if `is_first_shard == 0`, then `initial_shard != 1`.
                // This asserts that if `initial_shard == 1`, then `is_first_shard == 1`.
                builder.assert_felt_ne(
                    (SymbolicFelt::one() - is_first_shard) * initial_shard,
                    C::F::one(),
                );

                // If it's the first shard (which is the first execution shard), then the `start_pc`
                // should be vk.pc_start.
                builder.assert_felt_eq(is_first_shard * (start_pc - vk.pc_start), C::F::zero());
                // Assert that `init_addr` and `finalize_addr` are zero for the first shard.
                builder.assert_felt_eq(is_first_shard * current_init_addr, C::F::zero());
                builder.assert_felt_eq(is_first_shard * current_finalize_addr, C::F::zero());
            }

            let shard_start = [
                public_values.global.interval.start.x,
                public_values.global.interval.start.y,
                public_values.global.interval.start.z,
            ];
            let shard_end = [
                public_values.global.interval.end.x,
                public_values.global.interval.end.y,
                public_values.global.interval.end.z,
            ];
            assert_canonical_shard_interval(
                builder,
                public_values.global.has_global_opening,
                public_values.global.count,
                shard_start,
                shard_end,
            );
            if global_interval_seen {
                for coordinate in 0..3 {
                    for limb in 0..11 {
                        builder.assert_felt_eq(
                            global_interval_end[coordinate][limb],
                            shard_start[coordinate][limb],
                        );
                    }
                }
            } else {
                global_interval_start = shard_start;
                global_interval_seen = true;
            }
            global_interval_end = shard_end;
            global_count = builder.eval(global_count + public_values.global.count);
            global_has_opening = builder.eval(
                global_has_opening + public_values.global.has_global_opening -
                    global_has_opening * public_values.global.has_global_opening,
            );

            // Verify the shard.
            //
            // Do not verify the cumulative sum here, since the permutation challenge is shared
            // between all shards.

            // Prepare a challenger.
            let mut challenger = machine.config().challenger_variable(builder);

            vk.observe_into(builder, &mut challenger);

            // Observe the public values.
            challenger.observe_slice(builder, shard_proof.public_values.iter().copied());
            tracing::debug_span!("verify shard").in_scope(|| {
                StarkVerifier::verify_shard(builder, &vk, machine, &mut challenger, &shard_proof)
            });

            // Assert that first shard has a "CPU". Equivalently, assert that if the shard does
            // not have a "CPU", then the current shard is not 1.
            if !contains_cpu {
                builder.assert_felt_ne(current_shard, C::F::one());
            }

            // CPU log degree bound check constraints (this assertion is made in compile time).
            if shard_proof.contains_cpu() {
                let log_degree_cpu = shard_proof.log_degree_cpu();
                assert!(log_degree_cpu <= MAX_CPU_LOG_DEGREE);
            }

            // Shard constraints.
            {
                // Assert that the shard of the proof is equal to the current shard.
                builder.assert_felt_eq(current_shard, public_values.shard);

                // Increment the current shard by one.
                current_shard = builder.eval(current_shard + C::F::one());
            }

            // Execution shard constraints.
            {
                // If the shard has a "CPU" chip, then the execution shard should be incremented by
                // 1.
                if contains_cpu {
                    // If this is the first time we've seen the CPU, we initialize the initial and
                    // current execution shards.
                    if !cpu_shard_seen {
                        initial_execution_shard = public_values.execution_shard;
                        current_execution_shard = initial_execution_shard;
                        cpu_shard_seen = true;
                    }

                    builder.assert_felt_eq(current_execution_shard, public_values.execution_shard);

                    current_execution_shard = builder.eval(current_execution_shard + C::F::one());
                }
            }

            // Program counter constraints.
            {
                // Assert that the start_pc of the proof is equal to the current pc.
                builder.assert_felt_eq(current_pc, public_values.start_pc);

                // If it's not a shard with "CPU", then assert that the start_pc equals the
                // next_pc.
                if !contains_cpu {
                    builder.assert_felt_eq(public_values.start_pc, public_values.next_pc);
                } else {
                    // If it's a shard with "CPU", then assert that the start_pc is not zero.
                    builder.assert_felt_ne(public_values.start_pc, C::F::zero());
                }

                // Update current_pc to be the end_pc of the current proof.
                current_pc = public_values.next_pc;
            }

            // Exit code constraints.
            {
                // Assert that the exit code is zero (success) for all proofs.
                builder.assert_felt_eq(exit_code, C::F::zero());
            }

            // Memory initialization & finalization constraints.
            {
                // Assert that the MemoryInitialize address matches the current loop variable.
                builder.assert_felt_eq(current_init_addr, public_values.previous_init_addr);

                // Assert that the MemoryFinalize address matches the current loop variable.
                builder.assert_felt_eq(current_finalize_addr, public_values.previous_finalize_addr);

                // Assert that if MemoryInit is not present, then the addresses are the same.
                if !contains_memory_init {
                    builder.assert_felt_eq(
                        public_values.previous_init_addr,
                        public_values.last_init_addr,
                    );
                }

                // Assert that if MemoryFinalize is not present, then the addresses are the same.
                if !contains_memory_finalize {
                    builder.assert_felt_eq(
                        public_values.previous_finalize_addr,
                        public_values.last_finalize_addr,
                    );
                }

                // Update the MemoryInitialize address.
                current_init_addr = public_values.last_init_addr;

                // Update the MemoryFinalize address.
                current_finalize_addr = public_values.last_finalize_addr;
            }

            // Digest constraints.
            {
                // // If `committed_value_digest` is not zero, then the current value should be
                // equal to `public_values.committed_value_digest`.

                // Set flags to indicate whether `committed_value_digest` is non-zero. The flags are
                // given by the elements of the array, and they will be used as filters to constrain
                // the equality.
                let mut is_non_zero_flags = vec![];
                for word in committed_value_digest {
                    for byte in word {
                        is_non_zero_flags.push(byte);
                    }
                }

                // Using the flags, we can constrain the equality.
                for is_non_zero in is_non_zero_flags {
                    for (word_current, word_public) in
                        committed_value_digest.into_iter().zip(public_values.committed_value_digest)
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

                // If it's not a shard with "CPU", then the committed value digest shouldn't change.
                if !contains_cpu {
                    for (word_d, pub_word_d) in committed_value_digest
                        .iter()
                        .zip(public_values.committed_value_digest.iter())
                    {
                        for (d, pub_d) in word_d.0.iter().zip(pub_word_d.0.iter()) {
                            builder.assert_felt_eq(*d, *pub_d);
                        }
                    }
                }

                // Update the committed value digest.
                for (word_d, pub_word_d) in committed_value_digest
                    .iter_mut()
                    .zip(public_values.committed_value_digest.iter())
                {
                    for (d, pub_d) in word_d.0.iter_mut().zip(pub_word_d.0.iter()) {
                        *d = *pub_d;
                    }
                }

                // Update the exit code.
                exit_code = public_values.exit_code;

                // If `deferred_proofs_digest` is not zero, then the current value should be equal
                // to `public_values.deferred_proofs_digest.

                // Set a flag to indicate whether `deferred_proofs_digest` is non-zero. The flags
                // are given by the elements of the array, and they will be used as filters to
                // constrain the equality.
                let mut is_non_zero_flags = vec![];
                for element in deferred_proofs_digest {
                    is_non_zero_flags.push(element);
                }

                // Using the flags, we can constrain the equality.
                for is_non_zero in is_non_zero_flags {
                    for (deferred_current, deferred_public) in deferred_proofs_digest
                        .iter()
                        .zip(public_values.deferred_proofs_digest.iter())
                    {
                        builder.assert_felt_eq(
                            is_non_zero * (*deferred_current - *deferred_public),
                            C::F::zero(),
                        );
                    }
                }

                // If it's not a shard with "CPU", then the deferred proofs digest should not
                // change.
                if !contains_cpu {
                    for (d, pub_d) in deferred_proofs_digest
                        .iter()
                        .zip(public_values.deferred_proofs_digest.iter())
                    {
                        builder.assert_felt_eq(*d, *pub_d);
                    }
                }

                // Update the deferred proofs digest.
                deferred_proofs_digest.copy_from_slice(&public_values.deferred_proofs_digest);
            }

            // Verify that the number of shards is not too large, i.e. that for every shard, we
            // have shard < 2^{MAX_LOG_NUMBER_OF_SHARDS}.
            C::range_check_felt(builder, public_values.shard, MAX_LOG_NUMBER_OF_SHARDS);

        }

        // Assert that the last exit code is zero.
        builder.assert_felt_eq(exit_code, C::F::zero());

        // Write all values to the public values struct and commit to them.
        {
            // Compute the vk digest.
            let vk_digest = vk.hash(builder);

            // Initialize the public values we will commit to.
            let zero: Felt<_> = builder.eval(C::F::zero());
            let mut recursion_public_values_stream = [zero; RECURSIVE_PROOF_NUM_PV_ELTS];
            let recursion_public_values: &mut RecursionPublicValues<_> =
                recursion_public_values_stream.as_mut_slice().borrow_mut();
            recursion_public_values.committed_value_digest = committed_value_digest;
            recursion_public_values.deferred_proofs_digest = deferred_proofs_digest;
            recursion_public_values.start_pc = start_pc;
            recursion_public_values.next_pc = current_pc;
            recursion_public_values.start_shard = initial_shard;
            recursion_public_values.next_shard = current_shard;
            recursion_public_values.start_execution_shard = initial_execution_shard;
            recursion_public_values.next_execution_shard = current_execution_shard;
            recursion_public_values.previous_init_addr = initial_previous_init_addr;
            recursion_public_values.last_init_addr = current_init_addr;
            recursion_public_values.previous_finalize_addr = initial_previous_finalize_addr;
            recursion_public_values.last_finalize_addr = current_finalize_addr;
            recursion_public_values.dt_vk_digest = vk_digest;
            recursion_public_values.global_has_opening = global_has_opening;
            recursion_public_values.global_count = global_count;
            recursion_public_values.global_interval_start = global_interval_start;
            recursion_public_values.global_interval_end = global_interval_end;
            let identity = global_identity(builder);
            for coordinate in 0..3 {
                for limb in 0..11 {
                    builder.assert_felt_eq(
                        is_complete *
                            (global_interval_start[coordinate][limb] -
                                vk.program_global_seed[coordinate][limb]),
                        C::F::zero(),
                    );
                    builder.assert_felt_eq(
                        is_complete *
                            (global_interval_end[coordinate][limb] - identity[coordinate][limb]),
                        C::F::zero(),
                    );
                }
            }
            recursion_public_values.start_reconstruct_deferred_digest = reconstruct_deferred_digest;
            recursion_public_values.end_reconstruct_deferred_digest = reconstruct_deferred_digest;
            recursion_public_values.exit_code = exit_code;
            recursion_public_values.is_complete = is_complete;
            // Set the contains an execution shard flag.
            recursion_public_values.contains_execution_shard =
                builder.eval(C::F::from_bool(cpu_shard_seen));
            recursion_public_values.vk_root = vk_root;

            // Calculate the digest and set it in the public values.
            recursion_public_values.digest =
                recursion_public_values_digest::<C, SC>(builder, recursion_public_values);

            assert_complete(builder, recursion_public_values, is_complete);

            SC::commit_recursion_public_values(builder, *recursion_public_values);
        }
    }
}

impl<SC: BabyBearFriConfig> DTRecursionWitnessValues<SC> {
    pub fn shape(&self) -> DTRecursionShape {
        let proof_shapes = self.shard_proofs.iter().map(|proof| proof.shape()).collect();

        DTRecursionShape { proof_shapes, is_complete: self.is_complete }
    }
}

impl DTRecursionWitnessValues<BabyBearPoseidon2> {
    pub fn dummy(
        machine: &StarkMachine<BabyBearPoseidon2, RiscvAir<BabyBear>>,
        shape: &DTRecursionShape,
    ) -> Self {
        let (mut vks, shard_proofs): (Vec<_>, Vec<_>) =
            shape.proof_shapes.iter().map(|shape| dummy_vk_and_shard_proof(machine, shape)).unzip();
        let vk = vks.pop().unwrap();
        Self {
            vk,
            shard_proofs,
            reconstruct_deferred_digest: [BabyBear::zero(); DIGEST_SIZE],
            is_complete: false,
            is_first_shard: false,
            vk_root: [BabyBear::zero(); DIGEST_SIZE],
        }
    }
}

impl From<OrderedShape> for DTRecursionShape {
    fn from(proof_shape: OrderedShape) -> Self {
        Self { proof_shapes: vec![proof_shape], is_complete: false }
    }
}
