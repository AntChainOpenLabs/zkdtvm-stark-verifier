use crate::{OuterSC, SCStarkVerifyingKey};
use anyhow::Result;
use dt_core_executor::{subproof::SubproofVerifier, DTReduceProof};
use dt_core_machine::{
    riscv::MAX_CPU_LOG_DEGREE,
    shape::{chip_log_height_threshold, num_skip_rounds},
};
use dt_primitives::{
    consts::WORD_SIZE,
    io::{blake3_hash, DTPublicValues},
    SCField,
};
use dt_recursion_circuit::machine::RootPublicValues;
use dt_recursion_core::air::RecursionPublicValues;
use dt_recursion_gnark_ffi::{
    Groth16Bn254Proof, Groth16Bn254Prover, PlonkBn254Proof, PlonkBn254Prover,
};
use num_bigint::BigUint;
use p3_field::{AbstractField, PrimeField};
use std::path::Path;
// use dt_stark::sumcheck::types::SCStarkVerifyingKey;
use std::{borrow::Borrow, str::FromStr};

use dt_stark::{
    air::{PublicValues, POSEIDON_NUM_WORDS, PV_DIGEST_NUM_WORDS},
    sumcheck::{proof::SCMachineProof, prover::SCMachineProver},
    MachineVerificationError, StarkGenericConfig, Word,
};
use polyair::prover::SCMachineProver as PolyAirMachineProver;
use thiserror::Error;

use crate::{
    components::DTProverComponents,
    utils::{is_recursion_public_values_valid, is_root_public_values_valid},
    CoreSC, DTCoreProofData, DTProver, DTVerifyingKey, HashableKey, InnerSC, RootSC,
};

#[derive(Error, Debug)]
pub enum PlonkVerificationError {
    #[error(
        "the verifying key does not match the inner plonk bn254 proof's committed verifying key"
    )]
    InvalidVerificationKey,
    #[error(
        "the public values in the dt proof do not match the public values in the inner plonk bn254 proof"
    )]
    InvalidPublicValues,
}

#[derive(Error, Debug)]
pub enum Groth16VerificationError {
    #[error(
        "the verifying key does not match the inner groth16 bn254 proof's committed verifying key"
    )]
    InvalidVerificationKey,
    #[error(
        "the public values in the dt proof do not match the public values in the inner groth16 bn254 proof"
    )]
    InvalidPublicValues,
}

impl<C: DTProverComponents> DTProver<C> {
    /// Verify a core proof by verifying the shards, verifying lookup bus, verifying that the
    /// shards are contiguous and complete.
    pub fn verify(
        &self,
        proof: &DTCoreProofData,
        vk: &DTVerifyingKey,
    ) -> Result<(), MachineVerificationError<CoreSC>> {
        // The proof should not be empty.
        if proof.0.is_empty() {
            return Err(MachineVerificationError::EmptyProof);
        }

        // Cpu has been split into multiple chips, so we don't need to check if the first shard has
        // a "CPU". // First shard has a "CPU" constraint.
        // //
        // // Check that the first shard has a "CPU".
        // // SAFETY: The proof is already checked to not be empty.
        // let first_shard = proof.0.first().unwrap();
        // if !first_shard.contains_cpu() {
        //     return Err(MachineVerificationError::MissingCpuInFirstShard);
        // }

        // CPU log degree bound constraints.
        //
        // Check that the CPU log degree does not exceed `MAX_CPU_LOG_DEGREE`. This is to ensure
        // that the lookup argument's multiplicities do not overflow.
        for shard_proof in proof.0.iter() {
            if shard_proof.contains_cpu() {
                let log_degree_cpu = shard_proof.log_degree_cpu();
                if log_degree_cpu > MAX_CPU_LOG_DEGREE {
                    return Err(MachineVerificationError::CpuLogDegreeTooLarge(log_degree_cpu));
                }
            }
        }

        // Shard constraints.
        //
        // Initialization:
        // - Shard should start at one.
        //
        // Transition:
        // - Shard should increment by one for each shard.
        let mut current_shard = SCField::zero();
        for shard_proof in proof.0.iter() {
            let public_values: &PublicValues<Word<_>, _> =
                shard_proof.public_values.as_slice().borrow();
            current_shard += SCField::one();
            if public_values.shard != current_shard {
                return Err(MachineVerificationError::InvalidPublicValues(
                    "shard index should be the previous shard index + 1 and start at 1",
                ));
            }
        }

        // Execution shard constraints.
        //
        // Initialization:
        // - Execution shard should start at one.
        //
        // Transition:
        // - Execution shard should increment by one for each shard with "CPU".
        // - Execution shard should stay the same for non-CPU shards.
        // - For the other shards, execution shard does not matter.
        let mut current_execution_shard = SCField::zero();
        for shard_proof in proof.0.iter() {
            let public_values: &PublicValues<Word<_>, _> =
                shard_proof.public_values.as_slice().borrow();
            if shard_proof.contains_cpu() {
                current_execution_shard += SCField::one();
                if public_values.execution_shard != current_execution_shard {
                    return Err(MachineVerificationError::InvalidPublicValues(
                        "execution shard index should be the previous execution shard index + 1 if cpu exists and start at 1",
                    ));
                }
            }
        }

        // Program counter constraints.
        //
        // Initialization:
        // - `start_pc` should start as `vk.start_pc`.
        //
        // Transition:
        // - `next_pc` of the previous shard should equal `start_pc`.
        // - If it's not a shard with "CPU", then `start_pc` equals `next_pc`.
        // - If it's a shard with "CPU", then `start_pc` should never equal zero.
        //
        // Finalization:
        // - `next_pc` should equal zero.
        let mut prev_next_pc = SCField::zero();
        for (i, shard_proof) in proof.0.iter().enumerate() {
            let public_values: &PublicValues<Word<_>, _> =
                shard_proof.public_values.as_slice().borrow();
            if i == 0 && public_values.start_pc != vk.vk.pc_start {
                return Err(MachineVerificationError::InvalidPublicValues(
                    "start_pc != vk.start_pc: program counter should start at vk.start_pc",
                ));
            } else if i != 0 && public_values.start_pc != prev_next_pc {
                return Err(MachineVerificationError::InvalidPublicValues(
                    "start_pc != next_pc_prev: start_pc should equal next_pc_prev for all shards",
                ));
            } else if !shard_proof.contains_cpu() && public_values.start_pc != public_values.next_pc
            {
                return Err(MachineVerificationError::InvalidPublicValues(
                    "start_pc != next_pc: start_pc should equal next_pc for non-cpu shards",
                ));
            } else if shard_proof.contains_cpu() && public_values.start_pc == SCField::zero() {
                return Err(MachineVerificationError::InvalidPublicValues(
                    "start_pc == 0: execution should never start at halted state",
                ));
            } else if i == proof.0.len() - 1 && public_values.next_pc != SCField::zero() {
                return Err(MachineVerificationError::InvalidPublicValues(
                    "next_pc != 0: execution should have halted",
                ));
            }
            prev_next_pc = public_values.next_pc;
        }

        // Exit code constraints.
        //
        // - In every shard, the exit code should be zero.
        for shard_proof in proof.0.iter() {
            let public_values: &PublicValues<Word<_>, _> =
                shard_proof.public_values.as_slice().borrow();
            if public_values.exit_code != SCField::zero() {
                return Err(MachineVerificationError::InvalidPublicValues(
                    "exit_code != 0: exit code should be zero for all shards",
                ));
            }
        }

        // Memory initialization & finalization constraints.
        //
        // Initialization:
        // - `previous_init_addr` should be zero.
        // - `previous_finalize_addr` should be zero.
        //
        // Transition:
        // - For all shards, `previous_init_addr` should equal `last_init_addr` of the previous
        //   shard.
        // - For all shards, `previous_finalize_addr` should equal `last_finalize_addr` of the
        //   previous shard.
        // - For shards without "MemoryInit", `previous_init_addr` should equal `last_init_addr`.
        // - For shards without "MemoryFinalize", `previous_finalize_addr` should equal
        //   `last_finalize_addr`.
        let mut last_init_addr_prev = SCField::zero();
        let mut last_finalize_addr_prev = SCField::zero();
        for shard_proof in proof.0.iter() {
            let public_values: &PublicValues<Word<_>, _> =
                shard_proof.public_values.as_slice().borrow();
            if public_values.previous_init_addr != last_init_addr_prev {
                return Err(MachineVerificationError::InvalidPublicValues(
                    "previous_init_addr != last_init_addr_prev",
                ));
            } else if public_values.previous_finalize_addr != last_finalize_addr_prev {
                return Err(MachineVerificationError::InvalidPublicValues(
                    "last_init_addr != last_finalize_addr_prev",
                ));
            } else if !shard_proof.contains_global_memory_init() &&
                public_values.previous_init_addr != public_values.last_init_addr
            {
                return Err(MachineVerificationError::InvalidPublicValues(
                    "previous_init_addr != last_init_addr",
                ));
            } else if !shard_proof.contains_global_memory_finalize() &&
                public_values.previous_finalize_addr != public_values.last_finalize_addr
            {
                return Err(MachineVerificationError::InvalidPublicValues(
                    "previous_finalize_addr != last_finalize_addr",
                ));
            }
            last_init_addr_prev = public_values.last_init_addr;
            last_finalize_addr_prev = public_values.last_finalize_addr;
        }

        // Digest constraints.
        //
        // Initialization:
        // - `committed_value_digest` should be zero.
        // - `deferred_proofs_digest` should be zero.
        //
        // Transition:
        // - If `committed_value_digest_prev` is not zero, then `committed_value_digest` should
        //   equal
        //  `committed_value_digest_prev`. Otherwise, `committed_value_digest` should equal zero.
        // - If `deferred_proofs_digest_prev` is not zero, then `deferred_proofs_digest` should
        //   equal
        //  `deferred_proofs_digest_prev`. Otherwise, `deferred_proofs_digest` should equal zero.
        // - If it's not a shard with "CPU", then `committed_value_digest` should not change from
        //   the
        //  previous shard.
        // - If it's not a shard with "CPU", then `deferred_proofs_digest` should not change from
        //   the
        //  previous shard.
        let zero_committed_value_digest = [Word([SCField::zero(); WORD_SIZE]); PV_DIGEST_NUM_WORDS];
        let zero_deferred_proofs_digest = [SCField::zero(); POSEIDON_NUM_WORDS];
        let mut committed_value_digest_prev = zero_committed_value_digest;
        let mut deferred_proofs_digest_prev = zero_deferred_proofs_digest;
        for shard_proof in proof.0.iter() {
            let public_values: &PublicValues<Word<_>, _> =
                shard_proof.public_values.as_slice().borrow();
            if committed_value_digest_prev != zero_committed_value_digest &&
                public_values.committed_value_digest != committed_value_digest_prev
            {
                return Err(MachineVerificationError::InvalidPublicValues(
                    "committed_value_digest != committed_value_digest_prev",
                ));
            } else if deferred_proofs_digest_prev != zero_deferred_proofs_digest &&
                public_values.deferred_proofs_digest != deferred_proofs_digest_prev
            {
                return Err(MachineVerificationError::InvalidPublicValues(
                    "deferred_proofs_digest != deferred_proofs_digest_prev",
                ));
            } else if !shard_proof.contains_cpu() &&
                public_values.committed_value_digest != committed_value_digest_prev
            {
                return Err(MachineVerificationError::InvalidPublicValues(
                    "committed_value_digest != committed_value_digest_prev",
                ));
            } else if !shard_proof.contains_cpu() &&
                public_values.deferred_proofs_digest != deferred_proofs_digest_prev
            {
                return Err(MachineVerificationError::InvalidPublicValues(
                    "deferred_proofs_digest != deferred_proofs_digest_prev",
                ));
            }
            committed_value_digest_prev = public_values.committed_value_digest;
            deferred_proofs_digest_prev = public_values.deferred_proofs_digest;
        }

        // Verify that the number of shards is not too large.
        if proof.0.len() >= 1 << 16 {
            return Err(MachineVerificationError::TooManyShards);
        }

        // Verify the shard proof.
        let mut challenger = self.core_prover.config().challenger();
        let machine_proof = SCMachineProof { shard_proofs: proof.0.to_vec() };
        self.core_prover.machine().verify(
            &vk.vk,
            &machine_proof,
            &mut challenger,
            dt_core_machine::utils::prove_polyair::POLYAIR_NUM_SKIP_ROUNDS,
            dt_core_machine::utils::prove_polyair::POLYAIR_CHIP_LOG_HEIGHT_THRESHOLD,
        )?;

        Ok(())
    }

    /// Verify a compressed proof, auto-detecting the native vs DSL shape.
    ///
    /// Note: the native arm runs first and the fallthrough boundary is exact —
    /// only a machine-shape mismatch (the presented vk differing from the native
    /// root vk) falls through to the DSL chain; once a proof is native-shaped,
    /// any failure returns immediately and no DSL fallback may mask it.
    pub fn verify_compressed(
        &self,
        proof: &DTReduceProof<RootSC>,
        vk: &DTVerifyingKey,
    ) -> Result<(), MachineVerificationError<CoreSC>> {
        #[cfg(feature = "native-recursion")]
        {
            match self.native_backend() {
                Ok(backend) => {
                    let is_native = backend.is_native_proof(proof).map_err(|err| {
                        MachineVerificationError::NativeRecursion(err.to_string())
                    })?;
                    if is_native {
                        return backend.verify_native(proof, vk).map_err(|err| {
                            MachineVerificationError::NativeRecursion(err.to_string())
                        });
                    }
                    // Not native-shaped: fall through to the DSL chain.
                }
                Err(err) => {
                    // Backend init failed (config authority is fail-closed): the
                    // native arm is unavailable. DSL proofs must keep verifying, so
                    // fall through; a native proof then fails the DSL machine chain
                    // (fail-closed in effect, with the init error logged here).
                    if native_recursion::debug_prints_enabled() {
                        println!("native verify arm unavailable: {err}");
                    }
                    tracing::warn!(
                        "native verify arm unavailable (backend init failed): {err}; \
                         falling through to the DSL verification chain"
                    );
                }
            }
        }

        let DTReduceProof { vk: compress_vk, proof } = proof;
        self.root_shrink_prover
            .machine()
            .verify(
                compress_vk,
                &SCMachineProof { shard_proofs: vec![proof.clone()] },
                &mut self.root_shrink_prover.config().challenger(),
                num_skip_rounds(),
                chip_log_height_threshold(),
            )
            .map_err(|_| {
                MachineVerificationError::InvalidPublicValues("root_shrink proof failed")
            })?;

        // Validate public values
        let public_values: &RecursionPublicValues<_> = proof.public_values.as_slice().borrow();

        if !is_recursion_public_values_valid(self.compress_prover.machine().config(), public_values)
        {
            return Err(MachineVerificationError::InvalidPublicValues(
                "recursion public values are invalid",
            ));
        }

        if public_values.vk_root != self.recursion_vk_root {
            return Err(MachineVerificationError::InvalidPublicValues("vk_root mismatch"));
        }

        // The root_shrink VK itself is not recursively verified, so it is not
        // part of the Poseidon2 recursion VK allowlist.

        // `is_complete` should be 1. In the reduce program, this ensures that the proof is fully
        // reduced.
        if public_values.is_complete != SCField::one() {
            return Err(MachineVerificationError::InvalidPublicValues("is_complete is not 1"));
        }

        // Verify that the proof is for the dt vkey we are expecting.
        let vkey_hash = vk.hash_babybear();
        if public_values.dt_vk_digest != vkey_hash {
            return Err(MachineVerificationError::InvalidPublicValues("dt vk hash mismatch"));
        }

        Ok(())
    }

    /// Verify a shrink proof.
    pub fn verify_shrink(
        &self,
        proof: &DTReduceProof<InnerSC>,
        vk: &DTVerifyingKey,
    ) -> Result<(), MachineVerificationError<CoreSC>> {
        let mut challenger = self.shrink_prover.config().challenger();
        let machine_proof = SCMachineProof { shard_proofs: vec![proof.proof.clone()] };
        self.shrink_prover.machine().verify(
            &proof.vk,
            &machine_proof,
            &mut challenger,
            num_skip_rounds(),
            chip_log_height_threshold(),
        )?;

        // Validate public values
        let public_values: &RecursionPublicValues<_> =
            proof.proof.public_values.as_slice().borrow();
        if !is_recursion_public_values_valid(self.compress_prover.machine().config(), public_values)
        {
            return Err(MachineVerificationError::InvalidPublicValues(
                "recursion public values are invalid",
            ));
        }
        if public_values.vk_root != self.recursion_vk_root {
            return Err(MachineVerificationError::InvalidPublicValues("vk_root mismatch"));
        }

        if self.vk_verification && !self.recursion_vk_map.contains_key(&proof.vk.hash_babybear()) {
            return Err(MachineVerificationError::InvalidVerificationKey);
        }

        // `is_complete` should be 1. In the reduce program, this ensures that the proof is fully
        // reduced.
        if public_values.is_complete != SCField::one() {
            return Err(MachineVerificationError::InvalidPublicValues("is_complete is not 1"));
        }

        // Verify that the proof is for the dt vkey we are expecting.
        let vkey_hash = vk.hash_babybear();
        if public_values.dt_vk_digest != vkey_hash {
            return Err(MachineVerificationError::InvalidPublicValues("dt vk hash mismatch"));
        }

        Ok(())
    }

    // /// Verify a wrap bn254 proof.
    pub fn verify_wrap_bn254(
        &self,
        proof: &DTReduceProof<OuterSC>,
        vk: &DTVerifyingKey,
    ) -> Result<(), MachineVerificationError<OuterSC>> {
        let mut challenger = self.wrap_prover.config().challenger();
        let machine_proof = SCMachineProof { shard_proofs: vec![proof.proof.clone()] };

        let wrap_vk = self.wrap_vk.get().ok_or(MachineVerificationError::InvalidPublicValues(
            "wrap verifier key not set (wrap_bn254 must be called before verify)",
        ))?;
        self.wrap_prover.machine().verify(
            wrap_vk,
            &machine_proof,
            &mut challenger,
            num_skip_rounds(),
            chip_log_height_threshold(),
        )?;

        // Validate public values
        let public_values: &RootPublicValues<_> = proof.proof.public_values.as_slice().borrow();
        if !is_root_public_values_valid(self.shrink_prover.machine().config(), public_values) {
            return Err(MachineVerificationError::InvalidPublicValues(
                "root public values are invalid",
            ));
        }

        // Verify that the proof is for the dt vkey we are expecting.
        let vkey_hash = vk.hash_babybear();
        if *public_values.dt_vk_digest() != vkey_hash {
            return Err(MachineVerificationError::InvalidPublicValues("dt vk hash mismatch"));
        }

        Ok(())
    }

    /// Verifies a PLONK proof using the circuit artifacts in the build directory.
    pub fn verify_plonk_bn254(
        &self,
        proof: &PlonkBn254Proof,
        vk: &DTVerifyingKey,
        public_values: &DTPublicValues,
        build_dir: &Path,
    ) -> Result<()> {
        let prover = PlonkBn254Prover::new();

        let vkey_hash = BigUint::from_str(&proof.public_inputs[0])?;
        let committed_values_digest = BigUint::from_str(&proof.public_inputs[1])?;

        // Verify the proof with the corresponding public inputs.
        prover.verify(proof, &vkey_hash, &committed_values_digest, build_dir)?;

        verify_plonk_bn254_public_inputs(vk, public_values, &proof.public_inputs)?;

        Ok(())
    }

    /// Verifies a Groth16 proof using the circuit artifacts in the build directory.
    pub fn verify_groth16_bn254(
        &self,
        proof: &Groth16Bn254Proof,
        vk: &DTVerifyingKey,
        public_values: &DTPublicValues,
        build_dir: &Path,
    ) -> Result<()> {
        let prover = Groth16Bn254Prover::new();

        let vkey_hash = BigUint::from_str(&proof.public_inputs[0])?;
        let committed_values_digest = BigUint::from_str(&proof.public_inputs[1])?;

        // Verify the proof with the corresponding public inputs.
        prover.verify(proof, &vkey_hash, &committed_values_digest, build_dir)?;

        verify_groth16_bn254_public_inputs(vk, public_values, &proof.public_inputs)?;

        Ok(())
    }
}

/// Verify the vk_hash and public_values_hash in the public inputs of the PlonkBn254Proof match the
/// expected values.
pub fn verify_plonk_bn254_public_inputs(
    vk: &DTVerifyingKey,
    public_values: &DTPublicValues,
    plonk_bn254_public_inputs: &[String],
) -> Result<()> {
    let expected_vk_hash = BigUint::from_str(&plonk_bn254_public_inputs[0])?;
    let expected_public_values_hash = BigUint::from_str(&plonk_bn254_public_inputs[1])?;

    let vk_hash = vk.hash_bn254().as_canonical_biguint();
    if vk_hash != expected_vk_hash {
        return Err(PlonkVerificationError::InvalidVerificationKey.into());
    }

    verify_public_values(public_values, expected_public_values_hash)?;

    Ok(())
}

/// Verify the vk_hash and public_values_hash in the public inputs of the Groth16Bn254Proof match
/// the expected values.
pub fn verify_groth16_bn254_public_inputs(
    vk: &DTVerifyingKey,
    public_values: &DTPublicValues,
    groth16_bn254_public_inputs: &[String],
) -> Result<()> {
    let expected_vk_hash = BigUint::from_str(&groth16_bn254_public_inputs[0])?;
    let expected_public_values_hash = BigUint::from_str(&groth16_bn254_public_inputs[1])?;

    let vk_hash = vk.hash_bn254().as_canonical_biguint();
    if vk_hash != expected_vk_hash {
        return Err(Groth16VerificationError::InvalidVerificationKey.into());
    }

    verify_public_values(public_values, expected_public_values_hash)?;

    Ok(())
}

/// In zkDTVM, a proof's public values can either be hashed with SHA2 or Blake3. In zkDTVM V4, there
/// is no metadata attached to the proof about which hasher function was used for public values
/// hashing. Instead, when verifying the proof, the public values are hashed with SHA2 and Blake3,
/// and if either matches the `expected_public_values_hash`, the verification is successful.
///
/// The security for this verification in zkDTVM V4 derives from the fact that both SHA2 and Blake3
/// are designed to be collision resistant. It is computationally infeasible to find an input i1 for
/// SHA256 and an input i2 for Blake3 that the same hash value. Doing so would require breaking both
/// algorithms simultaneously.
fn verify_public_values(
    public_values: &DTPublicValues,
    expected_public_values_hash: BigUint,
) -> Result<()> {
    // First, check if the public values are hashed with SHA256. If that fails, attempt hashing with
    // Blake3. If neither match, return an error.
    let sha256_public_values_hash = public_values.hash_bn254();
    if sha256_public_values_hash != expected_public_values_hash {
        let blake3_public_values_hash = public_values.hash_bn254_with_fn(blake3_hash);
        if blake3_public_values_hash != expected_public_values_hash {
            return Err(Groth16VerificationError::InvalidPublicValues.into());
        }
    }

    Ok(())
}

impl<C: DTProverComponents> SubproofVerifier for DTProver<C> {
    fn verify_deferred_proof(
        &self,
        proof: &DTReduceProof<CoreSC>,
        vk: &SCStarkVerifyingKey<CoreSC>,
        vk_hash: [u32; 8],
        committed_value_digest: [u32; 8],
    ) -> Result<(), MachineVerificationError<CoreSC>> {
        // Check that the vk hash matches the vk hash from the input.
        if vk.hash_u32() != vk_hash {
            return Err(MachineVerificationError::InvalidPublicValues(
                "vk hash from syscall does not match vkey from input",
            ));
        }
        // Deferred proofs are still recursive-chain Poseidon2 proofs; final
        // SHA256 root_shrink proofs are not accepted as deferred proofs.
        let reduce_proof = DTReduceProof { vk: proof.vk.clone(), proof: proof.proof.clone() };
        if self
            .compress_prover
            .machine()
            .verify(
                &reduce_proof.vk,
                &SCMachineProof { shard_proofs: vec![reduce_proof.proof.clone()] },
                &mut self.compress_prover.config().challenger(),
                num_skip_rounds(),
                chip_log_height_threshold(),
            )
            .is_err()
        {
            self.shrink_prover.machine().verify(
                &reduce_proof.vk,
                &SCMachineProof { shard_proofs: vec![reduce_proof.proof.clone()] },
                &mut self.shrink_prover.config().challenger(),
                num_skip_rounds(),
                chip_log_height_threshold(),
            )?;
        }
        // Check that the committed value digest matches the one from syscall
        let public_values: &RecursionPublicValues<_> =
            proof.proof.public_values.as_slice().borrow();
        if public_values.vk_root != self.recursion_vk_root {
            return Err(MachineVerificationError::InvalidPublicValues("vk_root mismatch"));
        }

        for (i, word) in public_values.committed_value_digest.iter().enumerate() {
            if *word != committed_value_digest[i].into() {
                return Err(MachineVerificationError::InvalidPublicValues(
                    "committed_value_digest does not match",
                ));
            }
        }
        Ok(())
    }
}
