use core::fmt;
use dt_core_machine::riscv::MAX_LOG_NUMBER_OF_SHARDS;
use dt_stark::{global_d11::ProgramImageBoundaryV1, Word};
use p3_field::{AbstractField, PrimeField32};
use serde::{Deserialize, Serialize};

use crate::{
    config::{DIGEST_SIZE, F, POSEIDON2_WIDTH},
    proof_shape_dt::bus::PROOF_SHAPE_VK_META_PC_START,
    symbolic_expr_fixed_dt::RecursionChildRole,
    system_dt::{
        RecursionNativeProgram, RecursionProofRecord, RecursionRecord, RecursionStatementRole,
        StatementConfigRow,
    },
    transcript_dt::poseidon2::{poseidon2_permute, RecursionPoseidon2Memo},
};

pub const STATEMENT_CONFIG_CLASS_BAKED_LIFT: usize = 0;
/// ReduceL3's second baked class: vk_L2 children (the self-thread closure).
pub const STATEMENT_CONFIG_CLASS_BAKED_L2: usize = 1;
/// RootShrink's only baked class: vk_L3 children.
pub const STATEMENT_CONFIG_CLASS_BAKED_L3: usize = 2;

pub const NATIVE_RECURSION_NUM_PV_ELTS: usize = 159;
pub const NATIVE_RECURSION_NUM_PV_ELMS_TO_HASH: usize = 151;

pub const NATIVE_PV_COMMITTED_VALUE_DIGEST_START: usize = 0;
pub const NATIVE_PV_COMMITTED_VALUE_DIGEST_ELTS: usize = 32;
pub const NATIVE_PV_DEFERRED_PROOFS_DIGEST_START: usize = 32;
pub const NATIVE_PV_DEFERRED_PROOFS_DIGEST_ELTS: usize = 8;
pub const NATIVE_PV_START_PC: usize = 40;
pub const NATIVE_PV_NEXT_PC: usize = 41;
pub const NATIVE_PV_START_SHARD: usize = 42;
pub const NATIVE_PV_NEXT_SHARD: usize = 43;
pub const NATIVE_PV_START_EXECUTION_SHARD: usize = 44;
pub const NATIVE_PV_NEXT_EXECUTION_SHARD: usize = 45;
pub const NATIVE_PV_PREVIOUS_INIT_ADDR: usize = 46;
pub const NATIVE_PV_LAST_INIT_ADDR: usize = 47;
pub const NATIVE_PV_PREVIOUS_FINALIZE_ADDR: usize = 48;
pub const NATIVE_PV_LAST_FINALIZE_ADDR: usize = 49;
pub const NATIVE_PV_START_RECONSTRUCT_DEFERRED_DIGEST_START: usize = 50;
pub const NATIVE_PV_END_RECONSTRUCT_DEFERRED_DIGEST_START: usize = 58;
pub const NATIVE_PV_DT_VK_DIGEST_START: usize = 66;
pub const NATIVE_PV_VK_ROOT_START: usize = 74;
pub const NATIVE_PV_GLOBAL_INTERVAL_START: usize = 82;
pub const NATIVE_PV_GLOBAL_STATE_ELTS: usize = 33;
pub const NATIVE_PV_GLOBAL_INTERVAL_END: usize = 115;
pub const NATIVE_PV_IS_COMPLETE: usize = 148;
pub const NATIVE_PV_CONTAINS_EXECUTION_SHARD: usize = 149;
pub const NATIVE_PV_EXIT_CODE: usize = 150;
pub const NATIVE_PV_DIGEST_START: usize = 151;

pub const CORE_CHILD_NUM_PUBLIC_VALUES: usize = dt_stark::air::DT_PROOF_NUM_PV_ELTS;
pub const CORE_PV_COMMITTED_VALUE_DIGEST_START: usize = 0;
pub const CORE_PV_COMMITTED_VALUE_DIGEST_ELTS: usize = 32;
pub const CORE_PV_DEFERRED_PROOFS_DIGEST_START: usize = 32;
pub const CORE_PV_DEFERRED_PROOFS_DIGEST_ELTS: usize = 8;
pub const CORE_PV_START_PC: usize = 40;
pub const CORE_PV_NEXT_PC: usize = 41;
pub const CORE_PV_EXIT_CODE: usize = 42;
pub const CORE_PV_SHARD: usize = 43;
pub const CORE_PV_EXECUTION_SHARD: usize = 44;
pub const CORE_PV_PREVIOUS_INIT_ADDR: usize = 45;
pub const CORE_PV_LAST_INIT_ADDR: usize = 46;
pub const CORE_PV_PREVIOUS_FINALIZE_ADDR: usize = 47;
pub const CORE_PV_LAST_FINALIZE_ADDR: usize = 48;
pub const CORE_PV_START_CLK: usize = 49;
pub const CORE_PV_EXIT_CLK: usize = 50;
pub const CORE_PV_EMPTY: usize = 51;
pub const CORE_PV_GLOBAL_HAS: usize = dt_stark::air::GLOBAL_CLAIM_START;
pub const CORE_PV_GLOBAL_COUNT: usize = CORE_PV_GLOBAL_HAS + 1;
pub const CORE_PV_GLOBAL_INTERVAL_START: usize = CORE_PV_GLOBAL_HAS + 2;
pub const CORE_PV_GLOBAL_INTERVAL_END: usize = CORE_PV_GLOBAL_INTERVAL_START + 33;

macro_rules! ensure {
    ($cond:expr, $($arg:tt)*) => {
        if !$cond {
            return Err($crate::statement_dt::SpecStatementError::new(format!($($arg)*)));
        }
    };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeRecursionPublicValues<T> {
    pub committed_value_digest: [Word<T>; 8],
    pub deferred_proofs_digest: [T; 8],
    pub start_pc: T,
    pub next_pc: T,
    pub start_shard: T,
    pub next_shard: T,
    pub start_execution_shard: T,
    pub next_execution_shard: T,
    pub previous_init_addr: T,
    pub last_init_addr: T,
    pub previous_finalize_addr: T,
    pub last_finalize_addr: T,
    pub start_reconstruct_deferred_digest: [T; 8],
    pub end_reconstruct_deferred_digest: [T; 8],
    pub dt_vk_digest: [T; DIGEST_SIZE],
    pub vk_root: [T; DIGEST_SIZE],
    pub global_interval_start: [[T; 11]; 3],
    pub global_interval_end: [[T; 11]; 3],
    pub is_complete: T,
    pub contains_execution_shard: T,
    pub exit_code: T,
    pub digest: [T; DIGEST_SIZE],
}

impl<T: Default + Copy> Default for NativeRecursionPublicValues<T> {
    fn default() -> Self {
        Self {
            committed_value_digest: [Word([T::default(); 4]); 8],
            deferred_proofs_digest: [T::default(); 8],
            start_pc: T::default(),
            next_pc: T::default(),
            start_shard: T::default(),
            next_shard: T::default(),
            start_execution_shard: T::default(),
            next_execution_shard: T::default(),
            previous_init_addr: T::default(),
            last_init_addr: T::default(),
            previous_finalize_addr: T::default(),
            last_finalize_addr: T::default(),
            start_reconstruct_deferred_digest: [T::default(); 8],
            end_reconstruct_deferred_digest: [T::default(); 8],
            dt_vk_digest: [T::default(); DIGEST_SIZE],
            vk_root: [T::default(); DIGEST_SIZE],
            global_interval_start: [[T::default(); 11]; 3],
            global_interval_end: [[T::default(); 11]; 3],
            is_complete: T::default(),
            contains_execution_shard: T::default(),
            exit_code: T::default(),
            digest: [T::default(); DIGEST_SIZE],
        }
    }
}

impl<T: Default + Copy> NativeRecursionPublicValues<T> {
    pub fn as_array(&self) -> [T; NATIVE_RECURSION_NUM_PV_ELTS] {
        let mut values = [T::default(); NATIVE_RECURSION_NUM_PV_ELTS];
        for word_idx in 0..8 {
            for byte_idx in 0..4 {
                values[NATIVE_PV_COMMITTED_VALUE_DIGEST_START + 4 * word_idx + byte_idx] =
                    self.committed_value_digest[word_idx][byte_idx];
            }
        }
        values[NATIVE_PV_DEFERRED_PROOFS_DIGEST_START..NATIVE_PV_DEFERRED_PROOFS_DIGEST_START + 8]
            .copy_from_slice(&self.deferred_proofs_digest);
        values[NATIVE_PV_START_PC] = self.start_pc;
        values[NATIVE_PV_NEXT_PC] = self.next_pc;
        values[NATIVE_PV_START_SHARD] = self.start_shard;
        values[NATIVE_PV_NEXT_SHARD] = self.next_shard;
        values[NATIVE_PV_START_EXECUTION_SHARD] = self.start_execution_shard;
        values[NATIVE_PV_NEXT_EXECUTION_SHARD] = self.next_execution_shard;
        values[NATIVE_PV_PREVIOUS_INIT_ADDR] = self.previous_init_addr;
        values[NATIVE_PV_LAST_INIT_ADDR] = self.last_init_addr;
        values[NATIVE_PV_PREVIOUS_FINALIZE_ADDR] = self.previous_finalize_addr;
        values[NATIVE_PV_LAST_FINALIZE_ADDR] = self.last_finalize_addr;
        values[NATIVE_PV_START_RECONSTRUCT_DEFERRED_DIGEST_START..
            NATIVE_PV_START_RECONSTRUCT_DEFERRED_DIGEST_START + 8]
            .copy_from_slice(&self.start_reconstruct_deferred_digest);
        values[NATIVE_PV_END_RECONSTRUCT_DEFERRED_DIGEST_START..
            NATIVE_PV_END_RECONSTRUCT_DEFERRED_DIGEST_START + 8]
            .copy_from_slice(&self.end_reconstruct_deferred_digest);
        values[NATIVE_PV_DT_VK_DIGEST_START..NATIVE_PV_DT_VK_DIGEST_START + DIGEST_SIZE]
            .copy_from_slice(&self.dt_vk_digest);
        values[NATIVE_PV_VK_ROOT_START..NATIVE_PV_VK_ROOT_START + DIGEST_SIZE]
            .copy_from_slice(&self.vk_root);
        for coordinate in 0..3 {
            values[NATIVE_PV_GLOBAL_INTERVAL_START + coordinate * 11..
                NATIVE_PV_GLOBAL_INTERVAL_START + (coordinate + 1) * 11]
                .copy_from_slice(&self.global_interval_start[coordinate]);
            values[NATIVE_PV_GLOBAL_INTERVAL_END + coordinate * 11..
                NATIVE_PV_GLOBAL_INTERVAL_END + (coordinate + 1) * 11]
                .copy_from_slice(&self.global_interval_end[coordinate]);
        }
        values[NATIVE_PV_IS_COMPLETE] = self.is_complete;
        values[NATIVE_PV_CONTAINS_EXECUTION_SHARD] = self.contains_execution_shard;
        values[NATIVE_PV_EXIT_CODE] = self.exit_code;
        values[NATIVE_PV_DIGEST_START..NATIVE_PV_DIGEST_START + DIGEST_SIZE]
            .copy_from_slice(&self.digest);
        values
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpecStatement {
    pub public_values: NativeRecursionPublicValues<F>,
}

impl SpecStatement {
    pub fn from_record(
        record: &RecursionRecord,
        native_program: &RecursionNativeProgram<F>,
    ) -> Result<Self, SpecStatementError> {
        let program = &native_program.constraint_program;
        let statement_role = native_program.statement_role;
        let statement_config = &native_program.statement_config;
        if statement_role != RecursionStatementRole::ReduceL2 {
            ensure!(
                record.statement_vk_root == [F::zero(); DIGEST_SIZE],
                "statement vk_root input must be zero for {statement_role:?}"
            );
        }
        let mut proofs = record.proof_records.iter().collect::<Vec<_>>();
        proofs.sort_by_key(|proof| proof.proof_idx);
        ensure!(!proofs.is_empty(), "lift statement requires at least one child proof");
        for (expected, proof) in proofs.iter().enumerate() {
            ensure!(
                proof.proof_idx == expected,
                "lift statement proof_idx must be dense: expected {expected}, got {}",
                proof.proof_idx
            );
        }

        let mut values = NativeRecursionPublicValues::<F>::default();
        let mut dt_vk_digest = None;
        let mut current_pc = F::zero();
        let mut current_shard = 0u32;
        let mut current_execution_shard = 0u32;
        let mut current_init_addr = F::zero();
        let mut current_finalize_addr = F::zero();
        let mut committed_value_digest = [Word([F::zero(); 4]); 8];
        let mut committed_digest_is_live = false;
        let mut contains_execution_shard = false;
        let mut global_interval_start = None;
        let mut global_interval_end = None;

        for (row, proof) in proofs.iter().enumerate() {
            let shape = &proof.proof_shape;
            let expected_role_id = child_role_id(program.role);
            ensure!(
                shape.role_id == expected_role_id,
                "recursion statement child {} has role_id {} expected {} for {:?}",
                proof.proof_idx,
                shape.role_id,
                expected_role_id,
                program.role
            );
            match program.role {
                RecursionChildRole::Core => {
                    let child = CoreChildPublicValues::from_proof(proof)?;
                    append_global_interval(
                        &mut global_interval_start,
                        &mut global_interval_end,
                        child.global_interval_start,
                        child.global_interval_end,
                        proof.proof_idx,
                    )?;
                    ensure!(
                        child.deferred_proofs_digest.iter().all(|value| *value == F::zero()),
                        "lift statement child {} has a non-empty deferred_proofs_digest",
                        proof.proof_idx
                    );
                    ensure!(
                        child.empty == F::zero(),
                        "lift statement child {} has a non-zero core public-value padding slot",
                        proof.proof_idx
                    );
                    ensure!(
                        child.exit_code == F::zero(),
                        "lift statement child {} has non-zero exit_code",
                        proof.proof_idx
                    );

                    let shard = canonical_u32(child.shard, "shard", proof.proof_idx)?;
                    ensure_shard_in_range(shard, proof.proof_idx)?;
                    // Lift semantics: the exported dt_vk digest IS the shared core vk digest.
                    merge_statement_dt_vk_digest(
                        &mut dt_vk_digest,
                        child_vk_digest_with_memo(shape, &record.poseidon2_memo),
                        proof.proof_idx,
                    )?;

                    if row == 0 {
                        values.start_pc = child.start_pc;
                        values.start_shard = child.shard;
                        values.start_execution_shard = child.execution_shard;
                        values.previous_init_addr = child.previous_init_addr;
                        values.previous_finalize_addr = child.previous_finalize_addr;
                        current_pc = child.start_pc;
                        current_shard = shard;
                        current_execution_shard = canonical_u32(
                            child.execution_shard,
                            "execution_shard",
                            proof.proof_idx,
                        )?;
                        current_init_addr = child.previous_init_addr;
                        current_finalize_addr = child.previous_finalize_addr;
                        committed_value_digest = child.committed_value_digest;
                        committed_digest_is_live =
                            committed_digest_nonzero(&committed_value_digest);
                    }

                    ensure!(
                        child.shard == F::from_canonical_u32(current_shard),
                        "lift statement child {} shard chain mismatch: expected {}, got {}",
                        proof.proof_idx,
                        current_shard,
                        child.shard.as_canonical_u32()
                    );
                    ensure!(
                        child.start_pc == current_pc,
                        "lift statement child {} pc chain mismatch",
                        proof.proof_idx
                    );

                    let is_execution_child = child.start_clk != child.exit_clk;
                    if is_execution_child {
                        ensure!(
                            child.start_pc != F::zero(),
                            "lift statement child {} execution shard has start_pc=0",
                            proof.proof_idx
                        );
                        let execution_shard = canonical_u32(
                            child.execution_shard,
                            "execution_shard",
                            proof.proof_idx,
                        )?;
                        if !contains_execution_shard {
                            values.start_execution_shard = child.execution_shard;
                            current_execution_shard = execution_shard;
                            contains_execution_shard = true;
                        }
                        ensure!(
                            execution_shard == current_execution_shard,
                            "lift statement child {} execution shard chain mismatch: expected {}, got {}",
                            proof.proof_idx,
                            current_execution_shard,
                            execution_shard
                        );
                        current_execution_shard =
                            current_execution_shard.checked_add(1).ok_or_else(|| {
                                SpecStatementError::new(format!(
                                    "lift statement child {} execution shard counter overflowed",
                                    proof.proof_idx
                                ))
                            })?;
                    } else {
                        ensure!(
                            child.start_pc == child.next_pc,
                            "lift statement child {} non-execution shard changed pc",
                            proof.proof_idx
                        );
                        ensure!(
                            child.shard != F::one(),
                            "lift statement child {} has non-execution shard 1",
                            proof.proof_idx
                        );
                    }

                    ensure!(
                        child.previous_init_addr == current_init_addr,
                        "lift statement child {} init-address chain mismatch",
                        proof.proof_idx
                    );
                    ensure!(
                        child.previous_finalize_addr == current_finalize_addr,
                        "lift statement child {} finalize-address chain mismatch",
                        proof.proof_idx
                    );

                    if committed_digest_is_live || !is_execution_child {
                        ensure!(
                            child.committed_value_digest == committed_value_digest,
                            "lift statement child {} committed digest mismatch",
                            proof.proof_idx
                        );
                    }
                    if committed_digest_nonzero(&child.committed_value_digest) {
                        committed_digest_is_live = true;
                    }
                    committed_value_digest = child.committed_value_digest;

                    if child.shard == F::one() {
                        ensure!(
                            child.start_pc == shape.vk_meta[PROOF_SHAPE_VK_META_PC_START],
                            "lift statement child {} first shard start_pc does not match vk pc_start",
                            proof.proof_idx
                        );
                        ensure!(
                            child.previous_init_addr == F::zero(),
                            "lift statement child {} first shard previous_init_addr is non-zero",
                            proof.proof_idx
                        );
                        ensure!(
                            child.previous_finalize_addr == F::zero(),
                            "lift statement child {} first shard previous_finalize_addr is non-zero",
                            proof.proof_idx
                        );
                    }

                    current_pc = child.next_pc;
                    current_shard = current_shard.checked_add(1).ok_or_else(|| {
                        SpecStatementError::new(format!(
                            "lift statement child {} shard counter overflowed",
                            proof.proof_idx
                        ))
                    })?;
                    current_init_addr = child.last_init_addr;
                    current_finalize_addr = child.last_finalize_addr;
                }
                RecursionChildRole::Compress | RecursionChildRole::Shrink => {
                    let child = NativeChildPublicValues::from_proof(proof, &record.poseidon2_memo)?;
                    ensure_canonical_interval_point(
                        child.values.global_interval_start,
                        "native interval start",
                        proof.proof_idx,
                    )?;
                    ensure_canonical_interval_point(
                        child.values.global_interval_end,
                        "native interval end",
                        proof.proof_idx,
                    )?;
                    append_global_interval(
                        &mut global_interval_start,
                        &mut global_interval_end,
                        child.values.global_interval_start,
                        child.values.global_interval_end,
                        proof.proof_idx,
                    )?;
                    ensure!(
                        child.values.exit_code == F::zero(),
                        "reduce statement child {} has non-zero exit_code",
                        proof.proof_idx
                    );
                    ensure!(
                        child
                            .values
                            .start_reconstruct_deferred_digest
                            .iter()
                            .all(|value| *value == F::zero()) &&
                            child
                                .values
                                .end_reconstruct_deferred_digest
                                .iter()
                                .all(|value| *value == F::zero()),
                        "reduce statement child {} has non-empty reconstruct deferred digest",
                        proof.proof_idx
                    );

                    let start_shard =
                        canonical_u32(child.values.start_shard, "start_shard", proof.proof_idx)?;
                    let next_shard =
                        canonical_u32(child.values.next_shard, "next_shard", proof.proof_idx)?;
                    ensure_shard_in_range(start_shard, proof.proof_idx)?;
                    ensure_shard_in_range(next_shard, proof.proof_idx)?;
                    ensure!(
                        next_shard > start_shard,
                        "reduce statement child {} has a non-increasing shard span: start={} next={}",
                        proof.proof_idx,
                        start_shard,
                        next_shard
                    );

                    // Vk policy: the child's OWN vk digest resolves to an accepted class
                    // (baked rows / the ReduceL2 threaded slot); the exported dt_vk digest is
                    // the child's THREADED dt_vk PV, not the child's own vk digest.
                    match resolve_child_vk_class_with_memo(
                        proof,
                        record.statement_vk_root,
                        statement_config,
                        &record.poseidon2_memo,
                    )? {
                        ChildVkClass::Core => {
                            return Err(SpecStatementError::new(format!(
                                "reduce statement child {} resolved to the core vk class",
                                proof.proof_idx
                            )));
                        }
                        ChildVkClass::Baked(row_idx) => {
                            let row = &statement_config[row_idx];
                            // Per-role export policy (host mirror of the in-circuit
                            // class gates).
                            match (statement_role, row.class_id) {
                                (
                                    RecursionStatementRole::ReduceL2 |
                                    RecursionStatementRole::ReduceL3,
                                    STATEMENT_CONFIG_CLASS_BAKED_LIFT,
                                ) |
                                (
                                    RecursionStatementRole::RootShrink,
                                    STATEMENT_CONFIG_CLASS_BAKED_L3,
                                ) => {
                                    ensure!(
                                        child.values.vk_root == [F::zero(); DIGEST_SIZE],
                                        "statement baked child {} of class {} exports a non-zero vk_root",
                                        proof.proof_idx,
                                        row.class_id
                                    );
                                }
                                (
                                    RecursionStatementRole::ReduceL3,
                                    STATEMENT_CONFIG_CLASS_BAKED_L2,
                                ) => {
                                    // The self-thread closure: BAKED_L2 children export the
                                    // vk_L2 digest (their threaded vk_root == the class digest).
                                    ensure!(
                                        child.values.vk_root == row.digest,
                                        "statement BAKED_L2 child {} does not export the vk_L2 digest",
                                        proof.proof_idx
                                    );
                                }
                                _ => {
                                    return Err(SpecStatementError::new(format!(
                                        "statement child {} matched baked class {} unsupported at {:?}",
                                        proof.proof_idx, row.class_id, statement_role
                                    )));
                                }
                            }
                        }
                        ChildVkClass::Threaded => {
                            ensure!(
                                statement_role == RecursionStatementRole::ReduceL2,
                                "reduce statement child {} used the threaded-self class outside ReduceL2",
                                proof.proof_idx
                            );
                            ensure!(
                                child.values.vk_root == record.statement_vk_root,
                                "reduce statement threaded child {} does not re-export the threaded vk_root",
                                proof.proof_idx
                            );
                        }
                    }
                    merge_statement_dt_vk_digest(
                        &mut dt_vk_digest,
                        child.values.dt_vk_digest,
                        proof.proof_idx,
                    )?;

                    if row == 0 {
                        values.start_pc = child.values.start_pc;
                        values.start_shard = child.values.start_shard;
                        values.start_execution_shard = child.values.start_execution_shard;
                        values.previous_init_addr = child.values.previous_init_addr;
                        values.previous_finalize_addr = child.values.previous_finalize_addr;
                        current_pc = child.values.start_pc;
                        current_shard = start_shard;
                        current_execution_shard = canonical_u32(
                            child.values.start_execution_shard,
                            "start_execution_shard",
                            proof.proof_idx,
                        )?;
                        current_init_addr = child.values.previous_init_addr;
                        current_finalize_addr = child.values.previous_finalize_addr;
                        committed_value_digest = child.values.committed_value_digest;
                        committed_digest_is_live =
                            committed_digest_nonzero(&committed_value_digest);
                    }

                    ensure!(
                        child.values.start_shard == F::from_canonical_u32(current_shard),
                        "reduce statement child {} shard chain mismatch: expected {}, got {}",
                        proof.proof_idx,
                        current_shard,
                        child.values.start_shard.as_canonical_u32()
                    );
                    ensure!(
                        child.values.start_pc == current_pc,
                        "reduce statement child {} pc chain mismatch",
                        proof.proof_idx
                    );
                    ensure!(
                        child.values.previous_init_addr == current_init_addr,
                        "reduce statement child {} init-address chain mismatch",
                        proof.proof_idx
                    );
                    ensure!(
                        child.values.previous_finalize_addr == current_finalize_addr,
                        "reduce statement child {} finalize-address chain mismatch",
                        proof.proof_idx
                    );

                    let is_execution_child = child.values.contains_execution_shard == F::one();
                    ensure!(
                        child.values.contains_execution_shard == F::zero() ||
                            child.values.contains_execution_shard == F::one(),
                        "reduce statement child {} has non-boolean contains_execution_shard",
                        proof.proof_idx
                    );
                    if is_execution_child {
                        ensure!(
                            child.values.start_pc != F::zero(),
                            "reduce statement child {} execution shard has start_pc=0",
                            proof.proof_idx
                        );
                        let execution_shard = canonical_u32(
                            child.values.start_execution_shard,
                            "start_execution_shard",
                            proof.proof_idx,
                        )?;
                        let next_execution_shard = canonical_u32(
                            child.values.next_execution_shard,
                            "next_execution_shard",
                            proof.proof_idx,
                        )?;
                        if !contains_execution_shard {
                            values.start_execution_shard = child.values.start_execution_shard;
                            current_execution_shard = execution_shard;
                            contains_execution_shard = true;
                        }
                        ensure!(
                            execution_shard == current_execution_shard,
                            "reduce statement child {} execution shard chain mismatch: expected {}, got {}",
                            proof.proof_idx,
                            current_execution_shard,
                            execution_shard
                        );
                        ensure!(
                            next_execution_shard > execution_shard,
                            "reduce statement child {} has a non-increasing execution span: start={} next={}",
                            proof.proof_idx,
                            execution_shard,
                            next_execution_shard
                        );
                        current_execution_shard = next_execution_shard;
                    } else {
                        ensure!(
                            child.values.start_pc == child.values.next_pc,
                            "reduce statement child {} non-execution shard changed pc",
                            proof.proof_idx
                        );
                        ensure!(
                            child.values.next_execution_shard == child.values.start_execution_shard,
                            "reduce statement child {} non-execution shard changed execution counter",
                            proof.proof_idx
                        );
                        ensure!(
                            child.values.start_shard != F::one(),
                            "reduce statement child {} contains no execution shard but starts at shard 1",
                            proof.proof_idx
                        );
                    }

                    if committed_digest_is_live || !is_execution_child {
                        ensure!(
                            child.values.committed_value_digest == committed_value_digest,
                            "reduce statement child {} committed digest mismatch",
                            proof.proof_idx
                        );
                    }
                    if committed_digest_nonzero(&child.values.committed_value_digest) {
                        committed_digest_is_live = true;
                    }
                    committed_value_digest = child.values.committed_value_digest;
                    current_pc = child.values.next_pc;
                    current_shard = next_shard;
                    current_init_addr = child.values.last_init_addr;
                    current_finalize_addr = child.values.last_finalize_addr;
                }
            }
        }

        values.committed_value_digest = committed_value_digest;
        values.next_pc = current_pc;
        values.next_shard = F::from_canonical_u32(current_shard);
        values.next_execution_shard = F::from_canonical_u32(current_execution_shard);
        values.last_init_addr = current_init_addr;
        values.last_finalize_addr = current_finalize_addr;
        values.dt_vk_digest = dt_vk_digest.expect("non-empty proof list sets vk digest");
        // Parent export policy: ReduceL2 re-exports its threaded vk_root input; every
        // other statement role exports vk_root = 0 (enforced as zero input above).
        values.vk_root = match statement_role {
            RecursionStatementRole::ReduceL2 => record.statement_vk_root,
            _ => [F::zero(); DIGEST_SIZE],
        };
        values.global_interval_start =
            global_interval_start.expect("non-empty proof list sets Global interval start");
        values.global_interval_end =
            global_interval_end.expect("non-empty proof list sets Global interval end");
        values.is_complete = F::from_bool(record.statement_is_complete);
        values.contains_execution_shard = F::from_bool(contains_execution_shard);
        values.exit_code = F::zero();

        if record.statement_is_complete {
            check_complete_statement(&values)?;
        }

        let mut array = values.as_array();
        // The PV digest slot @151..159 carries the role-selected instance:
        // RootShrink exports the ROOT_DIGEST form, every other role the SELF digest.
        values.digest = match statement_role {
            RecursionStatementRole::RootShrink => {
                crate::statement_hash_air_dt::root_public_values_digest_with_memo(
                    &array,
                    &record.poseidon2_memo,
                )
            }
            _ => poseidon2_hash_slice_with_memo(
                &crate::statement_hash_air_dt::statement_self_digest_hash_input(&array),
                &record.poseidon2_memo,
            ),
        };
        array[NATIVE_PV_DIGEST_START..NATIVE_PV_DIGEST_START + DIGEST_SIZE]
            .copy_from_slice(&values.digest);
        debug_assert_eq!(array, values.as_array());

        Ok(Self { public_values: values })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpecStatementError {
    message: String,
}

impl SpecStatementError {
    pub fn new(message: impl Into<String>) -> Self {
        Self { message: message.into() }
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for SpecStatementError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for SpecStatementError {}

fn global_identity_coordinates() -> [[F; 11]; 3] {
    let mut point = [[F::zero(); 11]; 3];
    point[1][0] = F::one();
    point
}

fn ensure_canonical_interval_point(
    point: [[F; 11]; 3],
    name: &str,
    proof_idx: usize,
) -> Result<(), SpecStatementError> {
    ensure!(point[2][0] == F::zero() || point[2][0] == F::one(),
        "statement child {proof_idx} {name} has non-boolean canonical Z");
    ensure!(point[2][1..].iter().all(|value| *value == F::zero()),
        "statement child {proof_idx} {name} has non-canonical Z tail");
    if point[2][0] == F::zero() {
        ensure!(point == global_identity_coordinates(),
            "statement child {proof_idx} {name} has non-canonical infinity");
    }
    Ok(())
}

fn append_global_interval(
    first: &mut Option<[[F; 11]; 3]>,
    last: &mut Option<[[F; 11]; 3]>,
    start: [[F; 11]; 3],
    end: [[F; 11]; 3],
    proof_idx: usize,
) -> Result<(), SpecStatementError> {
    ensure_canonical_interval_point(start, "Global interval start", proof_idx)?;
    ensure_canonical_interval_point(end, "Global interval end", proof_idx)?;
    if let Some(previous) = last {
        ensure!(*previous == start,
            "statement child {proof_idx} Global interval is discontinuous");
    } else {
        *first = Some(start);
    }
    *last = Some(end);
    Ok(())
}

fn program_seed_coordinates(boundary: &ProgramImageBoundaryV1<u32>) -> [[F; 11]; 3] {
    match boundary {
        ProgramImageBoundaryV1::Infinity => global_identity_coordinates(),
        ProgramImageBoundaryV1::Affine { x, y } => {
            let mut point = [[F::zero(); 11]; 3];
            point[0] = x.map(F::from_canonical_u32);
            point[1] = y.map(F::from_canonical_u32);
            point[2][0] = F::one();
            point
        }
    }
}

struct CoreChildPublicValues {
    committed_value_digest: [Word<F>; 8],
    deferred_proofs_digest: [F; 8],
    start_pc: F,
    next_pc: F,
    exit_code: F,
    shard: F,
    execution_shard: F,
    previous_init_addr: F,
    last_init_addr: F,
    previous_finalize_addr: F,
    last_finalize_addr: F,
    start_clk: F,
    exit_clk: F,
    empty: F,
    global_interval_start: [[F; 11]; 3],
    global_interval_end: [[F; 11]; 3],
}

impl CoreChildPublicValues {
    fn from_proof(proof: &RecursionProofRecord) -> Result<Self, SpecStatementError> {
        let shape = &proof.proof_shape;
        ensure!(
            shape.num_public_values == CORE_CHILD_NUM_PUBLIC_VALUES,
            "lift statement child {} num_public_values={} expected {}",
            proof.proof_idx,
            shape.num_public_values,
            CORE_CHILD_NUM_PUBLIC_VALUES
        );
        ensure!(
            shape.public_values.len() == CORE_CHILD_NUM_PUBLIC_VALUES,
            "lift statement child {} has {} public values, expected {}",
            proof.proof_idx,
            shape.public_values.len(),
            CORE_CHILD_NUM_PUBLIC_VALUES
        );

        let committed_value_digest = core::array::from_fn(|word_idx| {
            Word(core::array::from_fn(|byte_idx| {
                shape.public_values[CORE_PV_COMMITTED_VALUE_DIGEST_START + 4 * word_idx + byte_idx]
            }))
        });
        let deferred_proofs_digest = core::array::from_fn(|idx| {
            shape.public_values[CORE_PV_DEFERRED_PROOFS_DIGEST_START + idx]
        });
        let global_interval_start = core::array::from_fn(|coordinate| {
            core::array::from_fn(|limb| {
                shape.public_values[CORE_PV_GLOBAL_INTERVAL_START + coordinate * 11 + limb]
            })
        });
        let global_interval_end = core::array::from_fn(|coordinate| {
            core::array::from_fn(|limb| {
                shape.public_values[CORE_PV_GLOBAL_INTERVAL_END + coordinate * 11 + limb]
            })
        });
        let has = shape.public_values[CORE_PV_GLOBAL_HAS];
        let count = shape.public_values[CORE_PV_GLOBAL_COUNT];
        ensure!(has == F::zero() || has == F::one(), "core Global presence is not boolean");
        ensure_canonical_interval_point(
            global_interval_start,
            "core Global interval start",
            proof.proof_idx,
        )?;
        ensure_canonical_interval_point(
            global_interval_end,
            "core Global interval end",
            proof.proof_idx,
        )?;
        if has == F::one() {
            ensure!(count != F::zero(), "active core Global claim has zero count");
        } else {
            ensure!(count == F::zero(), "absent core Global claim has non-zero count");
            ensure!(
                global_interval_end == global_interval_start,
                "absent core Global claim does not preserve its running endpoint"
            );
        }

        Ok(Self {
            committed_value_digest,
            deferred_proofs_digest,
            start_pc: shape.public_values[CORE_PV_START_PC],
            next_pc: shape.public_values[CORE_PV_NEXT_PC],
            exit_code: shape.public_values[CORE_PV_EXIT_CODE],
            shard: shape.public_values[CORE_PV_SHARD],
            execution_shard: shape.public_values[CORE_PV_EXECUTION_SHARD],
            previous_init_addr: shape.public_values[CORE_PV_PREVIOUS_INIT_ADDR],
            last_init_addr: shape.public_values[CORE_PV_LAST_INIT_ADDR],
            previous_finalize_addr: shape.public_values[CORE_PV_PREVIOUS_FINALIZE_ADDR],
            last_finalize_addr: shape.public_values[CORE_PV_LAST_FINALIZE_ADDR],
            start_clk: shape.public_values[CORE_PV_START_CLK],
            exit_clk: shape.public_values[CORE_PV_EXIT_CLK],
            empty: shape.public_values[CORE_PV_EMPTY],
            global_interval_start,
            global_interval_end,
        })
    }
}

struct NativeChildPublicValues {
    values: NativeRecursionPublicValues<F>,
}

impl NativeChildPublicValues {
    fn from_proof(
        proof: &RecursionProofRecord,
        memo: &RecursionPoseidon2Memo,
    ) -> Result<Self, SpecStatementError> {
        let shape = &proof.proof_shape;
        ensure!(
            shape.num_public_values == NATIVE_RECURSION_NUM_PV_ELTS,
            "reduce statement child {} num_public_values={} expected {}",
            proof.proof_idx,
            shape.num_public_values,
            NATIVE_RECURSION_NUM_PV_ELTS
        );
        ensure!(
            shape.public_values.len() == NATIVE_RECURSION_NUM_PV_ELTS,
            "reduce statement child {} has {} public values, expected {}",
            proof.proof_idx,
            shape.public_values.len(),
            NATIVE_RECURSION_NUM_PV_ELTS
        );

        let values = native_values_from_public_slice(&shape.public_values);
        let digest = poseidon2_hash_slice_with_memo(
            &crate::statement_hash_air_dt::statement_self_digest_hash_input(&shape.public_values),
            memo,
        );
        ensure!(
            values.digest == digest,
            "reduce statement child {} has an invalid native statement digest",
            proof.proof_idx
        );
        Ok(Self { values })
    }
}

fn native_values_from_public_slice(public: &[F]) -> NativeRecursionPublicValues<F> {
    let mut values = NativeRecursionPublicValues::<F>::default();
    values.committed_value_digest = core::array::from_fn(|word_idx| {
        Word(core::array::from_fn(|byte_idx| {
            public[NATIVE_PV_COMMITTED_VALUE_DIGEST_START + 4 * word_idx + byte_idx]
        }))
    });
    values.deferred_proofs_digest =
        core::array::from_fn(|idx| public[NATIVE_PV_DEFERRED_PROOFS_DIGEST_START + idx]);
    values.start_pc = public[NATIVE_PV_START_PC];
    values.next_pc = public[NATIVE_PV_NEXT_PC];
    values.start_shard = public[NATIVE_PV_START_SHARD];
    values.next_shard = public[NATIVE_PV_NEXT_SHARD];
    values.start_execution_shard = public[NATIVE_PV_START_EXECUTION_SHARD];
    values.next_execution_shard = public[NATIVE_PV_NEXT_EXECUTION_SHARD];
    values.previous_init_addr = public[NATIVE_PV_PREVIOUS_INIT_ADDR];
    values.last_init_addr = public[NATIVE_PV_LAST_INIT_ADDR];
    values.previous_finalize_addr = public[NATIVE_PV_PREVIOUS_FINALIZE_ADDR];
    values.last_finalize_addr = public[NATIVE_PV_LAST_FINALIZE_ADDR];
    values.start_reconstruct_deferred_digest =
        core::array::from_fn(|idx| public[NATIVE_PV_START_RECONSTRUCT_DEFERRED_DIGEST_START + idx]);
    values.end_reconstruct_deferred_digest =
        core::array::from_fn(|idx| public[NATIVE_PV_END_RECONSTRUCT_DEFERRED_DIGEST_START + idx]);
    values.dt_vk_digest = core::array::from_fn(|idx| public[NATIVE_PV_DT_VK_DIGEST_START + idx]);
    values.vk_root = core::array::from_fn(|idx| public[NATIVE_PV_VK_ROOT_START + idx]);
    values.global_interval_start = core::array::from_fn(|coordinate| {
        core::array::from_fn(|limb| {
            public[NATIVE_PV_GLOBAL_INTERVAL_START + coordinate * 11 + limb]
        })
    });
    values.global_interval_end = core::array::from_fn(|coordinate| {
        core::array::from_fn(|limb| public[NATIVE_PV_GLOBAL_INTERVAL_END + coordinate * 11 + limb])
    });
    values.is_complete = public[NATIVE_PV_IS_COMPLETE];
    values.contains_execution_shard = public[NATIVE_PV_CONTAINS_EXECUTION_SHARD];
    values.exit_code = public[NATIVE_PV_EXIT_CODE];
    values.digest = core::array::from_fn(|idx| public[NATIVE_PV_DIGEST_START + idx]);
    values
}

fn child_role_id(role: RecursionChildRole) -> usize {
    match role {
        RecursionChildRole::Core => 0,
        RecursionChildRole::Compress => 1,
        RecursionChildRole::Shrink => 2,
    }
}

fn ensure_shard_in_range(shard: u32, proof_idx: usize) -> Result<(), SpecStatementError> {
    ensure!(
        shard < (1u32 << u32::try_from(MAX_LOG_NUMBER_OF_SHARDS).expect("fits u32")),
        "statement child {} shard {} exceeds 2^{}",
        proof_idx,
        shard,
        MAX_LOG_NUMBER_OF_SHARDS
    );
    Ok(())
}

fn merge_statement_dt_vk_digest(
    slot: &mut Option<[F; DIGEST_SIZE]>,
    dt_vk_digest: [F; DIGEST_SIZE],
    proof_idx: usize,
) -> Result<(), SpecStatementError> {
    if let Some(expected) = slot {
        ensure!(
            *expected == dt_vk_digest,
            "statement child {} breaks the dt_vk digest thread",
            proof_idx
        );
    } else {
        *slot = Some(dt_vk_digest);
    }
    Ok(())
}

/// Which vk class a child proof resolves to under this machine's vk-candidate table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChildVkClass {
    /// Core child (lift machine); vk identity feeds the dt_vk export, not the class table.
    Core,
    /// The child's own vk digest matched the baked candidate row at this index.
    Baked(usize),
    /// The child's own vk digest matched the node's threaded `vk_root` input (ReduceL2 only).
    Threaded,
}

pub fn resolve_child_vk_class(
    proof: &RecursionProofRecord,
    statement_vk_root: [F; DIGEST_SIZE],
    statement_config: &[StatementConfigRow],
) -> Result<ChildVkClass, SpecStatementError> {
    if proof.proof_shape.role_id == child_role_id(RecursionChildRole::Core) {
        return Ok(ChildVkClass::Core);
    }
    resolve_child_vk_class_from_digest(
        proof,
        statement_vk_root,
        statement_config,
        child_vk_digest(&proof.proof_shape),
    )
}

pub(crate) fn resolve_child_vk_class_with_memo(
    proof: &RecursionProofRecord,
    statement_vk_root: [F; DIGEST_SIZE],
    statement_config: &[StatementConfigRow],
    memo: &RecursionPoseidon2Memo,
) -> Result<ChildVkClass, SpecStatementError> {
    if proof.proof_shape.role_id == child_role_id(RecursionChildRole::Core) {
        return Ok(ChildVkClass::Core);
    }
    resolve_child_vk_class_from_digest(
        proof,
        statement_vk_root,
        statement_config,
        child_vk_digest_with_memo(&proof.proof_shape, memo),
    )
}

fn resolve_child_vk_class_from_digest(
    proof: &RecursionProofRecord,
    statement_vk_root: [F; DIGEST_SIZE],
    statement_config: &[StatementConfigRow],
    digest: [F; DIGEST_SIZE],
) -> Result<ChildVkClass, SpecStatementError> {
    if proof.proof_shape.role_id == child_role_id(RecursionChildRole::Core) {
        return Ok(ChildVkClass::Core);
    }
    if let Some(row_idx) = statement_config.iter().position(|row| row.digest == digest) {
        return Ok(ChildVkClass::Baked(row_idx));
    }
    if statement_vk_root != [F::zero(); DIGEST_SIZE] && digest == statement_vk_root {
        return Ok(ChildVkClass::Threaded);
    }
    Err(SpecStatementError::new(format!(
        "statement child {} vk digest is not an accepted class (baked rows or threaded slot)",
        proof.proof_idx
    )))
}

pub fn core_vk_statement_digest<C: AsRef<[F; DIGEST_SIZE]>>(
    commit: &C,
    pc_start: F,
    program_boundary: &dt_stark::global_d11::ProgramImageBoundaryV1<u32>,
    global146_identity: &[u8; 32],
) -> [F; DIGEST_SIZE] {
    dt_stark::global_d11::validate_global146_identity(global146_identity)
        .expect("core VK has the current Global146 identity");
    let boundary = dt_stark::global_d11::canonical_program_boundary_fields_v1::<F>(
        program_boundary,
    )
    .expect("validated VK program boundary");
    let mut input = Vec::with_capacity(64);
    input.extend_from_slice(commit.as_ref());
    input.push(pc_start);
    input.extend_from_slice(&boundary);
    append_global146_identity(&mut input, global146_identity);
    poseidon2_hash_slice(&input)
}

pub fn native_vk_statement_digest<C: AsRef<[F; DIGEST_SIZE]>>(
    commit: &C,
    global146_identity: &[u8; 32],
) -> [F; DIGEST_SIZE] {
    dt_stark::global_d11::validate_global146_identity(global146_identity)
        .expect("native VK has the current Global146 identity");
    let mut input = Vec::with_capacity(40);
    input.extend_from_slice(commit.as_ref());
    append_global146_identity(&mut input, global146_identity);
    poseidon2_hash_slice(&input)
}

pub fn child_vk_digest(shape: &crate::system_dt::RecursionProofShapeRecord) -> [F; DIGEST_SIZE] {
    poseidon2_hash_slice(&child_vk_digest_input(shape))
}

pub(crate) fn child_vk_digest_with_memo(
    shape: &crate::system_dt::RecursionProofShapeRecord,
    memo: &RecursionPoseidon2Memo,
) -> [F; DIGEST_SIZE] {
    poseidon2_hash_slice_with_memo(&child_vk_digest_input(shape), memo)
}

pub(crate) fn child_vk_digest_input(shape: &crate::system_dt::RecursionProofShapeRecord) -> Vec<F> {
    let count = if shape.role_id == child_role_id(RecursionChildRole::Core) {
        crate::proof_shape_dt::bus::PROOF_SHAPE_CORE_VK_META_VALUE_COUNT
    } else {
        crate::proof_shape_dt::bus::PROOF_SHAPE_NATIVE_VK_META_VALUE_COUNT
    };
    let mut input = shape.vk_meta[..count].to_vec();
    input[..DIGEST_SIZE].copy_from_slice(&shape.vk_commit);
    append_global146_identity(&mut input, &dt_stark::global_d11::GLOBAL146_COMPOSITE_IDENTITY);
    input
}

fn append_global146_identity(input: &mut Vec<F>, identity: &[u8; 32]) {
    input.resize(input.len().next_multiple_of(8), F::zero());
    input.extend(identity.map(F::from_canonical_u8));
}

pub fn poseidon2_hash_slice(input: &[F]) -> [F; DIGEST_SIZE] {
    poseidon2_hash_slice_by(input, poseidon2_permute)
}

pub(crate) fn poseidon2_hash_slice_with_memo(
    input: &[F],
    memo: &RecursionPoseidon2Memo,
) -> [F; DIGEST_SIZE] {
    poseidon2_hash_slice_by(input, |state| memo.permute(state))
}

fn poseidon2_hash_slice_by(
    input: &[F],
    mut permute: impl FnMut([F; POSEIDON2_WIDTH]) -> [F; POSEIDON2_WIDTH],
) -> [F; DIGEST_SIZE] {
    let mut state = [F::zero(); POSEIDON2_WIDTH];
    for chunk in input.chunks(8) {
        for (lane, value) in chunk.iter().enumerate() {
            state[lane] += *value;
        }
        state = permute(state);
    }
    state[..DIGEST_SIZE].try_into().expect("digest width matches")
}

fn committed_digest_nonzero(digest: &[Word<F>; 8]) -> bool {
    digest.iter().flat_map(|word| word.0).any(|value| value != F::zero())
}

fn canonical_u32(value: F, name: &str, proof_idx: usize) -> Result<u32, SpecStatementError> {
    let canonical = value.as_canonical_u32();
    ensure!(
        F::from_canonical_u32(canonical) == value,
        "lift statement child {proof_idx} {name} is not canonical u32"
    );
    Ok(canonical)
}

fn check_complete_statement(
    values: &NativeRecursionPublicValues<F>,
) -> Result<(), SpecStatementError> {
    ensure!(values.next_pc == F::zero(), "complete statement has non-zero next_pc");
    ensure!(values.start_shard == F::one(), "complete statement does not start at shard 1");
    ensure!(values.next_shard != F::one(), "complete statement has next_shard=1");
    ensure!(
        values.contains_execution_shard == F::one(),
        "complete statement contains no execution shard"
    );
    ensure!(
        values.start_execution_shard == F::one(),
        "complete statement does not start at execution shard 1"
    );
    ensure!(
        values.deferred_proofs_digest.iter().all(|value| *value == F::zero()),
        "complete statement has non-empty deferred digest"
    );
    ensure!(
        values.start_reconstruct_deferred_digest.iter().all(|value| *value == F::zero()),
        "complete statement has non-empty start reconstruct deferred digest"
    );
    ensure!(
        values.end_reconstruct_deferred_digest == values.deferred_proofs_digest,
        "complete statement reconstruct digest does not match deferred digest"
    );
    ensure_canonical_interval_point(values.global_interval_start, "root interval start", 0)?;
    ensure_canonical_interval_point(values.global_interval_end, "root interval end", 0)?;
    ensure!(values.global_interval_end == global_identity_coordinates(),
        "complete statement Global interval does not end at exact identity");
    Ok(())
}

pub(crate) fn validate_native_root_global_interval(
    program_boundary: &dt_stark::global_d11::ProgramImageBoundaryV1<u32>,
    start: [[F; 11]; 3],
    end: [[F; 11]; 3],
) -> Result<(), SpecStatementError> {
    ensure_canonical_interval_point(start, "root interval start", 0)?;
    ensure_canonical_interval_point(end, "root interval end", 0)?;
    ensure!(start == program_seed_coordinates(program_boundary),
        "root Global interval start does not match the authenticated program seed");
    ensure!(end == global_identity_coordinates(),
        "root Global interval end is not exact canonical identity");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        symbolic_ir_dt::{
            RecursionD0CostLedger, RecursionPolyAirChipIr, RecursionPolyAirVerifierProgram,
            RecursionPolyAirWidths,
        },
        system_dt::{RecursionProofRecord, RecursionProofShapeChip, RecursionProofShapeRecord},
    };
    use dt_stark::air::InteractionScope;

    #[test]
    fn schema_offsets_match_dsl_recursion_public_values() {
        assert_eq!(NATIVE_RECURSION_NUM_PV_ELTS, 159);
        assert_eq!(NATIVE_RECURSION_NUM_PV_ELMS_TO_HASH, NATIVE_PV_DIGEST_START);
        assert_eq!(NATIVE_PV_COMMITTED_VALUE_DIGEST_START, 0);
        assert_eq!(NATIVE_PV_DEFERRED_PROOFS_DIGEST_START, 32);
        assert_eq!(NATIVE_PV_START_PC, 40);
        assert_eq!(NATIVE_PV_GLOBAL_INTERVAL_START, 82);
        assert_eq!(NATIVE_PV_GLOBAL_INTERVAL_END, 115);
        assert_eq!(NATIVE_PV_IS_COMPLETE, 148);
        assert_eq!(NATIVE_PV_CONTAINS_EXECUTION_SHARD, 149);
        assert_eq!(NATIVE_PV_EXIT_CODE, 150);
        assert_eq!(NATIVE_PV_DIGEST_START + DIGEST_SIZE, NATIVE_RECURSION_NUM_PV_ELTS);
    }

    #[test]
    fn host_lift_statement_builds_two_child_output() {
        let program = test_program();
        let identity = global_identity_coordinates();
        let middle = test_finite_point(3);
        let end = test_finite_point(7);
        let mut first = test_proof(0, 2, 7, 9, 10, 0, 4, true);
        let mut second = test_proof(1, 3, 9, 10, 11, 4, 8, true);
        set_core_global_interval(&mut first, identity, middle);
        set_core_global_interval(&mut second, middle, end);
        let record = RecursionRecord {
            proof_records: vec![first, second],
            ..Default::default()
        };

        let statement =
            SpecStatement::from_record(&record, &native_test_program(program)).expect("statement");
        assert!(record.poseidon2_memo.snapshot().misses > 0);
        let values = statement.public_values;
        assert_eq!(values.start_pc, f(7));
        assert_eq!(values.next_pc, f(11));
        assert_eq!(values.start_shard, f(2));
        assert_eq!(values.next_shard, f(4));
        assert_eq!(values.start_execution_shard, f(9));
        assert_eq!(values.next_execution_shard, f(11));
        assert_eq!(values.previous_init_addr, f(10));
        assert_eq!(values.last_init_addr, f(12));
        assert_eq!(values.contains_execution_shard, F::one());
        assert_eq!(values.global_interval_start, identity);
        assert_eq!(values.global_interval_end, end);
        assert_eq!(
            values.digest,
            poseidon2_hash_slice(&crate::statement_hash_air_dt::statement_self_digest_hash_input(
                &values.as_array(),
            ),)
        );
    }

    #[test]
    fn host_lift_statement_rejects_sparse_proof_idx() {
        let program = test_program();
        let record = RecursionRecord {
            proof_records: vec![test_proof(1, 2, 7, 9, 10, 0, 4, true)],
            ..Default::default()
        };

        let err = SpecStatement::from_record(&record, &native_test_program(program))
            .expect_err("sparse idx");
        assert!(err.message().contains("dense"), "{err}");
    }

    #[test]
    fn host_lift_statement_rejects_nonzero_deferred_digest() {
        let program = test_program();
        let mut record = RecursionRecord {
            proof_records: vec![test_proof(0, 2, 7, 9, 10, 0, 4, true)],
            ..Default::default()
        };
        record.proof_records[0].proof_shape.public_values[CORE_PV_DEFERRED_PROOFS_DIGEST_START] =
            F::one();

        let err = SpecStatement::from_record(&record, &native_test_program(program))
            .expect_err("deferred");
        assert!(err.message().contains("deferred_proofs_digest"), "{err}");
    }

    #[test]
    fn native_vk_metadata_digest_uses_only_the_native_eight_limb_layout() {
        let shape = RecursionProofShapeRecord {
            role_id: child_role_id(RecursionChildRole::Compress),
            vk_commit: [f(9); DIGEST_SIZE],
            vk_meta: vec![f(9); crate::proof_shape_dt::PROOF_SHAPE_NATIVE_VK_META_VALUE_COUNT],
            ..Default::default()
        };
        assert_eq!(shape.vk_meta.len(), 8);
        assert!(!child_vk_digest_input(&shape).is_empty());
    }

    #[test]
    fn native_l3_statement_accepts_an_eight_limb_child_vk_layout() {
        let identity = global_identity_coordinates();
        let end = test_finite_point(9);
        let mut values = NativeRecursionPublicValues::<F>::default();
        values.start_pc = f(7);
        values.next_pc = f(7);
        values.start_shard = f(2);
        values.next_shard = f(3);
        values.start_execution_shard = f(9);
        values.next_execution_shard = f(9);
        values.dt_vk_digest = [f(5); DIGEST_SIZE];
        values.global_interval_start = identity;
        values.global_interval_end = end;
        let mut public_values = values.as_array().to_vec();
        let digest = poseidon2_hash_slice(
            &crate::statement_hash_air_dt::statement_self_digest_hash_input(&public_values),
        );
        public_values[NATIVE_PV_DIGEST_START..NATIVE_PV_DIGEST_START + DIGEST_SIZE]
            .copy_from_slice(&digest);
        let mut proof = RecursionProofRecord {
            proof_shape: RecursionProofShapeRecord {
                role_id: child_role_id(RecursionChildRole::Compress),
                num_public_values: NATIVE_RECURSION_NUM_PV_ELTS,
                vk_commit: [f(9); DIGEST_SIZE],
                vk_meta: vec![
                    f(9);
                    crate::proof_shape_dt::PROOF_SHAPE_NATIVE_VK_META_VALUE_COUNT
                ],
                public_values,
                public_value_send_mults: vec![0; NATIVE_RECURSION_NUM_PV_ELTS],
                vk_meta_send_mults: vec![
                    0;
                    crate::proof_shape_dt::PROOF_SHAPE_NATIVE_VK_META_VALUE_COUNT
                ],
                ..Default::default()
            },
            ..Default::default()
        };
        let child_digest = child_vk_digest(&proof.proof_shape);
        let program = crate::system_dt::RecursionNativeProgram::new_with_roles(
            RecursionChildRole::Compress,
            RecursionStatementRole::ReduceL3,
            NATIVE_RECURSION_NUM_PV_ELTS,
            false,
            Vec::new(),
            test_program_for_role(RecursionChildRole::Compress),
            vec![
                StatementConfigRow {
                    class_id: STATEMENT_CONFIG_CLASS_BAKED_LIFT,
                    digest: child_digest,
                },
                StatementConfigRow {
                    class_id: STATEMENT_CONFIG_CLASS_BAKED_L2,
                    digest: [f(11); DIGEST_SIZE],
                },
            ],
        );
        proof.proof_idx = 0;
        let record = RecursionRecord { proof_records: vec![proof], ..Default::default() };

        let statement = SpecStatement::from_record(&record, &program).expect("L3 statement");
        assert_eq!(record.proof_records[0].proof_shape.vk_meta.len(), 8);
        assert_eq!(statement.public_values.global_interval_start, identity);
        assert_eq!(statement.public_values.global_interval_end, end);
    }

    #[test]
    fn native_root_interval_binds_seed_and_exact_identity_y() {
        let start = test_finite_point(5);
        let boundary = dt_stark::global_d11::ProgramImageBoundaryV1::Affine {
            x: start[0].map(|value| value.as_canonical_u32()),
            y: start[1].map(|value| value.as_canonical_u32()),
        };
        let identity = global_identity_coordinates();
        validate_native_root_global_interval(&boundary, start, identity).unwrap();
        assert!(validate_native_root_global_interval(
            &boundary,
            test_finite_point(6),
            identity,
        )
        .is_err());
        let mut wrong_infinity = identity;
        wrong_infinity[1][0] = f(2);
        assert!(validate_native_root_global_interval(
            &dt_stark::global_d11::ProgramImageBoundaryV1::Infinity,
            identity,
            wrong_infinity,
        )
        .is_err());
    }

    fn native_test_program(
        constraint_program: RecursionPolyAirVerifierProgram,
    ) -> crate::system_dt::RecursionNativeProgram<F> {
        crate::system_dt::RecursionNativeProgram::new_core(
            CORE_CHILD_NUM_PUBLIC_VALUES,
            true,
            Vec::new(),
            constraint_program,
        )
    }

    fn test_program() -> RecursionPolyAirVerifierProgram {
        test_program_for_role(RecursionChildRole::Core)
    }

    fn test_program_for_role(role: RecursionChildRole) -> RecursionPolyAirVerifierProgram {
        RecursionPolyAirVerifierProgram::try_new(
            crate::symbolic_ir_dt::CONSTRAINT_PROGRAM_SCHEMA_VERSION,
            role,
            [F::zero(); DIGEST_SIZE],
            vec![RecursionPolyAirChipIr {
                static_chip_id: 0,
                chip_name: "global".to_string(),
                widths: RecursionPolyAirWidths { preprocessed: 0, main: 1, public: 0 },
                commit_scope: InteractionScope::Global,
                logup_batch_size: 2,
                reserved_poly: vec![],
                derived_roots: vec![
                    crate::symbolic_ir_dt::RecursionPolyAirDerivedRoot::BetaPower { power: 0 },
                    crate::symbolic_ir_dt::RecursionPolyAirDerivedRoot::BetaSeptix,
                ],
                gate_roots: vec![],
                lookup_multiplicity_roots: vec![],
                node_table: vec![],
                num_constraints_from_builder: 0,
                cost_ledger: RecursionD0CostLedger {
                    node_count: 0,
                    op_mix: Default::default(),
                    gate_count: 0,
                    precompute_root_count: 0,
                    derived_root_count: 2,
                    expected_node_bus_rows: 0,
                    expected_wide_unroll_rows: 1,
                    expected_wide_unroll_width: 0,
                    internal_recursion_interactions_node_bus: 0,
                    internal_recursion_interactions_wide_unroll: 0,
                },
            }],
            0,
        )
        .expect("statement test constraint program")
    }

    fn test_proof(
        proof_idx: usize,
        shard: u32,
        start_pc: u32,
        execution_shard: u32,
        init_addr: u32,
        start_clk: u32,
        exit_clk: u32,
        with_global_chip: bool,
    ) -> RecursionProofRecord {
        let mut public_values = vec![F::zero(); CORE_CHILD_NUM_PUBLIC_VALUES];
        public_values[CORE_PV_START_PC] = f(start_pc);
        public_values[CORE_PV_NEXT_PC] = f(start_pc + 2);
        public_values[CORE_PV_SHARD] = f(shard);
        public_values[CORE_PV_EXECUTION_SHARD] = f(execution_shard);
        public_values[CORE_PV_PREVIOUS_INIT_ADDR] = f(init_addr);
        public_values[CORE_PV_LAST_INIT_ADDR] = f(init_addr + 1);
        public_values[CORE_PV_PREVIOUS_FINALIZE_ADDR] = f(init_addr + 20);
        public_values[CORE_PV_LAST_FINALIZE_ADDR] = f(init_addr + 21);
        public_values[CORE_PV_START_CLK] = f(start_clk);
        public_values[CORE_PV_EXIT_CLK] = f(exit_clk);
        public_values[0] = f(42);

        let identity = [
            [F::zero(); 11],
            {
                let mut y = [F::zero(); 11];
                y[0] = F::one();
                y
            },
            [F::zero(); 11],
        ];
        public_values[CORE_PV_GLOBAL_HAS] = F::one();
        public_values[CORE_PV_GLOBAL_COUNT] = F::one();
        for coordinate in 0..3 {
            public_values[CORE_PV_GLOBAL_INTERVAL_START + coordinate * 11..
                CORE_PV_GLOBAL_INTERVAL_START + (coordinate + 1) * 11]
                .copy_from_slice(&identity[coordinate]);
            public_values[CORE_PV_GLOBAL_INTERVAL_END + coordinate * 11..
                CORE_PV_GLOBAL_INTERVAL_END + (coordinate + 1) * 11]
                .copy_from_slice(&identity[coordinate]);
        }
        let chips = if with_global_chip {
            vec![RecursionProofShapeChip {
                chip_idx: 0,
                static_chip_id: 0,
                stable_air_id: 43,
                log_height: 1,
                prep_width: 0,
                main_width: 1,
                perm_width: 1,
                constraint_count: 1,
                gate_count: 1,
            }]
        } else {
            vec![]
        };
        RecursionProofRecord {
            proof_idx,
            proof_shape: RecursionProofShapeRecord {
                role_id: 0,
                num_public_values: CORE_CHILD_NUM_PUBLIC_VALUES,
                vk_commit: [f(3); DIGEST_SIZE],
                vk_meta: {
                    let mut values = vec![
                            F::zero();
                            crate::proof_shape_dt::bus::PROOF_SHAPE_VK_META_VALUE_COUNT
                        ];
                    values[..DIGEST_SIZE].copy_from_slice(&[f(3); DIGEST_SIZE]);
                    values[crate::proof_shape_dt::bus::PROOF_SHAPE_VK_META_PC_START] = f(7);
                    values
                },
                public_values,
                public_value_send_mults: vec![0; CORE_CHILD_NUM_PUBLIC_VALUES],
                vk_meta_send_mults: vec![
                    0;
                    crate::proof_shape_dt::bus::PROOF_SHAPE_VK_META_VALUE_COUNT
                ],
                main_commit: [F::zero(); DIGEST_SIZE],
                permutation_commit: [F::zero(); DIGEST_SIZE],
                chips,
                publish_external: false,
                publish_whir_inputs: false,
                publish_terminal_summary: false,
            },
            ..Default::default()
        }
    }

    fn test_finite_point(tag: u32) -> [[F; 11]; 3] {
        let mut point = [[F::zero(); 11]; 3];
        point[0][0] = f(tag);
        point[1][0] = f(tag + 1);
        point[2][0] = F::one();
        point
    }

    fn set_core_global_interval(
        proof: &mut RecursionProofRecord,
        start: [[F; 11]; 3],
        end: [[F; 11]; 3],
    ) {
        for coordinate in 0..3 {
            proof.proof_shape.public_values[CORE_PV_GLOBAL_INTERVAL_START + coordinate * 11..
                CORE_PV_GLOBAL_INTERVAL_START + (coordinate + 1) * 11]
                .copy_from_slice(&start[coordinate]);
            proof.proof_shape.public_values[CORE_PV_GLOBAL_INTERVAL_END + coordinate * 11..
                CORE_PV_GLOBAL_INTERVAL_END + (coordinate + 1) * 11]
                .copy_from_slice(&end[coordinate]);
        }
    }

    fn f(value: u32) -> F {
        F::from_canonical_u32(value)
    }
}
