use core::{
    borrow::{Borrow, BorrowMut},
    ops::Deref,
};
use std::{collections::BTreeMap, sync::Arc};

use dt_stark::{
    air::{FullAir, FullAirBuilder, MachineAir, PairCol},
    sumcheck::trace::{CompressedMatrix, PaddingRow},
};
use native_recursion_derive::AlignedBorrow;
use p3_air::BaseAir;
use p3_field::{AbstractField, Field, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use crate::{
    config::{DIGEST_SIZE, F},
    interaction_full_air_dt::RecursionFullAirBus,
    interaction_registry_dt::{
        STATEMENT_CHILD_FACTS_SCHEMA, STATEMENT_DIGEST_CHAIN_SCHEMA,
        STATEMENT_GLOBAL_INTERVAL_CHAIN_SCHEMA, STATEMENT_HASH_CHAIN_SCHEMA,
        STATEMENT_SCALAR_CHAIN_SCHEMA, STATEMENT_VK_DIGEST_SCHEMA,
    },
    primitives_dt::bus::{RangeCheckerBus, RangeCheckerBusMessage},
    proof_shape_dt::{
        bus::{
            PROOF_SHAPE_NAMESPACE_VK_META, PROOF_SHAPE_VK_META_BOUNDARY_KIND,
            PROOF_SHAPE_VK_META_BOUNDARY_X_BASE,
        },
        ProofShapeSummaryBus, ProofShapeValuesBus,
    },
    statement_config_air_dt::StatementConfigBus,
    statement_dt::{
        child_vk_digest_with_memo, resolve_child_vk_class_with_memo, ChildVkClass,
        CORE_PV_COMMITTED_VALUE_DIGEST_START, CORE_PV_DEFERRED_PROOFS_DIGEST_START,
        CORE_PV_EXECUTION_SHARD, CORE_PV_EXIT_CLK, CORE_PV_EXIT_CODE, CORE_PV_GLOBAL_INTERVAL_END,
        CORE_PV_GLOBAL_INTERVAL_START, CORE_PV_LAST_FINALIZE_ADDR, CORE_PV_LAST_INIT_ADDR,
        CORE_PV_NEXT_PC, CORE_PV_PREVIOUS_FINALIZE_ADDR, CORE_PV_PREVIOUS_INIT_ADDR, CORE_PV_SHARD,
        CORE_PV_START_CLK, CORE_PV_START_PC, NATIVE_PV_COMMITTED_VALUE_DIGEST_START,
        NATIVE_PV_CONTAINS_EXECUTION_SHARD, NATIVE_PV_DEFERRED_PROOFS_DIGEST_START,
        NATIVE_PV_DT_VK_DIGEST_START, NATIVE_PV_END_RECONSTRUCT_DEFERRED_DIGEST_START,
        NATIVE_PV_EXIT_CODE, NATIVE_PV_GLOBAL_INTERVAL_END, NATIVE_PV_GLOBAL_INTERVAL_START,
        NATIVE_PV_IS_COMPLETE, NATIVE_PV_LAST_FINALIZE_ADDR, NATIVE_PV_LAST_INIT_ADDR,
        NATIVE_PV_NEXT_EXECUTION_SHARD, NATIVE_PV_NEXT_PC, NATIVE_PV_NEXT_SHARD,
        NATIVE_PV_PREVIOUS_FINALIZE_ADDR, NATIVE_PV_PREVIOUS_INIT_ADDR,
        NATIVE_PV_START_EXECUTION_SHARD, NATIVE_PV_START_PC,
        NATIVE_PV_START_RECONSTRUCT_DEFERRED_DIGEST_START, NATIVE_PV_START_SHARD,
        NATIVE_PV_VK_ROOT_START, NATIVE_RECURSION_NUM_PV_ELTS,
    },
    statement_hash_air_dt::STATEMENT_HASH_KIND_VK_DIGEST,
    system_dt::{
        RecursionNativeProgram, RecursionRecord, RecursionStatementRole, StatementConfigRow,
    },
    whir_dt::WHIR_ROLE_CORE,
};

pub const STATEMENT_CVD_CHUNKS: usize = 4;
pub const STATEMENT_PAYLOAD_ELTS: usize = 8;
pub const STATEMENT_SCALAR_STATE_ELTS: usize = 11;
const STATEMENT_SCALAR_VALUE_ELTS: usize = 14;
const STATEMENT_DEFERRED_DIGEST_ELTS: usize = 8;
pub const STATEMENT_PV_SLOTS: usize = 33;
pub const STATEMENT_GLOBAL_STATE_ELTS: usize = 33;
pub const STATEMENT_GLOBAL_CHUNK_ELTS: usize = 11;
pub const STATEMENT_GLOBAL_CHUNKS: usize =
    STATEMENT_GLOBAL_STATE_ELTS.div_ceil(STATEMENT_GLOBAL_CHUNK_ELTS);
const STATEMENT_NATIVE_EXTRA_ELTS: usize = 26;
const STATEMENT_NATIVE_SCALAR_EXTRA_ELTS: usize = 22;

const SCALAR_PC: usize = 0;
const SCALAR_SHARD: usize = 1;
const SCALAR_EXEC: usize = 2;
const SCALAR_EXEC_SEEN: usize = 3;
const SCALAR_INIT_ADDR: usize = 4;
const SCALAR_FIN_ADDR: usize = 5;
const SCALAR_START_PC_OUT: usize = 6;
const SCALAR_START_SHARD_OUT: usize = 7;
const SCALAR_START_EXEC_OUT: usize = 8;
const SCALAR_PREV_INIT_OUT: usize = 9;
const SCALAR_PREV_FIN_OUT: usize = 10;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StatementScalarChainBus {
    bus: RecursionFullAirBus,
}

impl StatementScalarChainBus {
    pub const fn new() -> Self {
        Self { bus: RecursionFullAirBus::new(STATEMENT_SCALAR_CHAIN_SCHEMA) }
    }

    pub const fn required_max_beta_power_floor(&self) -> usize {
        self.bus.required_max_beta_power_floor()
    }

    pub fn denominator<AB: FullAirBuilder>(
        &self,
        builder: &AB,
        cursor: AB::VarMaybeExt,
        state: [AB::VarMaybeExt; STATEMENT_SCALAR_STATE_ELTS],
    ) -> AB::VarExt {
        let mut values = Vec::with_capacity(1 + STATEMENT_SCALAR_STATE_ELTS);
        values.push(cursor);
        values.extend(state);
        self.bus.denominator(builder, values)
    }
}

impl Default for StatementScalarChainBus {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StatementDigestChainBus {
    bus: RecursionFullAirBus,
}

impl StatementDigestChainBus {
    pub const fn new() -> Self {
        Self { bus: RecursionFullAirBus::new(STATEMENT_DIGEST_CHAIN_SCHEMA) }
    }

    pub const fn required_max_beta_power_floor(&self) -> usize {
        self.bus.required_max_beta_power_floor()
    }

    pub fn denominator<AB: FullAirBuilder>(
        &self,
        builder: &AB,
        chunk: AB::VarMaybeExt,
        cursor: AB::VarMaybeExt,
        digest: [AB::VarMaybeExt; STATEMENT_PAYLOAD_ELTS],
    ) -> AB::VarExt {
        let mut values = Vec::with_capacity(2 + STATEMENT_PAYLOAD_ELTS);
        values.push(chunk);
        values.push(cursor);
        values.extend(digest);
        self.bus.denominator(builder, values)
    }
}

impl Default for StatementDigestChainBus {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StatementGlobalIntervalChainBus {
    bus: RecursionFullAirBus,
}

impl StatementGlobalIntervalChainBus {
    pub const fn new() -> Self {
        Self { bus: RecursionFullAirBus::new(STATEMENT_GLOBAL_INTERVAL_CHAIN_SCHEMA) }
    }

    pub const fn required_max_beta_power_floor(&self) -> usize {
        self.bus.required_max_beta_power_floor()
    }

    pub fn denominator<AB: FullAirBuilder>(
        &self,
        builder: &AB,
        chunk: AB::VarMaybeExt,
        cursor: AB::VarMaybeExt,
        state: [AB::VarMaybeExt; STATEMENT_GLOBAL_CHUNK_ELTS],
    ) -> AB::VarExt {
        let mut values = Vec::with_capacity(2 + STATEMENT_GLOBAL_CHUNK_ELTS);
        values.push(chunk);
        values.push(cursor);
        values.extend(state);
        self.bus.denominator(builder, values)
    }
}

impl Default for StatementGlobalIntervalChainBus {
    fn default() -> Self {
        Self::new()
    }
}

macro_rules! per_proof_bus {
    ($name:ident, $schema:ident, $arity:expr) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub struct $name {
            bus: RecursionFullAirBus,
        }

        impl $name {
            pub const fn new() -> Self {
                Self { bus: RecursionFullAirBus::new($schema) }
            }
            pub const fn required_max_beta_power_floor(&self) -> usize {
                self.bus.required_max_beta_power_floor()
            }
            pub fn denominator<AB: FullAirBuilder>(
                &self,
                builder: &AB,
                proof_idx: AB::VarMaybeExt,
                payload: [AB::VarMaybeExt; $arity],
            ) -> AB::VarExt {
                self.bus.denominator_for_proof(builder, proof_idx, payload)
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }
    };
}

per_proof_bus!(StatementVkDigestBus, STATEMENT_VK_DIGEST_SCHEMA, 9);
per_proof_bus!(StatementHashChainBus, STATEMENT_HASH_CHAIN_SCHEMA, 18);
per_proof_bus!(StatementChildFactsBus, STATEMENT_CHILD_FACTS_SCHEMA, 3);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct StatementBoundaryCols<T> {
    pub proof_idx: T,
    pub is_valid: T,
    pub is_scalar: T,
    pub is_cvd: T,
    pub is_dt_vk: T,
    pub is_vk_root: T,
    pub is_interval: T,
    pub is_interval_export: T,
    pub is_export: T,
    pub cursor: T,
    pub cursor_out: T,
    pub child_count: T,
    pub child_count_inv: T,
    pub chunk_idx: T,
    pub is_exec: T,
    pub is_first_shard: T,
    pub first_seen: T,
    pub cursor_is_zero: T,
    pub cursor_inv: T,
    pub shard_minus_one_inv: T,
    pub exec_start_pc_inv: T,
    pub core_clk_delta: T,
    pub core_clk_delta_inv: T,
    pub num_rounds: T,
    pub c_chips: T,
    pub summary_id_base: T,

    pub start_pc: T,
    pub next_pc: T,
    pub shard: T,
    pub next_shard: T,
    pub execution_shard: T,
    pub next_execution_shard: T,
    pub previous_init_addr: T,
    pub last_init_addr: T,
    pub previous_finalize_addr: T,
    pub last_finalize_addr: T,
    pub exit_code: T,
    pub start_clk: T,
    pub exit_clk: T,
    pub deferred_digest: [T; STATEMENT_PAYLOAD_ELTS],
    pub interval_chunk_flags: [T; STATEMENT_GLOBAL_CHUNKS],
    pub interval_start: [T; STATEMENT_GLOBAL_CHUNK_ELTS],
    pub interval_end: [T; STATEMENT_GLOBAL_CHUNK_ELTS],

    pub scalar_in: [T; STATEMENT_SCALAR_STATE_ELTS],
    pub scalar_out: [T; STATEMENT_SCALAR_STATE_ELTS],
    /// Linearizes the first-shard canonical x/y metadata lookups.
    pub seed_admit: T,
    pub pv_idxs: [T; STATEMENT_PV_SLOTS],
    pub pv_values: [T; STATEMENT_PV_SLOTS],

    pub digest_acc_in: [T; STATEMENT_PAYLOAD_ELTS],
    pub digest_acc_out: [T; STATEMENT_PAYLOAD_ELTS],
    pub digest_values: [T; STATEMENT_PAYLOAD_ELTS],
    pub digest_nonzero: [T; STATEMENT_PAYLOAD_ELTS],
    pub digest_nonzero_inv: [T; STATEMENT_PAYLOAD_ELTS],
    pub cvd_freeze_active: T,
    pub export_cvd: [[T; STATEMENT_PAYLOAD_ELTS]; STATEMENT_CVD_CHUNKS],

    pub child_vk_digest: [T; DIGEST_SIZE],
    pub child_vk_root: [T; DIGEST_SIZE],
    pub f_baked: T,
    pub f_thread: T,
    pub class_id: T,
    pub complete_next_shard_inv: T,
    pub statement_is_complete: T,
    pub shard_lo: T,
    pub shard_hi: T,

}

pub const NUM_STATEMENT_BOUNDARY_COLS: usize = StatementBoundaryCols::<u8>::width();
pub const NUM_STATEMENT_BOUNDARY_NARROW_COLS: usize = NUM_STATEMENT_BOUNDARY_COLS - 16;
const _: () = {
    assert!(NUM_STATEMENT_BOUNDARY_COLS == 257);
    assert!(NUM_STATEMENT_BOUNDARY_NARROW_COLS == 241);
};

#[derive(Debug, Clone)]
pub struct StatementBoundaryAir {
    pub statement_role: RecursionStatementRole,
    pub num_public_values: usize,
    pub statement_config: Vec<StatementConfigRow>,
    scalar_bus: StatementScalarChainBus,
    digest_bus: StatementDigestChainBus,
    global_interval_bus: StatementGlobalIntervalChainBus,
    child_facts_bus: StatementChildFactsBus,
    proof_values_bus: ProofShapeValuesBus,
    summary_bus: ProofShapeSummaryBus,
    vk_digest_bus: StatementVkDigestBus,
    statement_config_bus: StatementConfigBus,
    range_bus: RangeCheckerBus,
}

impl StatementBoundaryAir {
    pub fn new(
        statement_role: RecursionStatementRole,
        num_public_values: usize,
        statement_config: Vec<StatementConfigRow>,
    ) -> Self {
        Self {
            statement_role,
            num_public_values,
            statement_config,
            scalar_bus: StatementScalarChainBus::new(),
            digest_bus: StatementDigestChainBus::new(),
            global_interval_bus: StatementGlobalIntervalChainBus::new(),
            child_facts_bus: StatementChildFactsBus::new(),
            proof_values_bus: ProofShapeValuesBus::new(),
            summary_bus: ProofShapeSummaryBus::new(),
            vk_digest_bus: StatementVkDigestBus::new(),
            statement_config_bus: StatementConfigBus::new(),
            range_bus: RangeCheckerBus::new(),
        }
    }

    const fn core_role(&self) -> bool {
        matches!(self.statement_role, RecursionStatementRole::Lift)
    }

    const fn active_pv_slots(&self) -> usize {
        STATEMENT_PV_SLOTS
    }
}

impl BaseAir<F> for StatementBoundaryAir {
    fn width(&self) -> usize {
        if self.core_role() {
            NUM_STATEMENT_BOUNDARY_COLS
        } else {
            NUM_STATEMENT_BOUNDARY_NARROW_COLS
        }
    }
}

impl<AB: FullAirBuilder> FullAir<AB> for StatementBoundaryAir {
    fn width(&self) -> usize {
        BaseAir::<F>::width(self)
    }

    fn required_max_beta_power(&self) -> usize {
        [
            self.scalar_bus.required_max_beta_power_floor(),
            self.digest_bus.required_max_beta_power_floor(),
            self.global_interval_bus.required_max_beta_power_floor(),
            self.child_facts_bus.required_max_beta_power_floor(),
            self.proof_values_bus.required_max_beta_power_floor(),
            self.summary_bus.required_max_beta_power_floor(),
            self.vk_digest_bus.required_max_beta_power_floor(),
            self.statement_config_bus.required_max_beta_power_floor(),
            self.range_bus.required_max_beta_power_floor(),
        ]
        .into_iter()
        .max()
        .unwrap()
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        (0..BaseAir::<F>::width(self)).map(PairCol::Main).collect()
    }

    fn precompute_lc(&self, builder: &mut AB) {
        let denominators = {
            let main = builder.main();
            let expanded = expand_statement_row::<AB>(main, self.core_role());
            let local: &StatementBoundaryCols<AB::VarMaybeExt> = expanded.as_slice().borrow();
            let mut denominators = Vec::new();
            denominators.push(self.scalar_bus.denominator(
                builder,
                local.cursor.clone(),
                local.scalar_in.clone(),
            ));
            denominators.push(self.scalar_bus.denominator(
                builder,
                local.cursor_out.clone(),
                local.scalar_out.clone(),
            ));
            denominators.push(self.digest_bus.denominator(
                builder,
                local.chunk_idx.clone(),
                local.cursor.clone(),
                local.digest_acc_in.clone(),
            ));
            denominators.push(self.digest_bus.denominator(
                builder,
                local.chunk_idx.clone(),
                local.cursor_out.clone(),
                local.digest_acc_out.clone(),
            ));
            for chunk in 0..STATEMENT_CVD_CHUNKS {
                denominators.push(self.digest_bus.denominator(
                    builder,
                    local.is_export.clone() * c::<AB>(chunk),
                    local.child_count.clone(),
                    local.export_cvd[chunk].clone(),
                ));
                denominators.push(self.digest_bus.denominator(
                    builder,
                    local.is_export.clone() * c::<AB>(chunk),
                    local.cursor_out.clone(),
                    core::array::from_fn(|_| AB::zero_maybe()),
                ));
            }
            denominators.push(self.global_interval_bus.denominator(
                builder,
                local.chunk_idx.clone(),
                local.cursor.clone(),
                local.interval_start.clone(),
            ));
            denominators.push(self.global_interval_bus.denominator(
                builder,
                local.chunk_idx.clone(),
                local.cursor_out.clone(),
                local.interval_end.clone(),
            ));
            denominators.push(self.global_interval_bus.denominator(
                builder,
                local.chunk_idx.clone(),
                AB::zero_maybe(),
                local.interval_start.clone(),
            ));
            denominators.push(self.global_interval_bus.denominator(
                builder,
                local.chunk_idx.clone(),
                local.child_count.clone(),
                local.interval_end.clone(),
            ));
            let facts = [local.cursor.clone(), local.is_exec.clone(), local.is_first_shard.clone()];
            denominators.push(self.child_facts_bus.denominator(
                builder,
                local.proof_idx.clone(),
                facts.clone(),
            ));
            denominators.push(self.child_facts_bus.denominator(
                builder,
                local.proof_idx.clone(),
                facts,
            ));
            for slot in 0..self.active_pv_slots() {
                let namespace = if slot >= 2 * STATEMENT_GLOBAL_CHUNK_ELTS {
                    local.seed_admit.clone() * c::<AB>(PROOF_SHAPE_NAMESPACE_VK_META)
                } else {
                    AB::zero_maybe()
                };
                denominators.push(self.proof_values_bus.denominator(
                    builder,
                    local.proof_idx.clone(),
                    namespace,
                    local.pv_idxs[slot].clone(),
                    local.pv_values[slot].clone(),
                ));
            }
            if self.core_role() {
                denominators.push(self.proof_values_bus.denominator(
                    builder,
                    local.proof_idx.clone(),
                    c::<AB>(PROOF_SHAPE_NAMESPACE_VK_META),
                    c::<AB>(PROOF_SHAPE_VK_META_BOUNDARY_KIND),
                    local.shard_minus_one_inv.clone(),
                ));
            }
            denominators.push(self.summary_bus.denominator(
                builder,
                local.proof_idx.clone(),
                local.num_rounds.clone(),
                local.c_chips.clone(),
                c::<AB>(self.num_public_values),
                local.summary_id_base.clone(),
            ));
            let mut vk_payload = core::array::from_fn(|_| AB::zero_maybe());
            vk_payload[0] = c::<AB>(STATEMENT_HASH_KIND_VK_DIGEST);
            vk_payload[1..].clone_from_slice(&local.child_vk_digest);
            denominators.push(self.vk_digest_bus.denominator(
                builder,
                local.proof_idx.clone(),
                vk_payload,
            ));
            denominators.push(self.statement_config_bus.denominator(
                builder,
                local.class_id.clone(),
                local.child_vk_digest.clone(),
            ));
            if self.core_role() {
                denominators.push(self.range_bus.denominator(
                    builder,
                    RangeCheckerBusMessage { value: local.shard_lo.clone(), max_bits: c::<AB>(8) },
                ));
                denominators.push(self.range_bus.denominator(
                    builder,
                    RangeCheckerBusMessage { value: local.shard_hi.clone(), max_bits: c::<AB>(8) },
                ));
            }
            denominators
        };
        for denominator in denominators {
            builder.retain_precomputed(denominator);
        }
    }

    fn eval(&self, builder: &mut AB) {
        let reserved = builder.reserved_poly();
        let local_row = reserved.row_slice(0);
        let expanded = expand_statement_row::<AB>(local_row.deref(), self.core_role());
        let local: &StatementBoundaryCols<AB::VarMaybeExt> = expanded.as_slice().borrow();
        let one = AB::one_maybe();
        let core = if self.core_role() { one.clone() } else { AB::zero_maybe() };
        let native = one.clone() - core.clone();

        for value in [
            local.is_valid.clone(),
            local.is_scalar.clone(),
            local.is_cvd.clone(),
            local.is_dt_vk.clone(),
            local.is_vk_root.clone(),
            local.is_interval.clone(),
            local.is_interval_export.clone(),
            local.is_export.clone(),
            local.is_exec.clone(),
            local.is_first_shard.clone(),
            local.first_seen.clone(),
            local.cursor_is_zero.clone(),
            local.f_baked.clone(),
            local.f_thread.clone(),
        ] {
            assert_bool(builder, value);
        }
        builder.assert_eq(
            local.is_scalar.clone() +
                local.is_cvd.clone() +
                local.is_dt_vk.clone() +
                local.is_vk_root.clone() +
                local.is_interval.clone() +
                local.is_interval_export.clone() +
                local.is_export.clone(),
            local.is_valid.clone(),
        );
        let child_row = local.is_scalar.clone() +
            local.is_cvd.clone() +
            local.is_dt_vk.clone() +
            local.is_vk_root.clone() +
            local.is_interval.clone();
        builder.assert_zero(child_row * (local.cursor.clone() - local.proof_idx.clone()));

        constrain_scalar(builder, local, core.clone());
        constrain_cvd(builder, local);
        constrain_interval(builder, local, self.statement_role, core.clone(), native.clone());
        constrain_pv_slots(builder, local, core.clone(), native.clone());
        constrain_vk(builder, local, self.statement_role, core, native);
        constrain_export(builder, local, self.statement_role);
    }

    fn lookup(&self, builder: &mut AB) {
        let reserved = builder.reserved_poly();
        let local_row = reserved.row_slice(0);
        let expanded = expand_statement_row::<AB>(local_row.deref(), self.core_role());
        let local: &StatementBoundaryCols<AB::VarMaybeExt> = expanded.as_slice().borrow();
        let core = if self.core_role() { AB::one_maybe() } else { AB::zero_maybe() };
        let native = AB::one_maybe() - core.clone();

        let scalar_mult = local.is_scalar.clone() + local.is_export.clone();
        builder.recv(scalar_mult.clone());
        builder.send(scalar_mult);
        builder.recv(local.is_cvd.clone());
        builder.send(local.is_cvd.clone());
        for _ in 0..STATEMENT_CVD_CHUNKS {
            builder.recv(local.is_export.clone());
            builder.send(local.is_export.clone());
        }
        builder.recv(local.is_interval.clone());
        builder.send(local.is_interval.clone());
        builder.send(local.is_interval_export.clone());
        builder.recv(local.is_interval_export.clone());
        builder.send(c::<AB>(9) * local.is_scalar.clone());
        builder.recv(
            local.is_cvd.clone() +
                local.is_dt_vk.clone() +
                local.is_vk_root.clone() +
                local.is_interval.clone(),
        );

        let pv_mults = pv_multiplicities::<AB>(local, core.clone(), native);
        for multiplicity in pv_mults.into_iter().take(self.active_pv_slots()) {
            builder.recv(multiplicity);
        }
        if self.core_role() {
            // Keep the multiplicity linear while excluding the three root/export rows.
            // Every child interval row carries the same authenticated boundary kind.
            builder.recv(boundary_kind_lookup_multiplicity(local));
        }
        builder.recv(local.is_vk_root.clone());
        builder.recv(local.is_vk_root.clone());
        builder.recv(local.f_baked.clone());
        if self.core_role() {
            builder.recv(core.clone() * local.is_scalar.clone());
            builder.recv(core * local.is_scalar.clone());
        }
    }
}

impl MachineAir<F> for StatementBoundaryAir {
    type Record = RecursionRecord;
    type Program = RecursionNativeProgram<F>;

    fn name(&self) -> String {
        "StatementBoundary".to_string()
    }

    fn generate_dependencies(&self, _: &Self::Record, _: &mut Self::Record) {}

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        Some(
            ((7 + STATEMENT_GLOBAL_CHUNKS) * input.proof_records.len() +
                1 +
                STATEMENT_GLOBAL_CHUNKS)
                .max(1)
                .next_power_of_two(),
        )
    }

    fn generate_trace(&self, input: &Self::Record, _: &mut Self::Record) -> CompressedMatrix<F> {
        let rows = statement_rows_cached(input, self.statement_role, &self.statement_config);
        let height = rows.len().max(1).next_power_of_two();
        let values = if self.core_role() {
            rows.iter().flatten().copied().collect()
        } else {
            let keep = statement_narrow_projection_columns();
            rows.iter().flat_map(|row| keep.iter().map(|&column| row[column])).collect()
        };
        CompressedMatrix::new(
            RowMajorMatrix::new(values, BaseAir::<F>::width(self)),
            PaddingRow::Zero { width: BaseAir::<F>::width(self) },
            height,
        )
    }

    fn included(&self, record: &Self::Record) -> bool {
        !record.proof_records.is_empty()
    }
    fn local_only(&self) -> bool {
        true
    }
}

fn statement_narrow_projection_columns() -> Vec<usize> {
    let mut keep = vec![true; NUM_STATEMENT_BOUNDARY_COLS];
    for (start, len) in [
        (core::mem::offset_of!(StatementBoundaryCols<u8>, core_clk_delta), 2),
        (core::mem::offset_of!(StatementBoundaryCols<u8>, exit_code), 11),
        (core::mem::offset_of!(StatementBoundaryCols<u8>, seed_admit), 1),
        (core::mem::offset_of!(StatementBoundaryCols<u8>, shard_lo), 2),
    ] {
        keep[start..start + len].fill(false);
    }
    let columns = keep
        .into_iter()
        .enumerate()
        .filter_map(|(column, keep)| keep.then_some(column))
        .collect::<Vec<_>>();
    debug_assert_eq!(columns.len(), NUM_STATEMENT_BOUNDARY_NARROW_COLS);
    columns
}

fn expand_statement_row<AB: FullAirBuilder>(
    row: &[AB::VarMaybeExt],
    core_role: bool,
) -> Vec<AB::VarMaybeExt> {
    if core_role {
        debug_assert_eq!(row.len(), NUM_STATEMENT_BOUNDARY_COLS);
        return row.to_vec();
    }
    debug_assert_eq!(row.len(), NUM_STATEMENT_BOUNDARY_NARROW_COLS);
    let mut expanded = vec![AB::zero_maybe(); NUM_STATEMENT_BOUNDARY_COLS];
    for (value, column) in row.iter().cloned().zip(statement_narrow_projection_columns()) {
        expanded[column] = value;
    }
    expanded
}

pub type StatementBoundaryRow = Vec<F>;

pub(crate) fn statement_rows_cached(
    record: &RecursionRecord,
    role: RecursionStatementRole,
    statement_config: &[StatementConfigRow],
) -> Arc<[StatementBoundaryRow]> {
    let authority = statement_config.iter().fold(role as u64, |acc, row| {
        row.digest.iter().fold(acc ^ row.class_id as u64, |acc, value| {
            acc.wrapping_mul(0x100000001b3) ^ u64::from(value.as_canonical_u32())
        })
    });
    let (installed, rows) = record
        .tracegen_artifacts
        .statement
        .get_or_init(|| (authority, Arc::from(boundary_rows(record, role, statement_config))));
    assert_eq!(*installed, authority, "one tracegen workspace used two D11 statement authorities");
    Arc::clone(rows)
}

pub fn annotate_child_statement_publications(proof: &mut crate::system_dt::RecursionProofRecord) {
    proof.proof_shape.vk_meta_send_mults.fill(0);
    if proof.proof_shape.public_value_send_mults.len() < proof.proof_shape.public_values.len() {
        proof.proof_shape.public_value_send_mults.resize(proof.proof_shape.public_values.len(), 0);
    }
    let core = proof.proof_shape.role_id == WHIR_ROLE_CORE;
    let offsets: Vec<usize> = if core {
        vec![
            CORE_PV_START_PC,
            CORE_PV_NEXT_PC,
            CORE_PV_SHARD,
            CORE_PV_EXECUTION_SHARD,
            CORE_PV_PREVIOUS_INIT_ADDR,
            CORE_PV_LAST_INIT_ADDR,
            CORE_PV_PREVIOUS_FINALIZE_ADDR,
            CORE_PV_LAST_FINALIZE_ADDR,
            CORE_PV_EXIT_CODE,
            CORE_PV_START_CLK,
            CORE_PV_EXIT_CLK,
        ]
    } else {
        vec![
            NATIVE_PV_START_PC,
            NATIVE_PV_NEXT_PC,
            NATIVE_PV_START_SHARD,
            NATIVE_PV_START_EXECUTION_SHARD,
            NATIVE_PV_PREVIOUS_INIT_ADDR,
            NATIVE_PV_LAST_INIT_ADDR,
            NATIVE_PV_PREVIOUS_FINALIZE_ADDR,
            NATIVE_PV_LAST_FINALIZE_ADDR,
            NATIVE_PV_NEXT_SHARD,
            NATIVE_PV_NEXT_EXECUTION_SHARD,
            NATIVE_PV_CONTAINS_EXECUTION_SHARD,
        ]
    };
    for offset in offsets {
        proof.proof_shape.public_value_send_mults[offset] =
            proof.proof_shape.public_value_send_mults[offset].saturating_add(1);
    }
    let cvd_start = if core {
        CORE_PV_COMMITTED_VALUE_DIGEST_START
    } else {
        NATIVE_PV_COMMITTED_VALUE_DIGEST_START
    };
    for offset in cvd_start..cvd_start + 32 {
        proof.proof_shape.public_value_send_mults[offset] =
            proof.proof_shape.public_value_send_mults[offset].saturating_add(1);
    }
    if core {
        for offset in CORE_PV_DEFERRED_PROOFS_DIGEST_START..CORE_PV_DEFERRED_PROOFS_DIGEST_START + 8
        {
            proof.proof_shape.public_value_send_mults[offset] =
                proof.proof_shape.public_value_send_mults[offset].saturating_add(1);
        }
        for offset in
            CORE_PV_GLOBAL_INTERVAL_START..CORE_PV_GLOBAL_INTERVAL_END + STATEMENT_GLOBAL_STATE_ELTS
        {
            proof.proof_shape.public_value_send_mults[offset] =
                proof.proof_shape.public_value_send_mults[offset].saturating_add(1);
        }
    } else {
        for offset in NATIVE_PV_DT_VK_DIGEST_START..NATIVE_PV_VK_ROOT_START + 8 {
            proof.proof_shape.public_value_send_mults[offset] =
                proof.proof_shape.public_value_send_mults[offset].saturating_add(1);
        }
        for offset in NATIVE_PV_GLOBAL_INTERVAL_START..
            NATIVE_PV_GLOBAL_INTERVAL_END + STATEMENT_GLOBAL_STATE_ELTS
        {
            proof.proof_shape.public_value_send_mults[offset] =
                proof.proof_shape.public_value_send_mults[offset].saturating_add(1);
        }
        for offset in native_extra_offsets() {
            proof.proof_shape.public_value_send_mults[offset] =
                proof.proof_shape.public_value_send_mults[offset].saturating_add(1);
        }
    }
    proof.proof_shape.vk_meta_send_mults.iter_mut().for_each(|mult| {
        *mult = mult.saturating_add(1);
    });
    if proof.proof_shape.publish_whir_inputs && core {
        proof.proof_shape.vk_meta_send_mults[PROOF_SHAPE_VK_META_BOUNDARY_KIND] = proof
            .proof_shape
            .vk_meta_send_mults[PROOF_SHAPE_VK_META_BOUNDARY_KIND]
            .saturating_add(STATEMENT_GLOBAL_CHUNKS as u32);
        if proof.proof_shape.public_values.get(CORE_PV_SHARD).copied() == Some(F::one()) {
            for mult in &mut proof.proof_shape.vk_meta_send_mults
                [PROOF_SHAPE_VK_META_BOUNDARY_X_BASE..PROOF_SHAPE_VK_META_BOUNDARY_X_BASE + 22]
            {
                *mult = mult.saturating_add(1);
            }
        }
    }
}

pub fn annotate_statement_publications(record: &mut RecursionRecord) {
    record.mark_semantic_mutation();
    record.proof_records.iter_mut().for_each(annotate_child_statement_publications);
}

pub type StatementBusResidualReport = BTreeMap<&'static str, BTreeMap<Vec<u32>, i64>>;

pub fn statement_part_b_bus_residual_report(
    _record: &RecursionRecord,
    _program: &RecursionNativeProgram<F>,
) -> StatementBusResidualReport {
    StatementBusResidualReport::new()
}

fn constrain_scalar<AB: FullAirBuilder>(
    builder: &mut AB,
    local: &StatementBoundaryCols<AB::VarMaybeExt>,
    core: AB::VarMaybeExt,
) {
    let active = local.is_scalar.clone();
    let one = AB::one_maybe();
    let non_exec = one.clone() - local.is_exec.clone();
    builder.assert_zero(
        active.clone() *
            (local.cursor.clone() * local.cursor_inv.clone() -
                (one.clone() - local.cursor_is_zero.clone())),
    );
    builder.assert_zero(active.clone() * local.cursor_is_zero.clone() * local.cursor.clone());
    builder.assert_zero(
        active.clone() * (local.cursor_out.clone() - local.cursor.clone() - one.clone()),
    );
    builder.assert_zero(
        active.clone() * (local.scalar_in[SCALAR_PC].clone() - local.start_pc.clone()),
    );
    builder.assert_zero(
        active.clone() * (local.scalar_out[SCALAR_PC].clone() - local.next_pc.clone()),
    );
    builder.assert_zero(
        active.clone() * (local.scalar_in[SCALAR_SHARD].clone() - local.shard.clone()),
    );
    builder.assert_zero(
        active.clone() * (local.scalar_out[SCALAR_SHARD].clone() - local.next_shard.clone()),
    );
    builder.assert_zero(
        active.clone() *
            core.clone() *
            (local.next_shard.clone() - local.shard.clone() - one.clone()),
    );
    builder.assert_zero(
        active.clone() *
            (local.scalar_in[SCALAR_INIT_ADDR].clone() - local.previous_init_addr.clone()),
    );
    builder.assert_zero(
        active.clone() *
            (local.scalar_out[SCALAR_INIT_ADDR].clone() - local.last_init_addr.clone()),
    );
    builder.assert_zero(
        active.clone() *
            (local.scalar_in[SCALAR_FIN_ADDR].clone() - local.previous_finalize_addr.clone()),
    );
    builder.assert_zero(
        active.clone() *
            (local.scalar_out[SCALAR_FIN_ADDR].clone() - local.last_finalize_addr.clone()),
    );
    builder.assert_zero(active.clone() * core.clone() * local.exit_code.clone());
    for value in &local.deferred_digest {
        builder.assert_zero(active.clone() * core.clone() * value.clone());
    }
    builder.assert_zero(
        active.clone() *
            core.clone() *
            (local.core_clk_delta.clone() - local.exit_clk.clone() + local.start_clk.clone()),
    );
    builder.assert_zero(
        active.clone() *
            core.clone() *
            (local.core_clk_delta.clone() * local.core_clk_delta_inv.clone() -
                local.is_exec.clone()),
    );
    builder.assert_zero(
        active.clone() * core.clone() * non_exec.clone() * local.core_clk_delta.clone(),
    );
    builder.assert_zero(
        active.clone() * non_exec.clone() * (local.start_pc.clone() - local.next_pc.clone()),
    );
    builder.assert_zero(
        active.clone() *
            (local.start_pc.clone() * local.exec_start_pc_inv.clone() - local.is_exec.clone()),
    );
    builder.assert_zero(active.clone() * non_exec.clone() * local.exec_start_pc_inv.clone());
    builder.assert_zero(active.clone() * non_exec.clone() * local.is_first_shard.clone());
    builder.assert_zero(
        active.clone() * local.is_first_shard.clone() * (local.shard.clone() - one.clone()),
    );
    builder.assert_zero(
        active.clone() *
            ((local.shard.clone() - one.clone()) * local.shard_minus_one_inv.clone() -
                (one.clone() - local.is_first_shard.clone())),
    );
    builder.assert_zero(
        active.clone() * local.is_first_shard.clone() * local.previous_init_addr.clone(),
    );
    builder.assert_zero(
        active.clone() * local.is_first_shard.clone() * local.previous_finalize_addr.clone(),
    );
    builder.assert_zero(
        active.clone() *
            (local.first_seen.clone() -
                local.is_exec.clone() *
                    (one.clone() - local.scalar_in[SCALAR_EXEC_SEEN].clone())),
    );
    builder.assert_zero(
        active.clone() *
            (local.is_exec.clone() - local.first_seen.clone()) *
            (local.execution_shard.clone() - local.scalar_in[SCALAR_EXEC].clone()),
    );
    builder.assert_zero(
        active.clone() *
            (local.scalar_out[SCALAR_EXEC_SEEN].clone() -
                local.scalar_in[SCALAR_EXEC_SEEN].clone() -
                local.first_seen.clone()),
    );
    builder.assert_zero(
        active.clone() *
            core *
            (local.next_execution_shard.clone() - local.execution_shard.clone() - one.clone()),
    );
    builder.assert_zero(
        active.clone() *
            (local.scalar_out[SCALAR_EXEC].clone() -
                local.scalar_in[SCALAR_EXEC].clone() -
                local.is_exec.clone() *
                    (local.next_execution_shard.clone() -
                        local.scalar_in[SCALAR_EXEC].clone())),
    );
    builder.assert_zero(
        active.clone() *
            local.first_seen.clone() *
            (local.scalar_out[SCALAR_START_EXEC_OUT].clone() - local.execution_shard.clone()),
    );
    builder.assert_zero(
        active.clone() *
            (one - local.first_seen.clone()) *
            (local.scalar_out[SCALAR_START_EXEC_OUT].clone() -
                local.scalar_in[SCALAR_START_EXEC_OUT].clone()),
    );
    for idx in
        [SCALAR_START_PC_OUT, SCALAR_START_SHARD_OUT, SCALAR_PREV_INIT_OUT, SCALAR_PREV_FIN_OUT]
    {
        builder.assert_zero(
            active.clone() * (local.scalar_out[idx].clone() - local.scalar_in[idx].clone()),
        );
    }
}

fn constrain_cvd<AB: FullAirBuilder>(
    builder: &mut AB,
    local: &StatementBoundaryCols<AB::VarMaybeExt>,
) {
    let active = local.is_cvd.clone();
    let one = AB::one_maybe();
    builder.assert_zero(
        active.clone() * (local.cursor_out.clone() - local.cursor.clone() - one.clone()),
    );
    builder.assert_zero(
        active.clone() *
            (local.cursor.clone() * local.cursor_inv.clone() -
                (one.clone() - local.cursor_is_zero.clone())),
    );
    builder.assert_zero(active.clone() * local.cursor_is_zero.clone() * local.cursor.clone());
    builder.assert_eq(
        local.cvd_freeze_active.clone(),
        active.clone() * (one.clone() - local.cursor_is_zero.clone()),
    );
    for idx in 0..STATEMENT_PAYLOAD_ELTS {
        assert_bool(builder, local.digest_nonzero[idx].clone());
        builder.assert_zero(
            active.clone() *
                (local.digest_acc_in[idx].clone() * local.digest_nonzero_inv[idx].clone() -
                    local.digest_nonzero[idx].clone()),
        );
        builder.assert_zero(
            active.clone() *
                (one.clone() - local.digest_nonzero[idx].clone()) *
                local.digest_acc_in[idx].clone(),
        );
        builder.assert_zero(
            local.cvd_freeze_active.clone() *
                local.digest_nonzero[idx].clone() *
                (local.digest_values[idx].clone() - local.digest_acc_in[idx].clone()),
        );
        builder.assert_zero(
            local.cvd_freeze_active.clone() *
                (one.clone() - local.is_exec.clone()) *
                (local.digest_values[idx].clone() - local.digest_acc_in[idx].clone()),
        );
        builder.assert_zero(
            active.clone() * (local.digest_acc_out[idx].clone() - local.digest_values[idx].clone()),
        );
    }
}

fn constrain_interval<AB: FullAirBuilder>(
    builder: &mut AB,
    local: &StatementBoundaryCols<AB::VarMaybeExt>,
    role: RecursionStatementRole,
    core: AB::VarMaybeExt,
    native: AB::VarMaybeExt,
) {
    let one = AB::one_maybe();
    let active = local.is_interval.clone() + local.is_interval_export.clone();
    let mut flag_sum = AB::zero_maybe();
    let mut chunk = AB::zero_maybe();
    for (idx, flag) in local.interval_chunk_flags.iter().enumerate() {
        assert_bool(builder, flag.clone());
        flag_sum += flag.clone();
        chunk += flag.clone() * c::<AB>(idx);
    }
    builder.assert_eq(flag_sum, active.clone());
    builder.assert_zero(active.clone() * (local.chunk_idx.clone() - chunk));
    builder.assert_zero(
        local.is_interval.clone() * (local.cursor_out.clone() - local.cursor.clone() - one.clone()),
    );
    builder.assert_zero(
        local.is_interval.clone() * local.cursor_is_zero.clone() * local.cursor.clone(),
    );
    builder.assert_zero(
        local.is_interval.clone() *
            (local.cursor.clone() * local.cursor_inv.clone() -
                (one.clone() - local.cursor_is_zero.clone())),
    );
    builder.assert_zero(local.is_interval_export.clone() * local.cursor.clone());
    builder.assert_zero(
        local.is_interval_export.clone() * (local.cursor_out.clone() - local.child_count.clone()),
    );

    for lane in 0..STATEMENT_GLOBAL_CHUNK_ELTS {
        let expected_live = active.clone();
        builder.assert_zero(
            (one.clone() - expected_live.clone()) * local.interval_start[lane].clone(),
        );
        builder.assert_zero((one.clone() - expected_live) * local.interval_end[lane].clone());

        let mut parent_start = AB::zero_maybe();
        let mut parent_end = AB::zero_maybe();
        for chunk_idx in 0..STATEMENT_GLOBAL_CHUNKS {
            let flat_idx = chunk_idx * STATEMENT_GLOBAL_CHUNK_ELTS + lane;
            if flat_idx < STATEMENT_GLOBAL_STATE_ELTS {
                let public_start: AB::VarMaybeExt =
                    builder.public()[NATIVE_PV_GLOBAL_INTERVAL_START + flat_idx].clone().into();
                let public_end: AB::VarMaybeExt =
                    builder.public()[NATIVE_PV_GLOBAL_INTERVAL_END + flat_idx].clone().into();
                parent_start += local.interval_chunk_flags[chunk_idx].clone() * public_start;
                parent_end += local.interval_chunk_flags[chunk_idx].clone() * public_end;
            }
        }
        builder.assert_zero(
            local.is_interval_export.clone() * (local.interval_start[lane].clone() - parent_start),
        );
        builder.assert_zero(
            local.is_interval_export.clone() * (local.interval_end[lane].clone() - parent_end),
        );
        if role == RecursionStatementRole::RootShrink {
            let complete: AB::VarMaybeExt = builder.public()[NATIVE_PV_IS_COMPLETE].clone().into();
            let expected_end = local
                .interval_chunk_flags
                .iter()
                .enumerate()
                .filter(|(chunk_idx, _)| {
                    chunk_idx * STATEMENT_GLOBAL_CHUNK_ELTS + lane == 11
                })
                .fold(AB::zero_maybe(), |acc, (_, value)| acc + value.clone());
            builder.assert_zero(
                local.is_interval_export.clone() *
                    complete *
                    (local.interval_end[lane].clone() - expected_end),
            );
        }

        let offset = local.chunk_idx.clone() * c::<AB>(STATEMENT_GLOBAL_CHUNK_ELTS) + c::<AB>(lane);
        let child_start_base = core.clone() * c::<AB>(CORE_PV_GLOBAL_INTERVAL_START) +
            native.clone() * c::<AB>(NATIVE_PV_GLOBAL_INTERVAL_START);
        let child_end_base = core.clone() * c::<AB>(CORE_PV_GLOBAL_INTERVAL_END) +
            native.clone() * c::<AB>(NATIVE_PV_GLOBAL_INTERVAL_END);
        builder.assert_zero(
            local.is_interval.clone() *
                (local.pv_idxs[lane].clone() - child_start_base - offset.clone()),
        );
        builder.assert_zero(
            local.is_interval.clone() *
                (local.pv_values[lane].clone() - local.interval_start[lane].clone()),
        );
        builder.assert_zero(
            local.is_interval.clone() *
                (local.pv_idxs[STATEMENT_GLOBAL_CHUNK_ELTS + lane].clone() -
                    child_end_base -
                    offset.clone()),
        );
        builder.assert_zero(
            local.is_interval.clone() *
                (local.pv_values[STATEMENT_GLOBAL_CHUNK_ELTS + lane].clone() -
                    local.interval_end[lane].clone()),
        );

    }

}

fn constrain_pv_slots<AB: FullAirBuilder>(
    builder: &mut AB,
    local: &StatementBoundaryCols<AB::VarMaybeExt>,
    core: AB::VarMaybeExt,
    native: AB::VarMaybeExt,
) {
    let scalar_values = [
        local.start_pc.clone(),
        local.next_pc.clone(),
        local.shard.clone(),
        local.execution_shard.clone(),
        local.previous_init_addr.clone(),
        local.last_init_addr.clone(),
        local.previous_finalize_addr.clone(),
        local.last_finalize_addr.clone(),
        local.next_shard.clone(),
        local.next_execution_shard.clone(),
        local.is_exec.clone(),
        local.exit_code.clone(),
        local.start_clk.clone(),
        local.exit_clk.clone(),
    ];
    let core_offsets = [
        CORE_PV_START_PC,
        CORE_PV_NEXT_PC,
        CORE_PV_SHARD,
        CORE_PV_EXECUTION_SHARD,
        CORE_PV_PREVIOUS_INIT_ADDR,
        CORE_PV_LAST_INIT_ADDR,
        CORE_PV_PREVIOUS_FINALIZE_ADDR,
        CORE_PV_LAST_FINALIZE_ADDR,
        0,
        0,
        0,
        CORE_PV_EXIT_CODE,
        CORE_PV_START_CLK,
        CORE_PV_EXIT_CLK,
    ];
    let native_offsets = [
        NATIVE_PV_START_PC,
        NATIVE_PV_NEXT_PC,
        NATIVE_PV_START_SHARD,
        NATIVE_PV_START_EXECUTION_SHARD,
        NATIVE_PV_PREVIOUS_INIT_ADDR,
        NATIVE_PV_LAST_INIT_ADDR,
        NATIVE_PV_PREVIOUS_FINALIZE_ADDR,
        NATIVE_PV_LAST_FINALIZE_ADDR,
        NATIVE_PV_NEXT_SHARD,
        NATIVE_PV_NEXT_EXECUTION_SHARD,
        NATIVE_PV_CONTAINS_EXECUTION_SHARD,
        0,
        0,
        0,
    ];
    assert_bool(builder, local.seed_admit.clone());
    builder.assert_eq(
        local.seed_admit.clone(),
        core.clone() *
            local.is_first_shard.clone() *
            (local.interval_chunk_flags[0].clone() + local.interval_chunk_flags[1].clone()),
    );
    let boundary_kind = local.shard_minus_one_inv.clone();
    let kind_lookup_active = core.clone() * local.is_interval.clone();
    builder.assert_zero(
        kind_lookup_active *
            boundary_kind.clone() *
            (boundary_kind.clone() - AB::one_maybe()),
    );
    let seed_z_active = core.clone() *
        local.is_first_shard.clone() *
        local.interval_chunk_flags[2].clone();
    builder.assert_zero(
        seed_z_active.clone() *
            (local.interval_start[0].clone() - boundary_kind.clone()),
    );
    for lane in 1..STATEMENT_GLOBAL_CHUNK_ELTS {
        builder.assert_zero(seed_z_active.clone() * local.interval_start[lane].clone());
    }
    for slot in 0..STATEMENT_PV_SLOTS {
        let mut expected_idx = AB::zero_maybe();
        let mut expected_value = AB::zero_maybe();
        if slot < scalar_values.len() {
            let scalar_core = if slot <= 7 || (11..=13).contains(&slot) {
                core.clone()
            } else {
                AB::zero_maybe()
            };
            let scalar_native = if slot <= 10 { native.clone() } else { AB::zero_maybe() };
            expected_idx += local.is_scalar.clone() *
                (scalar_core.clone() * c::<AB>(core_offsets[slot]) +
                    scalar_native.clone() * c::<AB>(native_offsets[slot]));
            expected_value += local.is_scalar.clone() *
                (scalar_core + scalar_native) *
                scalar_values[slot].clone();
        } else {
            let deferred_idx = slot - scalar_values.len();
            if deferred_idx < STATEMENT_DEFERRED_DIGEST_ELTS {
                expected_idx += local.is_scalar.clone() *
                    core.clone() *
                    c::<AB>(CORE_PV_DEFERRED_PROOFS_DIGEST_START + deferred_idx);
                expected_value += local.is_scalar.clone() *
                    core.clone() *
                    local.deferred_digest[deferred_idx].clone();
            }
        }
        if (11..11 + STATEMENT_NATIVE_SCALAR_EXTRA_ELTS).contains(&slot) {
            let extra = slot - 11;
            expected_idx +=
                local.is_scalar.clone() * native.clone() * c::<AB>(native_extra_offsets()[extra]);
        }
        if slot < STATEMENT_PAYLOAD_ELTS {
            expected_idx += local.is_cvd.clone() *
                (local.chunk_idx.clone() * c::<AB>(STATEMENT_PAYLOAD_ELTS) + c::<AB>(slot));
            expected_value += local.is_cvd.clone() * local.digest_values[slot].clone();
            expected_idx += local.is_dt_vk.clone() *
                native.clone() *
                c::<AB>(NATIVE_PV_DT_VK_DIGEST_START + slot);
            expected_value +=
                local.is_dt_vk.clone() * native.clone() * local.pv_values[slot].clone();
            expected_idx +=
                local.is_vk_root.clone() * native.clone() * c::<AB>(NATIVE_PV_VK_ROOT_START + slot);
            expected_value +=
                local.is_vk_root.clone() * native.clone() * local.child_vk_root[slot].clone();
        }
        if (8..12).contains(&slot) {
            let extra = STATEMENT_NATIVE_SCALAR_EXTRA_ELTS + slot - 8;
            let active = local.is_dt_vk.clone() * native.clone();
            expected_idx += active.clone() * c::<AB>(native_extra_offsets()[extra]);
            if slot == 10 {
                let value = local.pv_values[slot].clone();
                expected_value += active.clone() * value.clone();
                builder.assert_zero(active * value.clone() * (value - AB::one_maybe()));
            }
        }
        if slot < 2 * STATEMENT_GLOBAL_CHUNK_ELTS {
            let lane = slot % STATEMENT_GLOBAL_CHUNK_ELTS;
            let base = if slot < STATEMENT_GLOBAL_CHUNK_ELTS {
                core.clone() * c::<AB>(CORE_PV_GLOBAL_INTERVAL_START) +
                    native.clone() * c::<AB>(NATIVE_PV_GLOBAL_INTERVAL_START)
            } else {
                core.clone() * c::<AB>(CORE_PV_GLOBAL_INTERVAL_END) +
                    native.clone() * c::<AB>(NATIVE_PV_GLOBAL_INTERVAL_END)
            };
            let live = local.is_interval.clone();
            expected_idx += live.clone() *
                (base +
                    local.chunk_idx.clone() * c::<AB>(STATEMENT_GLOBAL_CHUNK_ELTS) +
                    c::<AB>(lane));
            expected_value += live *
                if slot < STATEMENT_GLOBAL_CHUNK_ELTS {
                    local.interval_start[lane].clone()
                } else {
                    local.interval_end[lane].clone()
                };
        }
        if (2 * STATEMENT_GLOBAL_CHUNK_ELTS..3 * STATEMENT_GLOBAL_CHUNK_ELTS).contains(&slot) {
            let lane = slot - 2 * STATEMENT_GLOBAL_CHUNK_ELTS;
            let seed_carrier = local.seed_admit.clone();
            expected_idx += seed_carrier.clone() *
                (c::<AB>(PROOF_SHAPE_VK_META_BOUNDARY_X_BASE) +
                    local.chunk_idx.clone() * c::<AB>(STATEMENT_GLOBAL_CHUNK_ELTS) +
                    c::<AB>(lane));
            let infinity_y_offset = if lane == 0 {
                local.interval_chunk_flags[1].clone() *
                    (AB::one_maybe() - boundary_kind.clone())
            } else {
                AB::zero_maybe()
            };
            expected_value += seed_carrier *
                (local.interval_start[lane].clone() - infinity_y_offset);
        }
        builder.assert_eq(local.pv_idxs[slot].clone(), expected_idx);
        builder.assert_eq(local.pv_values[slot].clone(), expected_value);
    }
    for idx in 0..DIGEST_SIZE {
        let public_dt: AB::VarMaybeExt =
            builder.public()[NATIVE_PV_DT_VK_DIGEST_START + idx].clone().into();
        builder.assert_zero(
            local.is_dt_vk.clone() * native.clone() * (local.pv_values[idx].clone() - public_dt),
        );
    }
}

fn constrain_vk<AB: FullAirBuilder>(
    builder: &mut AB,
    local: &StatementBoundaryCols<AB::VarMaybeExt>,
    role: RecursionStatementRole,
    core: AB::VarMaybeExt,
    native: AB::VarMaybeExt,
) {
    for idx in 0..DIGEST_SIZE {
        let public_dt: AB::VarMaybeExt =
            builder.public()[NATIVE_PV_DT_VK_DIGEST_START + idx].clone().into();
        builder.assert_zero(
            local.is_vk_root.clone() *
                core.clone() *
                (local.child_vk_digest[idx].clone() - public_dt),
        );
    }
    builder.assert_eq(
        local.f_baked.clone() + local.f_thread.clone(),
        local.is_vk_root.clone() * native,
    );
    match role {
        RecursionStatementRole::Lift | RecursionStatementRole::ReduceL2 => {
            builder.assert_zero(local.f_baked.clone() * local.class_id.clone());
            for value in &local.child_vk_root {
                builder.assert_zero(local.f_baked.clone() * value.clone());
            }
        }
        RecursionStatementRole::ReduceL3 => {
            for idx in 0..DIGEST_SIZE {
                builder.assert_zero(
                    local.f_baked.clone() *
                        (AB::one_maybe() - local.class_id.clone()) *
                        local.child_vk_root[idx].clone(),
                );
                builder.assert_zero(
                    local.f_baked.clone() *
                        local.class_id.clone() *
                        (local.child_vk_root[idx].clone() - local.child_vk_digest[idx].clone()),
                );
            }
        }
        RecursionStatementRole::RootShrink => {
            builder.assert_zero(local.f_baked.clone() * (local.class_id.clone() - c::<AB>(2)));
            for value in &local.child_vk_root {
                builder.assert_zero(local.f_baked.clone() * value.clone());
            }
        }
    }
    match role {
        RecursionStatementRole::ReduceL2 | RecursionStatementRole::ReduceL3 => builder.assert_zero(
            local.is_vk_root.clone() *
                (local.summary_id_base.clone() -
                    (local.f_baked.clone() * local.class_id.clone() + local.f_thread.clone()) *
                        c::<AB>(128)),
        ),
        _ => builder.assert_zero(local.is_vk_root.clone() * local.summary_id_base.clone()),
    }
    if role == RecursionStatementRole::ReduceL2 {
        for idx in 0..DIGEST_SIZE {
            let public_root: AB::VarMaybeExt =
                builder.public()[NATIVE_PV_VK_ROOT_START + idx].clone().into();
            builder.assert_zero(
                local.f_thread.clone() * (local.child_vk_digest[idx].clone() - public_root.clone()),
            );
            builder.assert_zero(
                local.f_thread.clone() * (local.child_vk_root[idx].clone() - public_root),
            );
        }
    } else {
        builder.assert_zero(local.f_thread.clone());
    }
}

fn constrain_export<AB: FullAirBuilder>(
    builder: &mut AB,
    local: &StatementBoundaryCols<AB::VarMaybeExt>,
    role: RecursionStatementRole,
) {
    let active = local.is_export.clone();
    let one = AB::one_maybe();
    assert_bool(builder, local.statement_is_complete.clone());
    builder
        .assert_zero(local.statement_is_complete.clone() * (one.clone() - local.is_export.clone()));
    builder.assert_zero(
        active.clone() * (local.child_count.clone() * local.child_count_inv.clone() - one.clone()),
    );
    builder.assert_zero(active.clone() * local.cursor_out.clone());
    for idx in 0..STATEMENT_SCALAR_STATE_ELTS {
        let expected = match idx {
            SCALAR_PC => local.scalar_in[SCALAR_START_PC_OUT].clone(),
            SCALAR_SHARD => local.scalar_in[SCALAR_START_SHARD_OUT].clone(),
            SCALAR_EXEC => local.scalar_in[SCALAR_START_EXEC_OUT].clone(),
            SCALAR_EXEC_SEEN => AB::zero_maybe(),
            SCALAR_INIT_ADDR => local.scalar_in[SCALAR_PREV_INIT_OUT].clone(),
            SCALAR_FIN_ADDR => local.scalar_in[SCALAR_PREV_FIN_OUT].clone(),
            _ => local.scalar_in[idx].clone(),
        };
        builder.assert_zero(active.clone() * (local.scalar_out[idx].clone() - expected));
    }
    for idx in 0..NATIVE_RECURSION_NUM_PV_ELTS {
        if (NATIVE_PV_GLOBAL_INTERVAL_START..
            NATIVE_PV_GLOBAL_INTERVAL_END + STATEMENT_GLOBAL_STATE_ELTS)
            .contains(&idx) ||
            (crate::statement_dt::NATIVE_PV_DIGEST_START..NATIVE_RECURSION_NUM_PV_ELTS)
                .contains(&idx)
        {
            continue;
        }
        if role == RecursionStatementRole::ReduceL2 &&
            (NATIVE_PV_VK_ROOT_START..NATIVE_PV_VK_ROOT_START + DIGEST_SIZE).contains(&idx)
        {
            continue;
        }
        let actual = match idx {
            0..=31 => local.export_cvd[idx / 8][idx % 8].clone(),
            NATIVE_PV_START_PC => local.scalar_in[SCALAR_START_PC_OUT].clone(),
            NATIVE_PV_NEXT_PC => local.scalar_in[SCALAR_PC].clone(),
            NATIVE_PV_START_SHARD => local.scalar_in[SCALAR_START_SHARD_OUT].clone(),
            NATIVE_PV_NEXT_SHARD => local.scalar_in[SCALAR_SHARD].clone(),
            NATIVE_PV_START_EXECUTION_SHARD => local.scalar_in[SCALAR_START_EXEC_OUT].clone(),
            NATIVE_PV_NEXT_EXECUTION_SHARD => local.scalar_in[SCALAR_EXEC].clone(),
            NATIVE_PV_PREVIOUS_INIT_ADDR => local.scalar_in[SCALAR_PREV_INIT_OUT].clone(),
            NATIVE_PV_LAST_INIT_ADDR => local.scalar_in[SCALAR_INIT_ADDR].clone(),
            NATIVE_PV_PREVIOUS_FINALIZE_ADDR => local.scalar_in[SCALAR_PREV_FIN_OUT].clone(),
            NATIVE_PV_LAST_FINALIZE_ADDR => local.scalar_in[SCALAR_FIN_ADDR].clone(),
            NATIVE_PV_DT_VK_DIGEST_START..NATIVE_PV_GLOBAL_INTERVAL_START => {
                builder.public()[idx].clone().into()
            }
            NATIVE_PV_IS_COMPLETE => local.statement_is_complete.clone(),
            NATIVE_PV_CONTAINS_EXECUTION_SHARD => local.scalar_in[SCALAR_EXEC_SEEN].clone(),
            _ => AB::zero_maybe(),
        };
        let public: AB::VarMaybeExt = builder.public()[idx].clone().into();
        builder.assert_zero(active.clone() * (actual - public));
    }
    let complete = local.statement_is_complete.clone();
    if role == RecursionStatementRole::RootShrink {
        builder.assert_zero(complete.clone() * local.scalar_in[SCALAR_PC].clone());
        builder.assert_zero(
            complete.clone() * (local.scalar_in[SCALAR_START_SHARD_OUT].clone() - one.clone()),
        );
        builder.assert_zero(
            complete.clone() *
                ((local.scalar_in[SCALAR_SHARD].clone() - one.clone()) *
                    local.complete_next_shard_inv.clone() -
                    one.clone()),
        );
        builder.assert_zero(
            complete.clone() * (local.scalar_in[SCALAR_EXEC_SEEN].clone() - one.clone()),
        );
        builder.assert_zero(complete * (local.scalar_in[SCALAR_START_EXEC_OUT].clone() - one));
    }
}

fn pv_multiplicities<AB: FullAirBuilder>(
    local: &StatementBoundaryCols<AB::VarMaybeExt>,
    core: AB::VarMaybeExt,
    native: AB::VarMaybeExt,
) -> [AB::VarMaybeExt; STATEMENT_PV_SLOTS] {
    core::array::from_fn(|slot| {
        let scalar = if slot <= 7 {
            AB::one_maybe()
        } else {
            let core_slot = if (11..STATEMENT_SCALAR_VALUE_ELTS + STATEMENT_DEFERRED_DIGEST_ELTS)
                .contains(&slot)
            {
                core.clone()
            } else {
                AB::zero_maybe()
            };
            let native_slot = if (8..STATEMENT_PV_SLOTS).contains(&slot) {
                native.clone()
            } else {
                AB::zero_maybe()
            };
            core_slot + native_slot
        };
        let interval = if slot < 2 * STATEMENT_GLOBAL_CHUNK_ELTS {
            local.is_interval.clone()
        } else if slot < 3 * STATEMENT_GLOBAL_CHUNK_ELTS {
            local.seed_admit.clone()
        } else {
            AB::zero_maybe()
        };
        local.is_scalar.clone() * scalar +
            if slot < 8 {
                local.is_cvd.clone() +
                    (local.is_dt_vk.clone() + local.is_vk_root.clone()) * native.clone()
            } else if slot < 12 {
                local.is_dt_vk.clone() * native.clone()
            } else {
                AB::zero_maybe()
            } +
            interval
    })
}

fn boundary_kind_lookup_multiplicity<T: Clone>(local: &StatementBoundaryCols<T>) -> T {
    local.is_interval.clone()
}

fn boundary_rows(
    record: &RecursionRecord,
    role: RecursionStatementRole,
    statement_config: &[StatementConfigRow],
) -> Vec<Vec<F>> {
    let Some(statement) = record.statement_public_values else {
        return Vec::new();
    };
    let mut rows = Vec::with_capacity(
        (8 + STATEMENT_GLOBAL_CHUNKS) * record.proof_records.len() + 1 + STATEMENT_GLOBAL_CHUNKS,
    );
    let mut scalar = scalar_seed(&statement);
    let mut cvd = [[F::zero(); 8]; 4];
    let child_count = record.proof_records.len();
    let mut proofs = record.proof_records.iter().collect::<Vec<_>>();
    proofs.sort_by_key(|proof| proof.proof_idx);
    for proof in proofs {
        let child = &proof.proof_shape.public_values;
        let core = proof.proof_shape.role_id == WHIR_ROLE_CORE;
        let mut row = zero_row();
        row.proof_idx = f(proof.proof_idx);
        row.is_valid = F::one();
        row.is_scalar = F::one();
        row.cursor = f(proof.proof_idx);
        row.cursor_out = f(proof.proof_idx + 1);
        row.child_count = f(child_count);
        row.scalar_in = scalar;
        if core {
            row.start_pc = child[CORE_PV_START_PC];
            row.next_pc = child[CORE_PV_NEXT_PC];
            row.shard = child[CORE_PV_SHARD];
            row.next_shard = row.shard + F::one();
            row.execution_shard = child[CORE_PV_EXECUTION_SHARD];
            row.next_execution_shard = row.execution_shard + F::one();
            row.previous_init_addr = child[CORE_PV_PREVIOUS_INIT_ADDR];
            row.last_init_addr = child[CORE_PV_LAST_INIT_ADDR];
            row.previous_finalize_addr = child[CORE_PV_PREVIOUS_FINALIZE_ADDR];
            row.last_finalize_addr = child[CORE_PV_LAST_FINALIZE_ADDR];
            row.exit_code = child[CORE_PV_EXIT_CODE];
            row.start_clk = child[CORE_PV_START_CLK];
            row.exit_clk = child[CORE_PV_EXIT_CLK];
            row.core_clk_delta = row.exit_clk - row.start_clk;
            row.deferred_digest =
                core::array::from_fn(|idx| child[CORE_PV_DEFERRED_PROOFS_DIGEST_START + idx]);
            row.is_exec = F::from_bool(row.core_clk_delta != F::zero());
        } else {
            row.start_pc = child[NATIVE_PV_START_PC];
            row.next_pc = child[NATIVE_PV_NEXT_PC];
            row.shard = child[NATIVE_PV_START_SHARD];
            row.next_shard = child[NATIVE_PV_NEXT_SHARD];
            row.execution_shard = child[NATIVE_PV_START_EXECUTION_SHARD];
            row.next_execution_shard = child[NATIVE_PV_NEXT_EXECUTION_SHARD];
            row.previous_init_addr = child[NATIVE_PV_PREVIOUS_INIT_ADDR];
            row.last_init_addr = child[NATIVE_PV_LAST_INIT_ADDR];
            row.previous_finalize_addr = child[NATIVE_PV_PREVIOUS_FINALIZE_ADDR];
            row.last_finalize_addr = child[NATIVE_PV_LAST_FINALIZE_ADDR];
            row.is_exec = child[NATIVE_PV_CONTAINS_EXECUTION_SHARD];
            for (slot, offset) in native_extra_offsets()
                .into_iter()
                .take(STATEMENT_NATIVE_SCALAR_EXTRA_ELTS)
                .enumerate()
            {
                row.pv_idxs[11 + slot] = f(offset);
                row.pv_values[11 + slot] = child[offset];
            }
        }
        row.is_first_shard = F::from_bool(row.shard == F::one());
        if core {
            let shard = row.shard.as_canonical_u32() as usize;
            row.shard_lo = f(shard & 0xff);
            row.shard_hi = f(shard >> 8);
        }
        row.cursor_is_zero = F::from_bool(proof.proof_idx == 0);
        row.cursor_inv = inv_or_zero(row.cursor);
        row.shard_minus_one_inv = inv_or_zero(row.shard - F::one());
        row.exec_start_pc_inv =
            if row.is_exec == F::one() { inv_or_zero(row.start_pc) } else { F::zero() };
        row.core_clk_delta_inv = inv_or_zero(row.core_clk_delta);
        row.first_seen =
            F::from_bool(row.is_exec == F::one() && scalar[SCALAR_EXEC_SEEN] == F::zero());
        let mut scalar_out = scalar;
        scalar_out[SCALAR_PC] = row.next_pc;
        scalar_out[SCALAR_SHARD] = row.next_shard;
        scalar_out[SCALAR_INIT_ADDR] = row.last_init_addr;
        scalar_out[SCALAR_FIN_ADDR] = row.last_finalize_addr;
        if row.is_exec == F::one() {
            scalar_out[SCALAR_EXEC] = row.next_execution_shard;
            if scalar[SCALAR_EXEC_SEEN] == F::zero() {
                scalar_out[SCALAR_START_EXEC_OUT] = row.execution_shard;
            }
            scalar_out[SCALAR_EXEC_SEEN] = F::one();
        }
        row.scalar_out = scalar_out;
        fill_scalar_pv(&mut row, core);
        rows.push(row_as_vec(row));

        for chunk in 0..STATEMENT_GLOBAL_CHUNKS {
            let mut row = zero_row();
            row.proof_idx = f(proof.proof_idx);
            row.is_valid = F::one();
            row.is_interval = F::one();
            row.cursor = f(proof.proof_idx);
            row.cursor_out = f(proof.proof_idx + 1);
            row.cursor_is_zero = F::from_bool(proof.proof_idx == 0);
            row.cursor_inv = inv_or_zero(row.cursor);
            row.child_count = f(child_count);
            row.chunk_idx = f(chunk);
            row.interval_chunk_flags[chunk] = F::one();
            row.is_exec = F::from_bool(if core {
                child[CORE_PV_START_CLK] != child[CORE_PV_EXIT_CLK]
            } else {
                child[NATIVE_PV_CONTAINS_EXECUTION_SHARD] == F::one()
            });
            row.is_first_shard = F::from_bool(
                if core { child[CORE_PV_SHARD] } else { child[NATIVE_PV_START_SHARD] } == F::one(),
            );
            let child_start =
                if core { CORE_PV_GLOBAL_INTERVAL_START } else { NATIVE_PV_GLOBAL_INTERVAL_START };
            let child_end =
                if core { CORE_PV_GLOBAL_INTERVAL_END } else { NATIVE_PV_GLOBAL_INTERVAL_END };
            let publish_seed = proof.proof_shape.publish_whir_inputs &&
                core &&
                child[CORE_PV_SHARD] == F::one();
            if proof.proof_shape.publish_whir_inputs && core {
                row.shard_minus_one_inv =
                    proof.proof_shape.vk_meta[PROOF_SHAPE_VK_META_BOUNDARY_KIND];
            }
            row.seed_admit = F::from_bool(publish_seed && (chunk == 0 || chunk == 1));
            for lane in 0..STATEMENT_GLOBAL_CHUNK_ELTS {
                let offset = chunk * STATEMENT_GLOBAL_CHUNK_ELTS + lane;
                if offset >= STATEMENT_GLOBAL_STATE_ELTS {
                    continue;
                }
                row.interval_start[lane] = child[child_start + offset];
                row.interval_end[lane] = child[child_end + offset];
                row.pv_idxs[lane] = f(child_start + offset);
                row.pv_values[lane] = row.interval_start[lane];
                row.pv_idxs[STATEMENT_GLOBAL_CHUNK_ELTS + lane] = f(child_end + offset);
                row.pv_values[STATEMENT_GLOBAL_CHUNK_ELTS + lane] = row.interval_end[lane];
                if publish_seed && chunk < 2 {
                    row.pv_idxs[2 * STATEMENT_GLOBAL_CHUNK_ELTS + lane] =
                        f(PROOF_SHAPE_VK_META_BOUNDARY_X_BASE + offset);
                    row.pv_values[2 * STATEMENT_GLOBAL_CHUNK_ELTS + lane] =
                        proof.proof_shape.vk_meta[PROOF_SHAPE_VK_META_BOUNDARY_X_BASE + offset];
                }
            }
            rows.push(row_as_vec(row));
        }

        for chunk in 0..4 {
            let mut row = zero_row();
            row.proof_idx = f(proof.proof_idx);
            row.is_valid = F::one();
            row.is_cvd = F::one();
            row.cursor = f(proof.proof_idx);
            row.cursor_out = f(proof.proof_idx + 1);
            row.child_count = f(child_count);
            row.chunk_idx = f(chunk);
            row.is_exec = F::from_bool(if core {
                child[CORE_PV_START_CLK] != child[CORE_PV_EXIT_CLK]
            } else {
                child[NATIVE_PV_CONTAINS_EXECUTION_SHARD] == F::one()
            });
            row.is_first_shard = F::from_bool(
                if core { child[CORE_PV_SHARD] } else { child[NATIVE_PV_START_SHARD] } == F::one(),
            );
            row.cursor_is_zero = F::from_bool(proof.proof_idx == 0);
            row.cursor_inv = inv_or_zero(row.cursor);
            row.digest_acc_in = cvd[chunk];
            row.digest_values = core::array::from_fn(|idx| {
                child[(if core {
                    CORE_PV_COMMITTED_VALUE_DIGEST_START
                } else {
                    NATIVE_PV_COMMITTED_VALUE_DIGEST_START
                }) + chunk * 8 +
                    idx]
            });
            row.digest_acc_out = row.digest_values;
            row.cvd_freeze_active = F::from_bool(proof.proof_idx != 0);
            for idx in 0..8 {
                row.digest_nonzero[idx] = F::from_bool(row.digest_acc_in[idx] != F::zero());
                row.digest_nonzero_inv[idx] = inv_or_zero(row.digest_acc_in[idx]);
                row.pv_idxs[idx] = f((if core {
                    CORE_PV_COMMITTED_VALUE_DIGEST_START
                } else {
                    NATIVE_PV_COMMITTED_VALUE_DIGEST_START
                }) + chunk * 8 +
                    idx);
                row.pv_values[idx] = row.digest_values[idx];
            }
            cvd[chunk] = row.digest_values;
            rows.push(row_as_vec(row));
        }

        let mut dt_row = zero_row();
        dt_row.proof_idx = f(proof.proof_idx);
        dt_row.is_valid = F::one();
        dt_row.is_dt_vk = F::one();
        dt_row.cursor = f(proof.proof_idx);
        dt_row.child_count = f(child_count);
        dt_row.is_exec = F::from_bool(if core {
            child[CORE_PV_START_CLK] != child[CORE_PV_EXIT_CLK]
        } else {
            child[NATIVE_PV_CONTAINS_EXECUTION_SHARD] == F::one()
        });
        dt_row.is_first_shard = F::from_bool(
            if core { child[CORE_PV_SHARD] } else { child[NATIVE_PV_START_SHARD] } == F::one(),
        );
        if !core {
            for idx in 0..8 {
                dt_row.pv_idxs[idx] = f(NATIVE_PV_DT_VK_DIGEST_START + idx);
                dt_row.pv_values[idx] = child[NATIVE_PV_DT_VK_DIGEST_START + idx];
            }
            for slot in 8..12 {
                let extra = STATEMENT_NATIVE_SCALAR_EXTRA_ELTS + slot - 8;
                let offset = native_extra_offsets()[extra];
                dt_row.pv_idxs[slot] = f(offset);
                dt_row.pv_values[slot] = child[offset];
            }
        }
        rows.push(row_as_vec(dt_row));

        let mut vk_row = zero_row();
        vk_row.proof_idx = f(proof.proof_idx);
        vk_row.is_valid = F::one();
        vk_row.is_vk_root = F::one();
        vk_row.cursor = f(proof.proof_idx);
        vk_row.child_count = f(child_count);
        vk_row.is_exec = F::from_bool(if core {
            child[CORE_PV_START_CLK] != child[CORE_PV_EXIT_CLK]
        } else {
            child[NATIVE_PV_CONTAINS_EXECUTION_SHARD] == F::one()
        });
        vk_row.is_first_shard = F::from_bool(
            if core { child[CORE_PV_SHARD] } else { child[NATIVE_PV_START_SHARD] } == F::one(),
        );
        vk_row.num_rounds = f(proof.batch_constraint.num_rounds);
        vk_row.c_chips = f(proof.batch_constraint.c_chips);
        vk_row.summary_id_base = f(proof.proof_shape.segment_id_base());
        vk_row.child_vk_digest =
            child_vk_digest_with_memo(&proof.proof_shape, &record.poseidon2_memo);
        if !core {
            for idx in 0..8 {
                vk_row.child_vk_root[idx] = child[NATIVE_PV_VK_ROOT_START + idx];
                vk_row.pv_idxs[idx] = f(NATIVE_PV_VK_ROOT_START + idx);
                vk_row.pv_values[idx] = vk_row.child_vk_root[idx];
            }
            match resolve_child_vk_class_with_memo(
                proof,
                record.statement_vk_root,
                statement_config,
                &record.poseidon2_memo,
            ) {
                Ok(ChildVkClass::Baked(index)) => {
                    vk_row.f_baked = F::one();
                    vk_row.class_id = f(statement_config[index].class_id);
                }
                Ok(ChildVkClass::Threaded) => vk_row.f_thread = F::one(),
                _ => {}
            }
        }
        rows.push(row_as_vec(vk_row));
        scalar = scalar_out;
    }

    for chunk in 0..STATEMENT_GLOBAL_CHUNKS {
        let mut row = zero_row();
        row.is_valid = F::one();
        row.is_interval_export = F::one();
        row.child_count = f(child_count);
        row.cursor_out = f(child_count);
        row.chunk_idx = f(chunk);
        row.interval_chunk_flags[chunk] = F::one();
        for lane in 0..STATEMENT_GLOBAL_CHUNK_ELTS {
            let offset = chunk * STATEMENT_GLOBAL_CHUNK_ELTS + lane;
            if offset >= STATEMENT_GLOBAL_STATE_ELTS {
                continue;
            }
            row.interval_start[lane] = flatten_global(&statement.global_interval_start)[offset];
            row.interval_end[lane] = flatten_global(&statement.global_interval_end)[offset];
        }
        rows.push(row_as_vec(row));
    }

    let mut export = zero_row();
    export.is_valid = F::one();
    export.is_export = F::one();
    export.statement_is_complete = statement.is_complete;
    export.cursor = f(child_count);
    export.child_count = f(child_count);
    export.child_count_inv = inv_or_zero(export.child_count);
    export.scalar_in = scalar;
    export.scalar_out = scalar_seed(&statement);
    export.export_cvd = cvd;
    if role == RecursionStatementRole::RootShrink && statement.is_complete == F::one() {
        export.complete_next_shard_inv = inv_or_zero(scalar[SCALAR_SHARD] - F::one());
    }
    rows.push(row_as_vec(export));
    rows
}

fn fill_scalar_pv(row: &mut StatementBoundaryCols<F>, core: bool) {
    let values = [
        row.start_pc,
        row.next_pc,
        row.shard,
        row.execution_shard,
        row.previous_init_addr,
        row.last_init_addr,
        row.previous_finalize_addr,
        row.last_finalize_addr,
        row.next_shard,
        row.next_execution_shard,
        row.is_exec,
        row.exit_code,
        row.start_clk,
        row.exit_clk,
    ];
    let core_offsets = [
        CORE_PV_START_PC,
        CORE_PV_NEXT_PC,
        CORE_PV_SHARD,
        CORE_PV_EXECUTION_SHARD,
        CORE_PV_PREVIOUS_INIT_ADDR,
        CORE_PV_LAST_INIT_ADDR,
        CORE_PV_PREVIOUS_FINALIZE_ADDR,
        CORE_PV_LAST_FINALIZE_ADDR,
        0,
        0,
        0,
        CORE_PV_EXIT_CODE,
        CORE_PV_START_CLK,
        CORE_PV_EXIT_CLK,
    ];
    let native_offsets = [
        NATIVE_PV_START_PC,
        NATIVE_PV_NEXT_PC,
        NATIVE_PV_START_SHARD,
        NATIVE_PV_START_EXECUTION_SHARD,
        NATIVE_PV_PREVIOUS_INIT_ADDR,
        NATIVE_PV_LAST_INIT_ADDR,
        NATIVE_PV_PREVIOUS_FINALIZE_ADDR,
        NATIVE_PV_LAST_FINALIZE_ADDR,
        NATIVE_PV_NEXT_SHARD,
        NATIVE_PV_NEXT_EXECUTION_SHARD,
        NATIVE_PV_CONTAINS_EXECUTION_SHARD,
        0,
        0,
        0,
    ];
    for slot in 0..14 {
        if (core && (slot <= 7 || slot >= 11)) || (!core && slot <= 10) {
            row.pv_idxs[slot] = f(if core { core_offsets[slot] } else { native_offsets[slot] });
            row.pv_values[slot] = values[slot];
        }
    }
    if core {
        for idx in 0..8 {
            row.pv_idxs[14 + idx] = f(CORE_PV_DEFERRED_PROOFS_DIGEST_START + idx);
            row.pv_values[14 + idx] = row.deferred_digest[idx];
        }
    }
}

fn scalar_seed(statement: &crate::statement_dt::NativeRecursionPublicValues<F>) -> [F; 11] {
    [
        statement.start_pc,
        statement.start_shard,
        statement.start_execution_shard,
        F::zero(),
        statement.previous_init_addr,
        statement.previous_finalize_addr,
        statement.start_pc,
        statement.start_shard,
        statement.start_execution_shard,
        statement.previous_init_addr,
        statement.previous_finalize_addr,
    ]
}

fn flatten_global(state: &[[F; 11]; 3]) -> [F; STATEMENT_GLOBAL_STATE_ELTS] {
    core::array::from_fn(|idx| state[idx / 11][idx % 11])
}

const fn native_extra_offsets() -> [usize; STATEMENT_NATIVE_EXTRA_ELTS] {
    let mut offsets = [0; STATEMENT_NATIVE_EXTRA_ELTS];
    let mut idx = 0;
    while idx < 8 {
        offsets[idx] = NATIVE_PV_DEFERRED_PROOFS_DIGEST_START + idx;
        offsets[8 + idx] = NATIVE_PV_START_RECONSTRUCT_DEFERRED_DIGEST_START + idx;
        offsets[16 + idx] = NATIVE_PV_END_RECONSTRUCT_DEFERRED_DIGEST_START + idx;
        idx += 1;
    }
    offsets[24] = NATIVE_PV_IS_COMPLETE;
    offsets[25] = NATIVE_PV_EXIT_CODE;
    offsets
}

fn zero_row() -> StatementBoundaryCols<F> {
    let values = vec![F::zero(); NUM_STATEMENT_BOUNDARY_COLS];
    let row: &StatementBoundaryCols<F> = values.as_slice().borrow();
    row.clone()
}

fn row_as_vec(row: StatementBoundaryCols<F>) -> Vec<F> {
    let mut values = vec![F::zero(); NUM_STATEMENT_BOUNDARY_COLS];
    *values.as_mut_slice().borrow_mut() = row;
    values
}

fn inv_or_zero(value: F) -> F {
    if value == F::zero() {
        F::zero()
    } else {
        value.inverse()
    }
}
fn f(value: usize) -> F {
    F::from_canonical_usize(value)
}
fn c<AB: FullAirBuilder>(value: usize) -> AB::VarMaybeExt {
    AB::VarMaybeExt::from(AB::F::from_canonical_usize(value))
}
fn assert_bool<AB: FullAirBuilder>(builder: &mut AB, value: AB::VarMaybeExt) {
    builder.assert_zero(value.clone() * (value - AB::one_maybe()));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::D_EF, symbolic_expr_fixed_dt::RecursionFixedSymbolicChip,
        symbolic_ir_dt::RecursionPolyAirChipIr, system_dt::RecursionProofRecord,
    };
    use polyair::Chip;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct SymbolicShape {
        main: usize,
        reserved: usize,
        precomputed: usize,
        permutation: usize,
        active: usize,
        gates: usize,
        alpha: usize,
        degree: usize,
        nodes: usize,
        nodes_padded: usize,
        roots: usize,
        roots_padded: usize,
        folds: usize,
        folds_padded: usize,
        lookups: usize,
    }

    fn symbolic_shape(air: StatementBoundaryAir) -> SymbolicShape {
        let chip = Chip::<StatementBoundaryAir, F, D_EF>::new(air);
        let fixed =
            RecursionFixedSymbolicChip::from_polyair_chip(0, &chip).expect("fixed Statement");
        let ir = RecursionPolyAirChipIr::compile(&fixed).expect("Statement IR");
        let nodes = ir.node_table.len();
        let roots = ir.gate_roots.len() + 2 * ir.lookup_multiplicity_roots.len();
        let folds = ir.gate_roots.len() +
            ir.lookup_multiplicity_roots.len().div_ceil(ir.logup_batch_size.max(1)) +
            1;
        SymbolicShape {
            main: chip.width(),
            reserved: chip.reserved_poly().len(),
            precomputed: chip.num_precompute(),
            permutation: chip.perm_width(),
            active: chip.reserved_poly().len() + chip.num_precompute() + chip.perm_width(),
            gates: chip.symbolic_builder.gate.len(),
            alpha: chip.num_alpha,
            degree: chip.degree,
            nodes,
            nodes_padded: nodes.max(1).next_power_of_two(),
            roots,
            roots_padded: roots.max(1).next_power_of_two(),
            folds,
            folds_padded: folds.max(1).next_power_of_two(),
            lookups: chip.num_lookup(),
        }
    }

    #[test]
    fn symbolic_analysis_is_literal_per_role_projection() {
        let lift = symbolic_shape(StatementBoundaryAir::new(
            RecursionStatementRole::Lift,
            dt_stark::air::DT_PROOF_NUM_PV_ELTS,
            Vec::new(),
        ));
        let native = symbolic_shape(StatementBoundaryAir::new(
            RecursionStatementRole::RootShrink,
            NATIVE_RECURSION_NUM_PV_ELTS,
            Vec::new(),
        ));
        assert_eq!(lift.main, 257);
        assert_eq!(native.main, 241);
        assert_eq!(lift.reserved, lift.main);
        assert_eq!(native.reserved, native.main);
        assert_eq!(lift.main - native.main, 16);
        assert_eq!(lift.precomputed, lift.lookups);
        assert_eq!(native.precomputed, native.lookups);
        assert_eq!(lift.active, lift.reserved + lift.precomputed + lift.permutation);
        assert_eq!(native.active, native.reserved + native.precomputed + native.permutation);
        assert_eq!(lift.degree, 3);
        assert_eq!(native.degree, 3);
        assert!(lift.gates > 0 && lift.alpha > 0 && lift.nodes > 0 && lift.roots > 0);
        assert!(native.gates > 0 && native.alpha > 0 && native.nodes > 0 && native.roots > 0);
        assert_eq!(lift.nodes_padded, lift.nodes.max(1).next_power_of_two());
        assert_eq!(native.nodes_padded, native.nodes.max(1).next_power_of_two());
        assert_eq!(lift.roots_padded, lift.roots.max(1).next_power_of_two());
        assert_eq!(native.roots_padded, native.roots.max(1).next_power_of_two());
        assert_eq!(lift.folds_padded, lift.folds.max(1).next_power_of_two());
        assert_eq!(native.folds_padded, native.folds.max(1).next_power_of_two());
    }

    #[test]
    fn boundary_uses_interval_rows_per_child_plus_export() {
        let air = StatementBoundaryAir::new(
            RecursionStatementRole::RootShrink,
            NATIVE_RECURSION_NUM_PV_ELTS,
            Vec::new(),
        );
        let one_child = RecursionRecord {
            proof_records: vec![RecursionProofRecord::default()],
            ..Default::default()
        };
        assert_eq!(MachineAir::<F>::num_rows(&air, &one_child), Some(16));

        let three_children = RecursionRecord {
            proof_records: vec![RecursionProofRecord::default(); 3],
            ..Default::default()
        };
        // 10 child rows * 3 + 1 export + 3 root chunks = 34 rows.
        assert_eq!(MachineAir::<F>::num_rows(&air, &three_children), Some(64));

        let eleven_children = RecursionRecord {
            proof_records: vec![RecursionProofRecord::default(); 11],
            ..Default::default()
        };
        assert_eq!(MachineAir::<F>::num_rows(&air, &eleven_children), Some(128));

        let six_children = RecursionRecord {
            proof_records: vec![RecursionProofRecord::default(); 6],
            ..Default::default()
        };
        assert_eq!(MachineAir::<F>::num_rows(&air, &six_children), Some(64));

        let twelve_children = RecursionRecord {
            proof_records: vec![RecursionProofRecord::default(); 12],
            ..Default::default()
        };
        assert_eq!(MachineAir::<F>::num_rows(&air, &twelve_children), Some(128));
    }

    #[test]
    fn narrow_projection_preserves_every_native_scalar_publication() {
        let mut child_values =
            (0..NATIVE_RECURSION_NUM_PV_ELTS).map(|idx| f(idx + 1)).collect::<Vec<_>>();
        install_identity_interval(&mut child_values);
        let record = RecursionRecord {
            statement_public_values: Some(Default::default()),
            proof_records: vec![RecursionProofRecord {
                proof_shape: crate::system_dt::RecursionProofShapeRecord {
                    role_id: crate::whir_dt::WHIR_ROLE_COMPRESS,
                    public_values: child_values,
                    vk_meta: vec![
                        F::zero();
                        crate::proof_shape_dt::PROOF_SHAPE_NATIVE_VK_META_VALUE_COUNT
                    ],
                    ..Default::default()
                },
                ..Default::default()
            }],
            ..Default::default()
        };
        let air = StatementBoundaryAir::new(
            RecursionStatementRole::RootShrink,
            NATIVE_RECURSION_NUM_PV_ELTS,
            Vec::new(),
        );
        let trace = air.generate_trace(&record, &mut RecursionRecord::default());
        let mut expanded = vec![F::zero(); NUM_STATEMENT_BOUNDARY_COLS];
        for (&value, column) in trace.main.values[..NUM_STATEMENT_BOUNDARY_NARROW_COLS]
            .iter()
            .zip(statement_narrow_projection_columns())
        {
            expanded[column] = value;
        }
        let row: &StatementBoundaryCols<F> = expanded.as_slice().borrow();
        for (slot, offset) in
            native_extra_offsets().into_iter().take(STATEMENT_NATIVE_SCALAR_EXTRA_ELTS).enumerate()
        {
            assert_eq!(row.pv_idxs[11 + slot], f(offset), "slot {slot} index");
            assert_eq!(row.pv_values[11 + slot], f(offset + 1), "slot {slot} value");
        }
    }


    #[test]
    fn root_export_carries_interval_without_group_workspace() {
        let mut statement = crate::statement_dt::NativeRecursionPublicValues::<F>::default();
        statement.is_complete = F::one();
        statement.global_interval_end[1][0] = f(7);
        let record =
            RecursionRecord { statement_public_values: Some(statement), ..Default::default() };
        let rows = boundary_rows(&record, RecursionStatementRole::RootShrink, &[]);
        let export: &StatementBoundaryCols<F> = rows[STATEMENT_GLOBAL_CHUNKS].as_slice().borrow();
        assert_eq!(export.is_export, F::one());
        assert_eq!(export.child_vk_digest, [F::zero(); DIGEST_SIZE]);
        assert_eq!(export.child_vk_root, [F::zero(); DIGEST_SIZE]);
    }

    #[test]
    fn boundary_kind_lookup_excludes_interval_export_rows() {
        let mut child = zero_row();
        child.is_interval = F::one();
        let mut export = zero_row();
        export.is_interval_export = F::one();
        for chunk in 0..STATEMENT_GLOBAL_CHUNKS {
            child.interval_chunk_flags = [F::zero(); STATEMENT_GLOBAL_CHUNKS];
            child.interval_chunk_flags[chunk] = F::one();
            export.interval_chunk_flags = child.interval_chunk_flags;
            assert_eq!(boundary_kind_lookup_multiplicity(&child), F::one());
            assert_eq!(boundary_kind_lookup_multiplicity(&export), F::zero());
        }
    }

    #[test]
    fn eleven_limb_interval_chunks_carry_every_lane_without_admit_columns() {
        let mut child_values = (0..NATIVE_RECURSION_NUM_PV_ELTS).map(f).collect::<Vec<_>>();
        install_identity_interval(&mut child_values);
        let mut statement = crate::statement_dt::NativeRecursionPublicValues::<F>::default();
        statement.global_interval_end[1][0] = F::one();
        let record = RecursionRecord {
            statement_public_values: Some(statement),
            proof_records: vec![RecursionProofRecord {
                proof_shape: crate::system_dt::RecursionProofShapeRecord {
                    role_id: crate::whir_dt::WHIR_ROLE_COMPRESS,
                    vk_meta: vec![
                        F::zero();
                        crate::proof_shape_dt::PROOF_SHAPE_NATIVE_VK_META_VALUE_COUNT
                    ],
                    vk_meta_send_mults: vec![
                        0;
                        crate::proof_shape_dt::PROOF_SHAPE_NATIVE_VK_META_VALUE_COUNT
                    ],
                    public_values: child_values.clone(),
                    ..Default::default()
                },
                ..Default::default()
            }],
            ..Default::default()
        };
        let rows = boundary_rows(&record, RecursionStatementRole::RootShrink, &[]);
        for chunk in 0..STATEMENT_GLOBAL_CHUNKS {
            let row: &StatementBoundaryCols<F> = rows[1 + chunk].as_slice().borrow();
            for lane in 0..STATEMENT_GLOBAL_CHUNK_ELTS {
                let offset = chunk * STATEMENT_GLOBAL_CHUNK_ELTS + lane;
                assert_eq!(
                    row.interval_end[lane],
                    child_values[NATIVE_PV_GLOBAL_INTERVAL_END + offset]
                );
            }
        }
    }

    fn install_identity_interval(public: &mut [F]) {
        let identity = [
            [F::zero(); 11],
            {
                let mut y = [F::zero(); 11];
                y[0] = F::one();
                y
            },
            [F::zero(); 11],
        ];
        for coordinate in 0..3 {
            public[NATIVE_PV_GLOBAL_INTERVAL_START + coordinate * 11..
                NATIVE_PV_GLOBAL_INTERVAL_START + (coordinate + 1) * 11]
                .copy_from_slice(&identity[coordinate]);
            public[NATIVE_PV_GLOBAL_INTERVAL_END + coordinate * 11..
                NATIVE_PV_GLOBAL_INTERVAL_END + (coordinate + 1) * 11]
                .copy_from_slice(&identity[coordinate]);
        }
    }
}
