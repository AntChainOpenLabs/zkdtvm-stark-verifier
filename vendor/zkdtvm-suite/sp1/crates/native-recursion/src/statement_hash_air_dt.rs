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
use p3_field::{AbstractField, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use crate::{
    config::{DIGEST_SIZE, F, POSEIDON2_WIDTH},
    proof_shape_dt::{ProofShapeValuesBus, PROOF_SHAPE_NAMESPACE_VK_META},
    statement_boundary_air_dt::{StatementHashChainBus, StatementVkDigestBus},
    statement_dt::{
        NATIVE_PV_COMMITTED_VALUE_DIGEST_ELTS, NATIVE_PV_COMMITTED_VALUE_DIGEST_START,
        NATIVE_PV_DIGEST_START, NATIVE_PV_DT_VK_DIGEST_START, NATIVE_RECURSION_NUM_PV_ELMS_TO_HASH,
    },
    system_dt::{RecursionNativeProgram, RecursionRecord, RecursionStatementRole},
    transcript_dt::{
        bus::Poseidon2PermuteBus,
        poseidon2::{RecursionPoseidon2Memo, RecursionPoseidon2Output},
    },
};

use crate::proof_shape_dt::PROOF_SHAPE_VK_META_VALUE_COUNT;

pub const STATEMENT_HASH_KIND_VK_DIGEST: usize = 0;
pub const STATEMENT_HASH_KIND_SELF_DIGEST: usize = 1;
pub const STATEMENT_HASH_KIND_ROOT_DIGEST: usize = 2;

pub const STATEMENT_HASH_RATE: usize = 8;
pub const STATEMENT_HASH_STATE_WIDTH: usize = POSEIDON2_WIDTH;
pub const STATEMENT_GLOBAL146_IDENTITY_ELTS: usize = 32;
pub const STATEMENT_SELF_DIGEST_DOMAIN: [u32; 2] = [0x3634_3147, 0x0031_5650];
pub const STATEMENT_SELF_DIGEST_INPUT_ELTS: usize = 160;
pub const STATEMENT_HASH_BLOCK_SELECTORS: usize = STATEMENT_SELF_DIGEST_BLOCKS;
pub const STATEMENT_VK_DIGEST_BLOCKS: usize = PROOF_SHAPE_VK_META_VALUE_COUNT
    .div_ceil(STATEMENT_HASH_RATE) +
    STATEMENT_GLOBAL146_IDENTITY_ELTS / STATEMENT_HASH_RATE;
pub const STATEMENT_SELF_DIGEST_BLOCKS: usize =
    STATEMENT_SELF_DIGEST_INPUT_ELTS.div_ceil(STATEMENT_HASH_RATE);
pub const STATEMENT_ROOT_DIGEST_BLOCKS: usize = 5;
// RootShrink only admits native children (five VK blocks with the Global146
// identity suffix). Keep the already-versioned six-selector root layout.
pub const STATEMENT_ROOT_BLOCK_SELECTORS: usize = 6;
pub const STATEMENT_ROOT_DIGEST_INPUT_ELTS: usize =
    DIGEST_SIZE + NATIVE_PV_COMMITTED_VALUE_DIGEST_ELTS;
#[cfg(test)]
const STATEMENT_SELF_FINAL_INPUT_LANES: usize = STATEMENT_HASH_RATE;
const STATEMENT_SELF_CHAIN_RATE_START: usize = 0;
const STATEMENT_SELF_CHAIN_RATE_LANES: usize =
    STATEMENT_HASH_RATE - STATEMENT_SELF_CHAIN_RATE_START;
const STATEMENT_ROOT_CHAIN_RATE_START: usize = 0;
const STATEMENT_ROOT_CHAIN_RATE_LANES: usize =
    STATEMENT_HASH_RATE - STATEMENT_ROOT_CHAIN_RATE_START;

/// Which final-digest instance the machine's hash chip runs: a constructor
/// constant from `statement_role` — SELF for Lift/L2/L3, ROOT for RootShrink. The
/// non-selected instance is disallowed in-circuit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatementDigestMode {
    SelfDigest,
    RootDigest,
}

impl StatementDigestMode {
    pub fn from_role(role: RecursionStatementRole) -> Self {
        if role == RecursionStatementRole::RootShrink {
            Self::RootDigest
        } else {
            Self::SelfDigest
        }
    }
}

/// The single source of truth for the ROOT_DIGEST preimage:
/// `public()[66..74] ‖ public()[0..32]` (dt_vk_digest ‖ committed_value_digest, 40 elts,
/// the `sc_root_public_values_digest` form). Used by the AIR gates, tracegen, the
/// SpecStatement host fill, and tests.
pub fn root_digest_input_pv_indices() -> [usize; STATEMENT_ROOT_DIGEST_INPUT_ELTS] {
    core::array::from_fn(|idx| {
        if idx < DIGEST_SIZE {
            NATIVE_PV_DT_VK_DIGEST_START + idx
        } else {
            NATIVE_PV_COMMITTED_VALUE_DIGEST_START + (idx - DIGEST_SIZE)
        }
    })
}

pub fn root_digest_hash_input(public: &[F]) -> [F; STATEMENT_ROOT_DIGEST_INPUT_ELTS] {
    root_digest_input_pv_indices().map(|idx| public[idx])
}

/// Canonical 160-field statement-digest preimage:
/// `"G146" || "PV1\\0" || public[0..151] || zero-padding || 1`.
pub fn statement_self_digest_hash_input(public: &[F]) -> [F; STATEMENT_SELF_DIGEST_INPUT_ELTS] {
    assert!(
        public.len() >= NATIVE_RECURSION_NUM_PV_ELMS_TO_HASH,
        "native statement public values are truncated"
    );
    let mut input = [F::zero(); STATEMENT_SELF_DIGEST_INPUT_ELTS];
    input[0] = F::from_canonical_u32(STATEMENT_SELF_DIGEST_DOMAIN[0]);
    input[1] = F::from_canonical_u32(STATEMENT_SELF_DIGEST_DOMAIN[1]);
    input[2..2 + NATIVE_RECURSION_NUM_PV_ELMS_TO_HASH]
        .copy_from_slice(&public[..NATIVE_RECURSION_NUM_PV_ELMS_TO_HASH]);
    input[STATEMENT_SELF_DIGEST_INPUT_ELTS - 1] = F::one();
    input
}

/// Host-side ROOT_DIGEST value (the sponge the chip proves): the shared rate-8
/// zero-seeded poseidon2 chain over the 40-element input.
pub fn root_public_values_digest(public: &[F]) -> [F; DIGEST_SIZE] {
    crate::statement_dt::poseidon2_hash_slice(&root_digest_hash_input(public))
}

pub(crate) fn root_public_values_digest_with_memo(
    public: &[F],
    memo: &RecursionPoseidon2Memo,
) -> [F; DIGEST_SIZE] {
    crate::statement_dt::poseidon2_hash_slice_with_memo(&root_digest_hash_input(public), memo)
}

macro_rules! define_statement_hash_cols {
    ($name:ident, $block_flags:expr, $chain_rate_lanes:expr) => {
        #[repr(C)]
        #[derive(AlignedBorrow, Debug, Clone)]
        pub struct $name<T> {
            pub proof_idx: T,
            /// Selects the mode-specific final digest; VK rows are
            /// `is_valid - is_final_digest`.
            pub is_final_digest: T,
            pub block_flags: [T; $block_flags],
            /// Previous rate state used by the additive absorb. Capacity
            /// coordinates are already present in `perm_input[8..16]`.
            pub chain_rate_carry: [T; $chain_rate_lanes],
            pub perm_input: [T; STATEMENT_HASH_STATE_WIDTH],
            pub perm_output: [T; STATEMENT_HASH_STATE_WIDTH],
            /// Degree-one lookup multiplicities for full and partial blocks of
            /// transcript-authenticated VK metadata. Composite-identity blocks
            /// are circuit constants and leave both selectors zero.
            pub vk_meta_full_block: T,
            pub vk_meta_tail_block: T,
            /// Degree-one lookup multiplicities cannot inline the product
            /// selecting the short final VK block.
            pub vk_final_block: T,
            /// Degree-one lookup multiplicities also need the mode-specific
            /// final statement block as an admitted selector.
            pub final_digest_block: T,
        }
    };
}

define_statement_hash_cols!(
    StatementHashCols,
    STATEMENT_HASH_BLOCK_SELECTORS,
    STATEMENT_SELF_CHAIN_RATE_LANES
);
define_statement_hash_cols!(
    StatementHashRootCols,
    STATEMENT_ROOT_BLOCK_SELECTORS,
    STATEMENT_ROOT_CHAIN_RATE_LANES
);

pub const NUM_STATEMENT_HASH_COLS: usize = StatementHashCols::<u8>::width();
pub const NUM_STATEMENT_HASH_ROOT_COLS: usize = StatementHashRootCols::<u8>::width();

macro_rules! define_statement_hash_reserved {
    ($name:ident, $block_flags:expr) => {
        #[repr(C)]
        #[derive(AlignedBorrow, Debug, Clone)]
        struct $name<T> {
            pub is_final_digest: T,
            pub block_flags: [T; $block_flags],
            pub vk_meta_full_block: T,
            pub vk_meta_tail_block: T,
            pub vk_final_block: T,
            pub final_digest_block: T,
        }
    };
}

define_statement_hash_reserved!(StatementHashReserved, STATEMENT_HASH_BLOCK_SELECTORS);
define_statement_hash_reserved!(StatementHashRootReserved, STATEMENT_ROOT_BLOCK_SELECTORS);

struct StatementHashMain<'a, T, const BLOCK_FLAGS: usize, const CHAIN_RATE_LANES: usize> {
    proof_idx: &'a T,
    is_final_digest: &'a T,
    block_flags: &'a [T; BLOCK_FLAGS],
    chain_rate_carry: &'a [T; CHAIN_RATE_LANES],
    perm_input: &'a [T; STATEMENT_HASH_STATE_WIDTH],
    perm_output: &'a [T; STATEMENT_HASH_STATE_WIDTH],
}

struct StatementHashDenominators<T> {
    chain_recv: T,
    chain_send: T,
    chain_seed: T,
    proof_values: [T; STATEMENT_HASH_RATE],
    poseidon2: T,
    vk_digest: T,
}

enum StatementHashLinearRelations<T> {
    SelfDigest([T; 8]),
    RootDigest([T; 8]),
}

#[derive(Debug, Clone, Copy)]
pub struct StatementHashAir {
    /// Constructor constant (enters the symbolic DAG): selects SELF vs ROOT for the
    /// machine's final-digest instance; the other kind is asserted absent.
    pub mode: StatementDigestMode,
    /// Child transcript metadata fields. The composite identity is deliberately
    /// excluded: it is a circuit/key constant, not a challenger observation.
    pub vk_meta_value_count: usize,
    pub vk_digest_input_count: usize,
    hash_chain_bus: StatementHashChainBus,
    vk_digest_bus: StatementVkDigestBus,
    proof_values_bus: ProofShapeValuesBus,
    poseidon2_bus: Poseidon2PermuteBus,
}

impl StatementHashAir {
    pub fn new(mode: StatementDigestMode) -> Self {
        let num_child_public_values = match mode {
            StatementDigestMode::SelfDigest => dt_stark::air::DT_PROOF_NUM_PV_ELTS,
            StatementDigestMode::RootDigest => crate::statement_dt::NATIVE_RECURSION_NUM_PV_ELTS,
        };
        Self::for_child(mode, num_child_public_values)
    }

    pub fn for_child(mode: StatementDigestMode, num_child_public_values: usize) -> Self {
        let vk_meta_value_count = if num_child_public_values == dt_stark::air::DT_PROOF_NUM_PV_ELTS
        {
            crate::proof_shape_dt::PROOF_SHAPE_CORE_VK_META_VALUE_COUNT
        } else {
            crate::proof_shape_dt::PROOF_SHAPE_NATIVE_VK_META_VALUE_COUNT
        };
        Self {
            mode,
            vk_meta_value_count,
            vk_digest_input_count: vk_meta_value_count.div_ceil(STATEMENT_HASH_RATE) *
                STATEMENT_HASH_RATE +
                STATEMENT_GLOBAL146_IDENTITY_ELTS,
            hash_chain_bus: StatementHashChainBus::new(),
            vk_digest_bus: StatementVkDigestBus::new(),
            proof_values_bus: ProofShapeValuesBus::new(),
            poseidon2_bus: Poseidon2PermuteBus::new(),
        }
    }
}

impl Default for StatementHashAir {
    fn default() -> Self {
        Self::new(StatementDigestMode::SelfDigest)
    }
}

impl BaseAir<F> for StatementHashAir {
    fn width(&self) -> usize {
        match self.mode {
            StatementDigestMode::SelfDigest => NUM_STATEMENT_HASH_COLS,
            StatementDigestMode::RootDigest => NUM_STATEMENT_HASH_ROOT_COLS,
        }
    }
}

impl<AB: FullAirBuilder> FullAir<AB> for StatementHashAir {
    fn width(&self) -> usize {
        BaseAir::<F>::width(self)
    }

    fn required_max_beta_power(&self) -> usize {
        [
            self.hash_chain_bus.required_max_beta_power_floor(),
            self.vk_digest_bus.required_max_beta_power_floor(),
            self.proof_values_bus.required_max_beta_power_floor(),
            self.poseidon2_bus.required_max_beta_power_floor(),
        ]
        .into_iter()
        .max()
        .expect("non-empty bus list")
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        match self.mode {
            StatementDigestMode::SelfDigest => (1..2 + STATEMENT_HASH_BLOCK_SELECTORS)
                .chain(NUM_STATEMENT_HASH_COLS - 4..NUM_STATEMENT_HASH_COLS)
                .map(PairCol::Main)
                .collect(),
            StatementDigestMode::RootDigest => (1..2 + STATEMENT_ROOT_BLOCK_SELECTORS)
                .chain(NUM_STATEMENT_HASH_ROOT_COLS - 4..NUM_STATEMENT_HASH_ROOT_COLS)
                .map(PairCol::Main)
                .collect(),
        }
    }

    fn precompute_lc(&self, builder: &mut AB) {
        match self.mode {
            StatementDigestMode::SelfDigest => {
                let (denominators, relations) = {
                    let main = builder.main();
                    let local: &StatementHashCols<AB::VarMaybeExt> = main.borrow();
                    let local = StatementHashMain {
                        proof_idx: &local.proof_idx,
                        is_final_digest: &local.is_final_digest,
                        block_flags: &local.block_flags,
                        chain_rate_carry: &local.chain_rate_carry,
                        perm_input: &local.perm_input,
                        perm_output: &local.perm_output,
                    };
                    (
                        statement_hash_denominators::<
                            AB,
                            STATEMENT_HASH_BLOCK_SELECTORS,
                            STATEMENT_SELF_CHAIN_RATE_LANES,
                        >(self, builder, &local),
                        statement_hash_linear_relations::<
                            AB,
                            STATEMENT_HASH_BLOCK_SELECTORS,
                            STATEMENT_SELF_CHAIN_RATE_LANES,
                        >(builder, self, &local),
                    )
                };
                retain_statement_hash_denominators(builder, denominators);
                let StatementHashLinearRelations::SelfDigest(relations) = relations else {
                    unreachable!("SelfDigest relation layout")
                };
                for (relation_idx, relation) in relations.into_iter().enumerate() {
                    if self.vk_meta_value_count % STATEMENT_HASH_RATE == 0 &&
                        matches!(relation_idx, 2 | 3)
                    {
                        continue;
                    }
                    builder.retain_precomputed(relation);
                }
            }
            StatementDigestMode::RootDigest => {
                let (denominators, relations) = {
                    let main = builder.main();
                    let local: &StatementHashRootCols<AB::VarMaybeExt> = main.borrow();
                    let local = StatementHashMain {
                        proof_idx: &local.proof_idx,
                        is_final_digest: &local.is_final_digest,
                        block_flags: &local.block_flags,
                        chain_rate_carry: &local.chain_rate_carry,
                        perm_input: &local.perm_input,
                        perm_output: &local.perm_output,
                    };
                    (
                        statement_hash_denominators::<
                            AB,
                            STATEMENT_ROOT_BLOCK_SELECTORS,
                            STATEMENT_ROOT_CHAIN_RATE_LANES,
                        >(self, builder, &local),
                        statement_hash_linear_relations::<
                            AB,
                            STATEMENT_ROOT_BLOCK_SELECTORS,
                            STATEMENT_ROOT_CHAIN_RATE_LANES,
                        >(builder, self, &local),
                    )
                };
                retain_statement_hash_denominators(builder, denominators);
                let StatementHashLinearRelations::RootDigest(relations) = relations else {
                    unreachable!("RootDigest relation layout")
                };
                for (relation_idx, relation) in relations.into_iter().enumerate() {
                    if self.vk_meta_value_count % STATEMENT_HASH_RATE == 0 &&
                        matches!(relation_idx, 2 | 3)
                    {
                        continue;
                    }
                    builder.retain_precomputed(relation);
                }
            }
        }
    }

    fn eval(&self, builder: &mut AB) {
        match self.mode {
            StatementDigestMode::SelfDigest => {
                let reserved = builder.reserved_poly();
                let local_binding = reserved.row_slice(0);
                let local: &StatementHashReserved<AB::VarMaybeExt> = local_binding.deref().borrow();
                eval_statement_hash::<AB, STATEMENT_HASH_BLOCK_SELECTORS>(
                    builder,
                    self.mode,
                    self.vk_meta_value_count,
                    self.vk_digest_input_count,
                    &local.is_final_digest,
                    &local.block_flags,
                    &local.vk_meta_full_block,
                    &local.vk_meta_tail_block,
                    &local.vk_final_block,
                    &local.final_digest_block,
                );
            }
            StatementDigestMode::RootDigest => {
                let reserved = builder.reserved_poly();
                let local_binding = reserved.row_slice(0);
                let local: &StatementHashRootReserved<AB::VarMaybeExt> =
                    local_binding.deref().borrow();
                eval_statement_hash::<AB, STATEMENT_ROOT_BLOCK_SELECTORS>(
                    builder,
                    self.mode,
                    self.vk_meta_value_count,
                    self.vk_digest_input_count,
                    &local.is_final_digest,
                    &local.block_flags,
                    &local.vk_meta_full_block,
                    &local.vk_meta_tail_block,
                    &local.vk_final_block,
                    &local.final_digest_block,
                );
            }
        }
    }

    fn lookup(&self, builder: &mut AB) {
        match self.mode {
            StatementDigestMode::SelfDigest => {
                let reserved = builder.reserved_poly();
                let local_binding = reserved.row_slice(0);
                let local: &StatementHashReserved<AB::VarMaybeExt> = local_binding.deref().borrow();
                lookup_statement_hash::<AB, STATEMENT_HASH_BLOCK_SELECTORS>(
                    builder,
                    self.mode,
                    self.vk_meta_value_count,
                    self.vk_digest_input_count,
                    &local.is_final_digest,
                    &local.block_flags,
                    &local.vk_meta_full_block,
                    &local.vk_meta_tail_block,
                    &local.vk_final_block,
                    &local.final_digest_block,
                );
            }
            StatementDigestMode::RootDigest => {
                let reserved = builder.reserved_poly();
                let local_binding = reserved.row_slice(0);
                let local: &StatementHashRootReserved<AB::VarMaybeExt> =
                    local_binding.deref().borrow();
                lookup_statement_hash::<AB, STATEMENT_ROOT_BLOCK_SELECTORS>(
                    builder,
                    self.mode,
                    self.vk_meta_value_count,
                    self.vk_digest_input_count,
                    &local.is_final_digest,
                    &local.block_flags,
                    &local.vk_meta_full_block,
                    &local.vk_meta_tail_block,
                    &local.vk_final_block,
                    &local.final_digest_block,
                );
            }
        }
    }
}

fn eval_statement_hash<AB: FullAirBuilder, const BLOCK_FLAGS: usize>(
    builder: &mut AB,
    mode: StatementDigestMode,
    vk_meta_value_count: usize,
    vk_digest_input_count: usize,
    is_final_digest: &AB::VarMaybeExt,
    block_flags: &[AB::VarMaybeExt; BLOCK_FLAGS],
    vk_meta_full_block: &AB::VarMaybeExt,
    vk_meta_tail_block: &AB::VarMaybeExt,
    vk_final_block: &AB::VarMaybeExt,
    final_digest_block: &AB::VarMaybeExt,
) {
    let is_valid = statement_hash_is_valid::<AB, BLOCK_FLAGS>(block_flags);
    assert_bool(builder, is_valid.clone());
    assert_bool(builder, is_final_digest.clone());
    assert_bool(builder, vk_meta_full_block.clone());
    assert_bool(builder, vk_meta_tail_block.clone());
    assert_bool(builder, final_digest_block.clone());
    for flag in block_flags {
        assert_bool(builder, flag.clone());
    }

    let final_blocks = match mode {
        StatementDigestMode::SelfDigest => STATEMENT_SELF_DIGEST_BLOCKS,
        StatementDigestMode::RootDigest => STATEMENT_ROOT_DIGEST_BLOCKS,
    };
    let vk_digest_blocks = vk_digest_input_count.div_ceil(STATEMENT_HASH_RATE);
    let invalid_vk = block_flags[vk_digest_blocks..].iter().cloned().sum::<AB::VarMaybeExt>();
    let invalid_final = block_flags[final_blocks..].iter().cloned().sum::<AB::VarMaybeExt>();
    builder.assert_zero(is_final_digest.clone() * (AB::one_maybe() - is_valid.clone()));
    builder.assert_zero((is_valid.clone() - is_final_digest.clone()) * invalid_vk);
    builder.assert_zero(is_final_digest.clone() * invalid_final);
    builder.assert_eq(
        vk_final_block.clone(),
        (is_valid.clone() - is_final_digest.clone()) * block_flags[vk_digest_blocks - 1].clone(),
    );
    builder.assert_eq(
        final_digest_block.clone(),
        is_final_digest.clone() * block_flags[final_blocks - 1].clone(),
    );
    let is_vk = is_valid - is_final_digest.clone();
    let full_meta_blocks = vk_meta_value_count / STATEMENT_HASH_RATE;
    let full_meta_selector =
        block_flags[..full_meta_blocks].iter().cloned().sum::<AB::VarMaybeExt>();
    builder.assert_eq(vk_meta_full_block.clone(), is_vk.clone() * full_meta_selector);
    if vk_meta_value_count % STATEMENT_HASH_RATE == 0 {
        builder.assert_zero(vk_meta_tail_block.clone());
    } else {
        builder
            .assert_eq(vk_meta_tail_block.clone(), is_vk * block_flags[full_meta_blocks].clone());
    }
    constrain_statement_hash_relations(
        builder,
        mode,
        vk_meta_value_count,
        vk_digest_input_count,
        is_final_digest,
        block_flags,
        vk_final_block,
        final_digest_block,
    );
}

fn lookup_statement_hash<AB: FullAirBuilder, const BLOCK_FLAGS: usize>(
    builder: &mut AB,
    _mode: StatementDigestMode,
    vk_meta_value_count: usize,
    _vk_digest_input_count: usize,
    _is_final_digest: &AB::VarMaybeExt,
    block_flags: &[AB::VarMaybeExt; BLOCK_FLAGS],
    vk_meta_full_block: &AB::VarMaybeExt,
    vk_meta_tail_block: &AB::VarMaybeExt,
    vk_final_block: &AB::VarMaybeExt,
    final_digest_block: &AB::VarMaybeExt,
) {
    let is_valid = statement_hash_is_valid::<AB, BLOCK_FLAGS>(block_flags);
    let is_final_block = vk_final_block.clone() + final_digest_block.clone();

    builder.recv(is_valid.clone());
    builder.send(is_valid.clone() - is_final_block);
    builder.send(block_flags[0].clone());
    for lane in 0..STATEMENT_HASH_RATE {
        let tail = if lane < vk_meta_value_count % STATEMENT_HASH_RATE {
            vk_meta_tail_block.clone()
        } else {
            AB::zero_maybe()
        };
        builder.recv(vk_meta_full_block.clone() + tail);
    }
    builder.recv(is_valid);
    builder.send(vk_final_block.clone());
}

impl MachineAir<F> for StatementHashAir {
    type Record = RecursionRecord;
    type Program = RecursionNativeProgram<F>;

    fn name(&self) -> String {
        "NativeStatementHash".to_string()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        Some(StatementHashTraceGenerator::trace_height(input, self.mode))
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        _output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        StatementHashTraceGenerator::generate_trace_compressed(input, self.mode)
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        for row in statement_hash_rows_cached(input, self.mode).iter() {
            output.poseidon2.record_poseidon2(row.perm_input);
        }
    }

    fn included(&self, record: &Self::Record) -> bool {
        record.statement_public_values.is_some() && !record.proof_records.is_empty()
    }

    fn local_only(&self) -> bool {
        true
    }
}

fn retain_statement_hash_denominators<AB: FullAirBuilder>(
    builder: &mut AB,
    denominators: StatementHashDenominators<AB::VarExt>,
) {
    builder.retain_precomputed(denominators.chain_recv);
    builder.retain_precomputed(denominators.chain_send);
    builder.retain_precomputed(denominators.chain_seed);
    for denominator in denominators.proof_values {
        builder.retain_precomputed(denominator);
    }
    builder.retain_precomputed(denominators.poseidon2);
    builder.retain_precomputed(denominators.vk_digest);
}

fn statement_hash_denominators<
    AB: FullAirBuilder,
    const BLOCK_FLAGS: usize,
    const CHAIN_RATE_LANES: usize,
>(
    air: &StatementHashAir,
    builder: &AB,
    local: &StatementHashMain<AB::VarMaybeExt, BLOCK_FLAGS, CHAIN_RATE_LANES>,
) -> StatementHashDenominators<AB::VarExt> {
    let hash_kind = statement_hash_kind::<AB>(air.mode, local.is_final_digest.clone());
    let block_idx = statement_hash_block_idx::<AB, BLOCK_FLAGS>(local.block_flags);
    let is_valid = statement_hash_is_valid::<AB, BLOCK_FLAGS>(local.block_flags);
    let chain_rate_start = statement_hash_chain_rate_start(air.mode);
    let chain_recv = air.hash_chain_bus.denominator(
        builder,
        (*local.proof_idx).clone(),
        hash_chain_payload(
            hash_kind.clone(),
            block_idx.clone(),
            statement_hash_chain_recv_state::<AB, CHAIN_RATE_LANES>(
                local.chain_rate_carry,
                local.perm_input,
                chain_rate_start,
            ),
        ),
    );
    let chain_send = air.hash_chain_bus.denominator(
        builder,
        (*local.proof_idx).clone(),
        hash_chain_payload(
            hash_kind.clone(),
            block_idx.clone() + is_valid,
            statement_hash_chain_send_state::<AB>(local.perm_output, chain_rate_start),
        ),
    );
    let chain_seed = air.hash_chain_bus.denominator(
        builder,
        (*local.proof_idx).clone(),
        hash_chain_payload(
            hash_kind.clone(),
            AB::zero_maybe(),
            core::array::from_fn(|_| AB::zero_maybe()),
        ),
    );
    let proof_values = core::array::from_fn(|lane| {
        air.proof_values_bus.denominator(
            builder,
            (*local.proof_idx).clone(),
            const_maybe::<AB>(PROOF_SHAPE_NAMESPACE_VK_META),
            statement_hash_vk_meta_idx::<AB, BLOCK_FLAGS>(
                local.block_flags,
                lane,
                air.vk_digest_input_count.div_ceil(STATEMENT_HASH_RATE),
            ),
            local.perm_input[lane].clone() - local.chain_rate_carry[lane].clone(),
        )
    });
    let poseidon2 = air.poseidon2_bus.denominator(
        builder,
        (*local.perm_input).clone(),
        (*local.perm_output).clone(),
    );
    let vk_digest = air.vk_digest_bus.denominator(
        builder,
        (*local.proof_idx).clone(),
        vk_digest_payload(hash_kind, (*local.perm_output).clone()),
    );
    StatementHashDenominators {
        chain_recv,
        chain_send,
        chain_seed,
        proof_values,
        poseidon2,
        vk_digest,
    }
}

fn statement_hash_linear_relations<
    AB: FullAirBuilder,
    const BLOCK_FLAGS: usize,
    const CHAIN_RATE_LANES: usize,
>(
    builder: &AB,
    air: &StatementHashAir,
    local: &StatementHashMain<AB::VarMaybeExt, BLOCK_FLAGS, CHAIN_RATE_LANES>,
) -> StatementHashLinearRelations<AB::VarExt> {
    let mode = air.mode;
    let public_residuals: [AB::VarMaybeExt; STATEMENT_HASH_RATE] = match mode {
        StatementDigestMode::SelfDigest => core::array::from_fn(|lane| {
            let expected =
                (0..STATEMENT_SELF_DIGEST_BLOCKS).fold(AB::zero_maybe(), |sum, block| {
                    let input_idx = block * STATEMENT_HASH_RATE + lane;
                    let value = if input_idx == 0 {
                        const_maybe::<AB>(STATEMENT_SELF_DIGEST_DOMAIN[0] as usize)
                    } else if input_idx == 1 {
                        const_maybe::<AB>(STATEMENT_SELF_DIGEST_DOMAIN[1] as usize)
                    } else if input_idx < 2 + NATIVE_RECURSION_NUM_PV_ELMS_TO_HASH {
                        builder.public()[input_idx - 2].clone().into()
                    } else if input_idx == STATEMENT_SELF_DIGEST_INPUT_ELTS - 1 {
                        AB::one_maybe()
                    } else {
                        AB::zero_maybe()
                    };
                    sum + local.block_flags[block].clone() * value
                });
            local.perm_input[lane].clone() - local.chain_rate_carry[lane].clone() - expected
        }),
        StatementDigestMode::RootDigest => {
            let root_pv_idxs = root_digest_input_pv_indices();
            core::array::from_fn(|lane| {
                let expected =
                    (0..STATEMENT_ROOT_DIGEST_BLOCKS).fold(AB::zero_maybe(), |sum, block| {
                        let public = builder.public()
                            [root_pv_idxs[block * STATEMENT_HASH_RATE + lane]]
                            .clone();
                        sum + local.block_flags[block].clone() * public.into()
                    });
                local.perm_input[lane].clone() - local.chain_rate_carry[lane].clone() - expected
            })
        }
    };

    let chain_rate_start = statement_hash_chain_rate_start(mode);
    let carry_residuals: [AB::VarMaybeExt; CHAIN_RATE_LANES] = core::array::from_fn(|lane| {
        local.perm_input[chain_rate_start + lane].clone() - local.chain_rate_carry[lane].clone()
    });
    let digest_residuals: [AB::VarMaybeExt; DIGEST_SIZE] = core::array::from_fn(|lane| {
        local.perm_output[lane].clone() -
            builder.public()[NATIVE_PV_DIGEST_START + lane].clone().into()
    });
    let identity = dt_stark::global_d11::GLOBAL146_COMPOSITE_IDENTITY;
    let identity_block_start = air.vk_meta_value_count.div_ceil(STATEMENT_HASH_RATE);
    let identity_residuals: [AB::VarMaybeExt; STATEMENT_HASH_RATE] = core::array::from_fn(|lane| {
        let expected = (0..STATEMENT_GLOBAL146_IDENTITY_ELTS / STATEMENT_HASH_RATE).fold(
            AB::zero_maybe(),
            |sum, identity_block| {
                sum + local.block_flags[identity_block_start + identity_block].clone() *
                    const_maybe::<AB>(
                        identity[identity_block * STATEMENT_HASH_RATE + lane] as usize,
                    )
            },
        );
        local.perm_input[lane].clone() - local.chain_rate_carry[lane].clone() - expected
    });

    match mode {
        StatementDigestMode::SelfDigest => StatementHashLinearRelations::SelfDigest([
            AB::pack_ext_limbs(&public_residuals[..5]),
            AB::pack_ext_limbs(&public_residuals[5..]),
            vk_padding_residual::<AB, CHAIN_RATE_LANES>(
                &carry_residuals,
                air.vk_meta_value_count,
                0,
            ),
            vk_padding_residual::<AB, CHAIN_RATE_LANES>(
                &carry_residuals,
                air.vk_meta_value_count,
                1,
            ),
            AB::pack_ext_limbs(&identity_residuals[..5]),
            AB::pack_ext_limbs(&identity_residuals[5..]),
            AB::pack_ext_limbs(&digest_residuals[..5]),
            AB::pack_ext_limbs(&digest_residuals[5..]),
        ]),
        StatementDigestMode::RootDigest => StatementHashLinearRelations::RootDigest([
            AB::pack_ext_limbs(&public_residuals[..5]),
            AB::pack_ext_limbs(&public_residuals[5..]),
            vk_padding_residual::<AB, CHAIN_RATE_LANES>(
                &carry_residuals,
                air.vk_meta_value_count,
                0,
            ),
            vk_padding_residual::<AB, CHAIN_RATE_LANES>(
                &carry_residuals,
                air.vk_meta_value_count,
                1,
            ),
            AB::pack_ext_limbs(&identity_residuals[..5]),
            AB::pack_ext_limbs(&identity_residuals[5..]),
            AB::pack_ext_limbs(&digest_residuals[..5]),
            AB::pack_ext_limbs(&digest_residuals[5..]),
        ]),
    }
}

fn vk_padding_residual<AB: FullAirBuilder, const N: usize>(
    carry_residuals: &[AB::VarMaybeExt; N],
    vk_meta_value_count: usize,
    part: usize,
) -> AB::VarExt {
    let remainder = vk_meta_value_count % STATEMENT_HASH_RATE;
    if remainder == 0 {
        return AB::pack_ext_limbs(&[AB::zero_maybe()]);
    }
    let split = (remainder + 5).min(STATEMENT_HASH_RATE);
    match part {
        0 => AB::pack_ext_limbs(&carry_residuals[remainder..split]),
        1 => {
            if split == STATEMENT_HASH_RATE {
                AB::pack_ext_limbs(&[AB::zero_maybe()])
            } else {
                AB::pack_ext_limbs(&carry_residuals[split..STATEMENT_HASH_RATE])
            }
        }
        _ => unreachable!("statement VK padding has two packed residuals"),
    }
}

fn statement_hash_kind<AB: FullAirBuilder>(
    mode: StatementDigestMode,
    is_final_digest: AB::VarMaybeExt,
) -> AB::VarMaybeExt {
    let kind = match mode {
        StatementDigestMode::SelfDigest => STATEMENT_HASH_KIND_SELF_DIGEST,
        StatementDigestMode::RootDigest => STATEMENT_HASH_KIND_ROOT_DIGEST,
    };
    const_maybe::<AB>(kind) * is_final_digest
}

fn statement_hash_block_idx<AB: FullAirBuilder, const BLOCK_FLAGS: usize>(
    block_flags: &[AB::VarMaybeExt; BLOCK_FLAGS],
) -> AB::VarMaybeExt {
    let mut block_idx = AB::zero_maybe();
    for idx in 0..BLOCK_FLAGS {
        block_idx += const_maybe::<AB>(idx) * block_flags[idx].clone();
    }
    block_idx
}

fn statement_hash_is_valid<AB: FullAirBuilder, const BLOCK_FLAGS: usize>(
    block_flags: &[AB::VarMaybeExt; BLOCK_FLAGS],
) -> AB::VarMaybeExt {
    block_flags.iter().cloned().fold(AB::zero_maybe(), |sum, flag| sum + flag)
}

fn statement_hash_vk_meta_idx<AB: FullAirBuilder, const BLOCK_FLAGS: usize>(
    block_flags: &[AB::VarMaybeExt; BLOCK_FLAGS],
    lane: usize,
    vk_digest_blocks: usize,
) -> AB::VarMaybeExt {
    let mut vk_meta_idx = AB::zero_maybe();
    for block in 0..vk_digest_blocks {
        vk_meta_idx +=
            const_maybe::<AB>(block * STATEMENT_HASH_RATE + lane) * block_flags[block].clone();
    }
    vk_meta_idx
}

fn constrain_statement_hash_relations<AB: FullAirBuilder, const BLOCK_FLAGS: usize>(
    builder: &mut AB,
    mode: StatementDigestMode,
    vk_meta_value_count: usize,
    vk_digest_input_count: usize,
    is_final_digest: &AB::VarMaybeExt,
    block_flags: &[AB::VarMaybeExt; BLOCK_FLAGS],
    _vk_final_block: &AB::VarMaybeExt,
    final_digest_block: &AB::VarMaybeExt,
) {
    let precomputed = builder.precomputed();
    let precomputed_binding = precomputed.row_slice(0);
    let values = precomputed_binding.deref();
    let mut relation_idx = 13;
    let meta_block_count = vk_meta_value_count.div_ceil(STATEMENT_HASH_RATE);
    let identity_block_count = STATEMENT_GLOBAL146_IDENTITY_ELTS / STATEMENT_HASH_RATE;
    debug_assert_eq!(vk_digest_input_count.div_ceil(STATEMENT_HASH_RATE), meta_block_count + 4);
    let vk_selector = AB::one_maybe() - is_final_digest.clone();
    let meta_final_selector = vk_selector.clone() * block_flags[meta_block_count - 1].clone();
    let identity_selector = vk_selector *
        block_flags[meta_block_count..meta_block_count + identity_block_count]
            .iter()
            .cloned()
            .sum::<AB::VarMaybeExt>();
    match mode {
        StatementDigestMode::SelfDigest => {
            let final_flag = final_digest_block.clone();
            for _ in 0..2 {
                builder.assert_zero_ext(values[relation_idx].clone() * is_final_digest.clone());
                relation_idx += 1;
            }
            if vk_meta_value_count % STATEMENT_HASH_RATE != 0 {
                for _ in 0..2 {
                    builder.assert_zero_ext(
                        values[relation_idx].clone() * meta_final_selector.clone(),
                    );
                    relation_idx += 1;
                }
            }
            for _ in 0..2 {
                builder.assert_zero_ext(values[relation_idx].clone() * identity_selector.clone());
                relation_idx += 1;
            }
            for _ in 0..2 {
                builder.assert_zero_ext(values[relation_idx].clone() * final_flag.clone());
                relation_idx += 1;
            }
        }
        StatementDigestMode::RootDigest => {
            let final_flag = final_digest_block.clone();
            for _ in 0..2 {
                builder.assert_zero_ext(values[relation_idx].clone() * is_final_digest.clone());
                relation_idx += 1;
            }
            if vk_meta_value_count % STATEMENT_HASH_RATE != 0 {
                for _ in 0..2 {
                    builder.assert_zero_ext(
                        values[relation_idx].clone() * meta_final_selector.clone(),
                    );
                    relation_idx += 1;
                }
            }
            for _ in 0..2 {
                builder.assert_zero_ext(values[relation_idx].clone() * identity_selector.clone());
                relation_idx += 1;
            }
            for _ in 0..2 {
                builder.assert_zero_ext(values[relation_idx].clone() * final_flag.clone());
                relation_idx += 1;
            }
        }
    }
}

fn statement_hash_chain_rate_start(mode: StatementDigestMode) -> usize {
    match mode {
        StatementDigestMode::SelfDigest => STATEMENT_SELF_CHAIN_RATE_START,
        StatementDigestMode::RootDigest => STATEMENT_ROOT_CHAIN_RATE_START,
    }
}

fn statement_hash_chain_recv_state<AB: FullAirBuilder, const CHAIN_RATE_LANES: usize>(
    chain_rate_carry: &[AB::VarMaybeExt; CHAIN_RATE_LANES],
    perm_input: &[AB::VarMaybeExt; STATEMENT_HASH_STATE_WIDTH],
    chain_rate_start: usize,
) -> [AB::VarMaybeExt; STATEMENT_HASH_STATE_WIDTH] {
    core::array::from_fn(|lane| {
        if lane < chain_rate_start {
            AB::zero_maybe()
        } else if lane < STATEMENT_HASH_RATE {
            chain_rate_carry[lane - chain_rate_start].clone()
        } else {
            perm_input[lane].clone()
        }
    })
}

fn statement_hash_chain_send_state<AB: FullAirBuilder>(
    perm_output: &[AB::VarMaybeExt; STATEMENT_HASH_STATE_WIDTH],
    chain_rate_start: usize,
) -> [AB::VarMaybeExt; STATEMENT_HASH_STATE_WIDTH] {
    core::array::from_fn(|lane| {
        if lane < chain_rate_start {
            AB::zero_maybe()
        } else {
            perm_output[lane].clone()
        }
    })
}

fn canonical_chain_recv_state(
    chain_rate_carry: &[F; STATEMENT_HASH_RATE],
    perm_input: &[F; STATEMENT_HASH_STATE_WIDTH],
    mode: StatementDigestMode,
) -> [F; STATEMENT_HASH_STATE_WIDTH] {
    let chain_rate_start = statement_hash_chain_rate_start(mode);
    core::array::from_fn(|lane| {
        if lane < chain_rate_start {
            F::zero()
        } else if lane < STATEMENT_HASH_RATE {
            chain_rate_carry[lane]
        } else {
            perm_input[lane]
        }
    })
}

fn canonical_chain_send_state(
    perm_output: &[F; STATEMENT_HASH_STATE_WIDTH],
    mode: StatementDigestMode,
) -> [F; STATEMENT_HASH_STATE_WIDTH] {
    let chain_rate_start = statement_hash_chain_rate_start(mode);
    core::array::from_fn(|lane| if lane < chain_rate_start { F::zero() } else { perm_output[lane] })
}

fn hash_chain_payload<T: Clone>(
    hash_kind: T,
    block_idx: T,
    state: [T; STATEMENT_HASH_STATE_WIDTH],
) -> [T; 18] {
    core::array::from_fn(|idx| match idx {
        0 => hash_kind.clone(),
        1 => block_idx.clone(),
        _ => state[idx - 2].clone(),
    })
}

fn vk_digest_payload<T: Clone>(hash_kind: T, state: [T; STATEMENT_HASH_STATE_WIDTH]) -> [T; 9] {
    core::array::from_fn(|idx| if idx == 0 { hash_kind.clone() } else { state[idx - 1].clone() })
}

#[derive(Debug, Clone, Copy, Default)]
pub struct StatementHashTraceGenerator;

impl StatementHashTraceGenerator {
    pub fn trace_height(record: &RecursionRecord, mode: StatementDigestMode) -> usize {
        statement_hash_rows_cached(record, mode).len().max(1).next_power_of_two()
    }

    pub fn generate_trace_compressed(
        record: &RecursionRecord,
        mode: StatementDigestMode,
    ) -> CompressedMatrix<F> {
        let source_rows = statement_hash_rows_cached(record, mode);
        let height = source_rows.len().max(1).next_power_of_two();
        match mode {
            StatementDigestMode::SelfDigest => {
                let mut values =
                    vec![F::zero(); source_rows.len().max(1) * NUM_STATEMENT_HASH_COLS];
                for (target, row) in
                    values.chunks_exact_mut(NUM_STATEMENT_HASH_COLS).zip(source_rows.iter())
                {
                    write_statement_hash_row(target, row);
                }
                compressed_flat_rows(
                    values,
                    NUM_STATEMENT_HASH_COLS,
                    source_rows.len().max(1),
                    height,
                )
            }
            StatementDigestMode::RootDigest => {
                let mut values =
                    vec![F::zero(); source_rows.len().max(1) * NUM_STATEMENT_HASH_ROOT_COLS];
                for (target, row) in
                    values.chunks_exact_mut(NUM_STATEMENT_HASH_ROOT_COLS).zip(source_rows.iter())
                {
                    write_statement_hash_root_row(target, row);
                }
                compressed_flat_rows(
                    values,
                    NUM_STATEMENT_HASH_ROOT_COLS,
                    source_rows.len().max(1),
                    height,
                )
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StatementHashRow {
    pub proof_idx: usize,
    pub hash_kind: usize,
    pub block_idx: usize,
    pub is_final_block: bool,
    pub input_len: usize,
    pub vk_meta_full_block: bool,
    pub vk_meta_tail_block: bool,
    /// The largest mode-specific rate suffix which can survive absorb.
    /// Root uses only the last coordinate (state lane 7).
    pub chain_rate_carry: [F; STATEMENT_HASH_RATE],
    pub perm_input: [F; STATEMENT_HASH_STATE_WIDTH],
    pub perm_output: [F; STATEMENT_HASH_STATE_WIDTH],
}

impl StatementHashRow {
    pub fn is_vk_digest(&self) -> bool {
        self.hash_kind == STATEMENT_HASH_KIND_VK_DIGEST
    }

    pub fn is_self_digest(&self) -> bool {
        self.hash_kind == STATEMENT_HASH_KIND_SELF_DIGEST
    }

    pub fn is_root_digest(&self) -> bool {
        self.hash_kind == STATEMENT_HASH_KIND_ROOT_DIGEST
    }

    pub fn is_first_block(&self) -> bool {
        self.block_idx == 0
    }

    pub fn is_final_block(&self) -> bool {
        self.is_final_block
    }

    #[cfg(test)]
    fn input_values(&self) -> [F; STATEMENT_HASH_RATE] {
        let input_len = self.input_len;
        core::array::from_fn(|lane| {
            if lane < input_len {
                self.perm_input[lane] - self.chain_rate_carry[lane]
            } else {
                F::zero()
            }
        })
    }
}

#[cfg(test)]
fn statement_hash_rows(
    record: &RecursionRecord,
    mode: StatementDigestMode,
) -> Vec<StatementHashRow> {
    statement_hash_rows_cached(record, mode).as_ref().to_vec()
}

pub(crate) fn statement_hash_rows_cached(
    record: &RecursionRecord,
    mode: StatementDigestMode,
) -> Arc<[StatementHashRow]> {
    let (installed_mode, rows) = record
        .tracegen_artifacts
        .statement_hash
        .get_or_init(|| (mode, Arc::from(statement_hash_rows_uncached(record, mode))));
    assert_eq!(
        *installed_mode, mode,
        "one tracegen workspace was used with two statement-hash authorities"
    );
    Arc::clone(rows)
}

fn statement_hash_rows_uncached(
    record: &RecursionRecord,
    mode: StatementDigestMode,
) -> Vec<StatementHashRow> {
    let Some(statement) = record.statement_public_values else {
        return Vec::new();
    };
    if record.proof_records.is_empty() {
        return Vec::new();
    }

    let digest_blocks = match mode {
        StatementDigestMode::SelfDigest => STATEMENT_SELF_DIGEST_BLOCKS,
        StatementDigestMode::RootDigest => STATEMENT_ROOT_DIGEST_BLOCKS,
    };
    let mut rows =
        Vec::with_capacity(STATEMENT_VK_DIGEST_BLOCKS * record.proof_records.len() + digest_blocks);
    for proof in &record.proof_records {
        extend_hash_instance_rows(
            &mut rows,
            proof.proof_idx,
            STATEMENT_HASH_KIND_VK_DIGEST,
            &crate::statement_dt::child_vk_digest_input(&proof.proof_shape),
            Some(proof.proof_shape.vk_meta.len()),
            &record.poseidon2_memo,
        );
    }
    let public = statement.as_array();
    match mode {
        StatementDigestMode::SelfDigest => {
            extend_hash_instance_rows(
                &mut rows,
                0,
                STATEMENT_HASH_KIND_SELF_DIGEST,
                &statement_self_digest_hash_input(&public),
                None,
                &record.poseidon2_memo,
            );
        }
        StatementDigestMode::RootDigest => {
            extend_hash_instance_rows(
                &mut rows,
                0,
                STATEMENT_HASH_KIND_ROOT_DIGEST,
                &root_digest_hash_input(&public),
                None,
                &record.poseidon2_memo,
            );
        }
    }
    rows
}

/// Compact provider inputs for statement hashing. This walks the statement's
/// canonical sources directly and never constructs or scans `StatementHashRow`.
pub(crate) fn statement_hash_poseidon2_inputs(
    record: &RecursionRecord,
    mode: StatementDigestMode,
) -> Vec<[F; STATEMENT_HASH_STATE_WIDTH]> {
    let Some(statement) = record.statement_public_values else {
        return Vec::new();
    };
    if record.proof_records.is_empty() {
        return Vec::new();
    }
    let digest_blocks = match mode {
        StatementDigestMode::SelfDigest => STATEMENT_SELF_DIGEST_BLOCKS,
        StatementDigestMode::RootDigest => STATEMENT_ROOT_DIGEST_BLOCKS,
    };
    let mut inputs =
        Vec::with_capacity(STATEMENT_VK_DIGEST_BLOCKS * record.proof_records.len() + digest_blocks);
    for proof in &record.proof_records {
        extend_hash_instance_poseidon2_inputs(
            &mut inputs,
            &crate::statement_dt::child_vk_digest_input(&proof.proof_shape),
            &record.poseidon2_memo,
        );
    }
    let public = statement.as_array();
    match mode {
        StatementDigestMode::SelfDigest => extend_hash_instance_poseidon2_inputs(
            &mut inputs,
            &statement_self_digest_hash_input(&public),
            &record.poseidon2_memo,
        ),
        StatementDigestMode::RootDigest => extend_hash_instance_poseidon2_inputs(
            &mut inputs,
            &root_digest_hash_input(&public),
            &record.poseidon2_memo,
        ),
    }
    inputs
}

fn extend_hash_instance_poseidon2_inputs(
    inputs: &mut Vec<[F; STATEMENT_HASH_STATE_WIDTH]>,
    input: &[F],
    output: &impl RecursionPoseidon2Output,
) {
    let mut state = [F::zero(); STATEMENT_HASH_STATE_WIDTH];
    for chunk in input.chunks(STATEMENT_HASH_RATE) {
        let mut perm_input = state;
        for (lane, value) in chunk.iter().enumerate() {
            perm_input[lane] += *value;
        }
        state = output.permute_output(perm_input);
        inputs.push(perm_input);
    }
}

fn extend_hash_instance_rows(
    rows: &mut Vec<StatementHashRow>,
    proof_idx: usize,
    hash_kind: usize,
    input: &[F],
    vk_meta_value_count: Option<usize>,
    output: &impl RecursionPoseidon2Output,
) {
    let mut state = [F::zero(); STATEMENT_HASH_STATE_WIDTH];
    let block_count = input.len().div_ceil(STATEMENT_HASH_RATE);
    for (block_idx, chunk) in input.chunks(STATEMENT_HASH_RATE).enumerate() {
        let mut perm_input = state;
        for (lane, value) in chunk.iter().enumerate() {
            perm_input[lane] += *value;
        }
        let perm_output = output.permute_output(perm_input);
        let block_start = block_idx * STATEMENT_HASH_RATE;
        let vk_meta_full_block =
            vk_meta_value_count.is_some_and(|count| block_start + STATEMENT_HASH_RATE <= count);
        let vk_meta_tail_block = vk_meta_value_count
            .is_some_and(|count| block_start < count && block_start + STATEMENT_HASH_RATE > count);
        rows.push(StatementHashRow {
            proof_idx,
            hash_kind,
            block_idx,
            is_final_block: block_idx + 1 == block_count,
            input_len: chunk.len(),
            vk_meta_full_block,
            vk_meta_tail_block,
            chain_rate_carry: state[..STATEMENT_HASH_RATE]
                .try_into()
                .expect("statement hash rate width"),
            perm_input,
            perm_output,
        });
        state = perm_output;
    }
}

fn write_statement_hash_row(values: &mut [F], row: &StatementHashRow) {
    let cols: &mut StatementHashCols<F> = values.borrow_mut();
    cols.proof_idx = f(row.proof_idx);
    cols.is_final_digest = f_bool(row.is_self_digest() || row.is_root_digest());
    cols.block_flags[row.block_idx] = F::one();
    cols.chain_rate_carry = row.chain_rate_carry
        [STATEMENT_SELF_CHAIN_RATE_START..STATEMENT_HASH_RATE]
        .try_into()
        .expect("self statement hash carry width");
    cols.perm_input = row.perm_input;
    cols.perm_output = row.perm_output;
    cols.vk_meta_full_block = f_bool(row.vk_meta_full_block);
    cols.vk_meta_tail_block = f_bool(row.vk_meta_tail_block);
    cols.vk_final_block = f_bool(row.is_vk_digest() && row.is_final_block());
    cols.final_digest_block = f_bool(row.is_self_digest() && row.is_final_block());
}

fn write_statement_hash_root_row(values: &mut [F], row: &StatementHashRow) {
    let cols: &mut StatementHashRootCols<F> = values.borrow_mut();
    cols.proof_idx = f(row.proof_idx);
    cols.is_final_digest = f_bool(row.is_root_digest());
    cols.block_flags[row.block_idx] = F::one();
    cols.chain_rate_carry = row.chain_rate_carry
        [STATEMENT_ROOT_CHAIN_RATE_START..STATEMENT_HASH_RATE]
        .try_into()
        .expect("root statement hash carry width");
    cols.perm_input = row.perm_input;
    cols.perm_output = row.perm_output;
    cols.vk_meta_full_block = f_bool(row.vk_meta_full_block);
    cols.vk_meta_tail_block = f_bool(row.vk_meta_tail_block);
    cols.vk_final_block = f_bool(row.is_vk_digest() && row.is_final_block());
    cols.final_digest_block = f_bool(row.is_root_digest() && row.is_final_block());
}

pub type StatementHashBusResidualReport = BTreeMap<&'static str, BTreeMap<Vec<u32>, i64>>;

pub fn statement_hash_bus_residual_report(
    record: &RecursionRecord,
    mode: StatementDigestMode,
) -> StatementHashBusResidualReport {
    let rows = statement_hash_rows_cached(record, mode);
    let mut report = StatementHashBusResidualReport::new();
    let checks: Vec<(&'static str, BTreeMap<Vec<u32>, i64>)> = vec![
        ("1038 StatementVkDigest", statement_vk_digest_residual(record, &rows)),
        ("1039 StatementHashChain", statement_hash_chain_residual(&rows, mode)),
    ];
    for (name, residual) in checks {
        if !residual.is_empty() {
            report.insert(name, residual);
        }
    }
    report
}

fn statement_vk_digest_residual(
    record: &RecursionRecord,
    rows: &[StatementHashRow],
) -> BTreeMap<Vec<u32>, i64> {
    let mut residual = BTreeMap::new();
    for row in rows {
        if row.is_vk_digest() && row.is_final_block() {
            apply_residual(
                &mut residual,
                vk_digest_key(
                    row.proof_idx,
                    row.hash_kind,
                    row.perm_output[..DIGEST_SIZE].try_into().expect("digest width"),
                ),
                1,
            );
        }
    }
    if record.statement_public_values.is_some() {
        for proof in &record.proof_records {
            // The statement's vk_root row consumes the child's OWN vk digest (for lift
            // children this equals the exported dt_vk digest; for native children it is
            // the vk-class membership material).
            apply_residual(
                &mut residual,
                vk_digest_key(
                    proof.proof_idx,
                    STATEMENT_HASH_KIND_VK_DIGEST,
                    crate::statement_dt::child_vk_digest_with_memo(
                        &proof.proof_shape,
                        &record.poseidon2_memo,
                    ),
                ),
                -1,
            );
        }
    }
    finalize_residual(residual)
}

fn statement_hash_chain_residual(
    rows: &[StatementHashRow],
    mode: StatementDigestMode,
) -> BTreeMap<Vec<u32>, i64> {
    let mut residual = BTreeMap::new();
    for row in rows {
        apply_residual(
            &mut residual,
            hash_chain_key(
                row.proof_idx,
                row.hash_kind,
                row.block_idx,
                canonical_chain_recv_state(&row.chain_rate_carry, &row.perm_input, mode),
            ),
            -1,
        );
        if !row.is_final_block() {
            apply_residual(
                &mut residual,
                hash_chain_key(
                    row.proof_idx,
                    row.hash_kind,
                    row.block_idx + 1,
                    canonical_chain_send_state(&row.perm_output, mode),
                ),
                1,
            );
        }
        if row.is_first_block() {
            apply_residual(
                &mut residual,
                hash_chain_key(
                    row.proof_idx,
                    row.hash_kind,
                    0,
                    [F::zero(); STATEMENT_HASH_STATE_WIDTH],
                ),
                1,
            );
        }
    }
    finalize_residual(residual)
}

fn vk_digest_key(proof_idx: usize, hash_kind: usize, digest: [F; DIGEST_SIZE]) -> Vec<u32> {
    let mut key = vec![proof_idx as u32, hash_kind as u32];
    key.extend(digest.map(field_u32));
    key
}

fn hash_chain_key(
    proof_idx: usize,
    hash_kind: usize,
    block_idx: usize,
    state: [F; STATEMENT_HASH_STATE_WIDTH],
) -> Vec<u32> {
    let mut key = vec![proof_idx as u32, hash_kind as u32, block_idx as u32];
    key.extend(state.map(field_u32));
    key
}

fn apply_residual(residual: &mut BTreeMap<Vec<u32>, i64>, key: Vec<u32>, delta: i64) {
    *residual.entry(key).or_default() += delta;
}

fn finalize_residual(mut residual: BTreeMap<Vec<u32>, i64>) -> BTreeMap<Vec<u32>, i64> {
    residual.retain(|_, value| *value != 0);
    residual
}

fn compressed_flat_rows(
    values: Vec<F>,
    width: usize,
    stored_height: usize,
    total_height: usize,
) -> CompressedMatrix<F> {
    let main = RowMajorMatrix::new(values, width);
    debug_assert_eq!(main.height(), stored_height);
    let padding = if main.height() < total_height {
        PaddingRow::General(vec![F::zero(); width])
    } else {
        PaddingRow::None
    };
    CompressedMatrix::new(main, padding, total_height)
}

fn assert_bool<AB: FullAirBuilder>(builder: &mut AB, value: AB::VarMaybeExt) {
    builder.assert_zero(value.clone() * (value - AB::one_maybe()));
}

fn const_maybe<AB: FullAirBuilder>(value: usize) -> AB::VarMaybeExt {
    AB::VarMaybeExt::from(AB::F::from_canonical_usize(value))
}

fn f(value: usize) -> F {
    F::from_canonical_usize(value)
}

fn f_bool(value: bool) -> F {
    F::from_bool(value)
}

fn field_u32(value: F) -> u32 {
    value.as_canonical_u32()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::{D_EF, EF},
        statement_dt::{
            poseidon2_hash_slice, NativeRecursionPublicValues, NATIVE_PV_GLOBAL_INTERVAL_END,
            NATIVE_RECURSION_NUM_PV_ELTS,
        },
        symbolic_expr_fixed_dt::RecursionFixedSymbolicChip,
        symbolic_ir_dt::RecursionPolyAirChipIr,
        system_dt::{RecursionProofRecord, RecursionProofShapeRecord},
        validate::{
            assert_provider_requests_match_sources_for_test, finalize_provider_requests_at_source,
        },
    };
    use p3_field::AbstractExtensionField;
    use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
    use polyair::{
        evaluator::ConstraintFolder, permutation::fused_precompute_reserved_permutation, Chip,
    };

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

    fn symbolic_shape(chip: &Chip<StatementHashAir, F, D_EF>) -> SymbolicShape {
        let fixed =
            RecursionFixedSymbolicChip::from_polyair_chip(0, chip).expect("fixed StatementHash");
        let ir = RecursionPolyAirChipIr::compile(&fixed).expect("StatementHash IR");
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

    fn fixture_proof(mode: StatementDigestMode, proof_idx: usize) -> RecursionProofRecord {
        let vk_commit = core::array::from_fn(|lane| f(101 + proof_idx * 31 + lane));
        let core = mode == StatementDigestMode::SelfDigest;
        let vk_meta_value_count = if core {
            crate::proof_shape_dt::PROOF_SHAPE_CORE_VK_META_VALUE_COUNT
        } else {
            crate::proof_shape_dt::PROOF_SHAPE_NATIVE_VK_META_VALUE_COUNT
        };
        let mut vk_meta =
            (0..vk_meta_value_count).map(|lane| f(151 + proof_idx * 31 + lane)).collect::<Vec<_>>();
        vk_meta[..DIGEST_SIZE].copy_from_slice(&vk_commit);
        RecursionProofRecord {
            proof_idx,
            proof_shape: RecursionProofShapeRecord {
                role_id: usize::from(!core),
                vk_commit,
                vk_meta,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn fixture_statement(mode: StatementDigestMode) -> NativeRecursionPublicValues<F> {
        let mut values = NativeRecursionPublicValues::<F>::default();
        for (word_idx, word) in values.committed_value_digest.iter_mut().enumerate() {
            for (byte_idx, byte) in word.0.iter_mut().enumerate() {
                *byte = f(1 + 4 * word_idx + byte_idx);
            }
        }
        values.start_pc = f(41);
        values.next_pc = f(42);
        values.start_shard = f(43);
        values.next_shard = f(44);
        values.start_execution_shard = f(45);
        values.next_execution_shard = f(46);
        values.previous_init_addr = f(47);
        values.last_init_addr = f(48);
        values.previous_finalize_addr = f(49);
        values.last_finalize_addr = f(50);
        values.dt_vk_digest = core::array::from_fn(|idx| f(67 + idx));
        values.vk_root = core::array::from_fn(|idx| f(75 + idx));
        values.contains_execution_shard = F::one();
        values.digest = match mode {
            StatementDigestMode::SelfDigest => {
                poseidon2_hash_slice(&statement_self_digest_hash_input(&values.as_array()))
            }
            StatementDigestMode::RootDigest => root_public_values_digest(&values.as_array()),
        };
        values
    }

    fn fixture_record(mode: StatementDigestMode, proof_indices: &[usize]) -> RecursionRecord {
        let mut record = RecursionRecord::default();
        record.statement_public_values = Some(fixture_statement(mode));
        record.proof_records = proof_indices
            .iter()
            .copied()
            .map(|proof_idx| fixture_proof(mode, proof_idx))
            .collect::<Vec<_>>();
        record
    }

    fn test_ext(seed: usize) -> EF {
        EF::from_base_fn(|limb| F::from_canonical_usize(seed + 7 * limb + limb * limb))
    }

    fn test_beta_data(seed: usize, powers: usize) -> (Vec<EF>, EF) {
        let beta = test_ext(seed);
        let beta_powers = beta.powers().take(powers).collect::<Vec<_>>();
        let beta_septix =
            beta_powers[7] - EF::from_canonical_usize(3) * beta - EF::from_canonical_usize(5);
        assert_ne!(beta_septix, EF::zero());
        (beta_powers, beta_septix)
    }

    #[test]
    fn symbolic_analysis() {
        let chip = Chip::<StatementHashAir, F, D_EF>::new(StatementHashAir::default());
        assert_eq!(
            symbolic_shape(&chip),
            SymbolicShape {
                main: 66,
                reserved: 25,
                precomputed: 19,
                permutation: 7,
                active: 51,
                gates: 38,
                alpha: 46,
                degree: 3,
                nodes: 923,
                nodes_padded: 1024,
                roots: 64,
                roots_padded: 64,
                folds: 46,
                folds_padded: 64,
                lookups: 13,
            }
        );
        assert!(chip.required_max_beta_power() >= 34);

        let root = Chip::<StatementHashAir, F, D_EF>::new(StatementHashAir::new(
            StatementDigestMode::RootDigest,
        ));
        assert_eq!(
            symbolic_shape(&root),
            SymbolicShape {
                main: 52,
                reserved: 11,
                precomputed: 19,
                permutation: 7,
                active: 37,
                gates: 24,
                alpha: 32,
                degree: 3,
                nodes: 540,
                nodes_padded: 1024,
                roots: 50,
                roots_padded: 64,
                folds: 32,
                folds_padded: 32,
                lookups: 13,
            }
        );

        let native_self = Chip::<StatementHashAir, F, D_EF>::new(StatementHashAir::for_child(
            StatementDigestMode::SelfDigest,
            NATIVE_RECURSION_NUM_PV_ELTS,
        ));
        let native_root = Chip::<StatementHashAir, F, D_EF>::new(StatementHashAir::for_child(
            StatementDigestMode::RootDigest,
            NATIVE_RECURSION_NUM_PV_ELTS,
        ));
        assert_eq!(
            symbolic_shape(&native_self),
            SymbolicShape {
                main: 66,
                reserved: 25,
                precomputed: 19,
                permutation: 7,
                active: 51,
                gates: 38,
                alpha: 46,
                degree: 3,
                nodes: 878,
                nodes_padded: 1024,
                roots: 64,
                roots_padded: 64,
                folds: 46,
                folds_padded: 64,
                lookups: 13,
            }
        );
        assert_eq!(
            symbolic_shape(&native_root),
            SymbolicShape {
                main: 52,
                reserved: 11,
                precomputed: 19,
                permutation: 7,
                active: 37,
                gates: 24,
                alpha: 32,
                degree: 3,
                nodes: 540,
                nodes_padded: 1024,
                roots: 50,
                roots_padded: 64,
                folds: 32,
                folds_padded: 32,
                lookups: 13,
            }
        );
        assert_eq!(native_self.air.vk_digest_input_count.div_ceil(STATEMENT_HASH_RATE), 5);
        assert_eq!(native_root.air.vk_digest_input_count.div_ceil(STATEMENT_HASH_RATE), 5);
    }

    #[test]
    fn committed_reserved_and_lookup_layouts_are_exact_per_mode() {
        assert_eq!(
            [
                ("proof_idx", core::mem::offset_of!(StatementHashCols<u8>, proof_idx)),
                ("is_final_digest", core::mem::offset_of!(StatementHashCols<u8>, is_final_digest),),
                ("block_flags", core::mem::offset_of!(StatementHashCols<u8>, block_flags),),
                (
                    "chain_rate_carry",
                    core::mem::offset_of!(StatementHashCols<u8>, chain_rate_carry),
                ),
                ("perm_input", core::mem::offset_of!(StatementHashCols<u8>, perm_input),),
                ("perm_output", core::mem::offset_of!(StatementHashCols<u8>, perm_output),),
                (
                    "vk_meta_full_block",
                    core::mem::offset_of!(StatementHashCols<u8>, vk_meta_full_block),
                ),
                (
                    "vk_meta_tail_block",
                    core::mem::offset_of!(StatementHashCols<u8>, vk_meta_tail_block),
                ),
                ("vk_final_block", core::mem::offset_of!(StatementHashCols<u8>, vk_final_block),),
                (
                    "final_digest_block",
                    core::mem::offset_of!(StatementHashCols<u8>, final_digest_block),
                ),
            ],
            [
                ("proof_idx", 0),
                ("is_final_digest", 1),
                ("block_flags", 2),
                ("chain_rate_carry", 22),
                ("perm_input", 30),
                ("perm_output", 46),
                ("vk_meta_full_block", 62),
                ("vk_meta_tail_block", 63),
                ("vk_final_block", 64),
                ("final_digest_block", 65),
            ]
        );
        assert_eq!(
            [
                ("proof_idx", core::mem::offset_of!(StatementHashRootCols<u8>, proof_idx),),
                (
                    "is_final_digest",
                    core::mem::offset_of!(StatementHashRootCols<u8>, is_final_digest),
                ),
                ("block_flags", core::mem::offset_of!(StatementHashRootCols<u8>, block_flags),),
                (
                    "chain_rate_carry",
                    core::mem::offset_of!(StatementHashRootCols<u8>, chain_rate_carry),
                ),
                ("perm_input", core::mem::offset_of!(StatementHashRootCols<u8>, perm_input),),
                ("perm_output", core::mem::offset_of!(StatementHashRootCols<u8>, perm_output),),
                (
                    "vk_meta_full_block",
                    core::mem::offset_of!(StatementHashRootCols<u8>, vk_meta_full_block),
                ),
                (
                    "vk_meta_tail_block",
                    core::mem::offset_of!(StatementHashRootCols<u8>, vk_meta_tail_block),
                ),
                (
                    "vk_final_block",
                    core::mem::offset_of!(StatementHashRootCols<u8>, vk_final_block),
                ),
                (
                    "final_digest_block",
                    core::mem::offset_of!(StatementHashRootCols<u8>, final_digest_block),
                ),
            ],
            [
                ("proof_idx", 0),
                ("is_final_digest", 1),
                ("block_flags", 2),
                ("chain_rate_carry", 8),
                ("perm_input", 16),
                ("perm_output", 32),
                ("vk_meta_full_block", 48),
                ("vk_meta_tail_block", 49),
                ("vk_final_block", 50),
                ("final_digest_block", 51),
            ]
        );

        for mode in [StatementDigestMode::SelfDigest, StatementDigestMode::RootDigest] {
            let chip = Chip::<StatementHashAir, F, D_EF>::new(StatementHashAir::new(mode));
            let expected_reserved = match mode {
                StatementDigestMode::SelfDigest => {
                    (1..22).chain(62..66).map(PairCol::Main).collect::<Vec<_>>()
                }
                StatementDigestMode::RootDigest => {
                    (1..8).chain(48..52).map(PairCol::Main).collect::<Vec<_>>()
                }
            };
            assert_eq!(chip.symbolic_builder.reserved_poly_output, expected_reserved);
            assert_eq!(
                chip.symbolic_builder
                    .lookup_infos
                    .iter()
                    .map(|lookup| lookup.is_send)
                    .collect::<Vec<_>>(),
                [
                    false, true, true, false, false, false, false, false, false, false, false,
                    false, true,
                ]
            );
        }
    }

    #[test]
    fn root_digest_input_is_dt_vk_then_committed_values() {
        let idxs = root_digest_input_pv_indices();
        assert_eq!(idxs.len(), 40);
        assert_eq!(&idxs[..DIGEST_SIZE], &[66, 67, 68, 69, 70, 71, 72, 73]);
        assert_eq!(idxs[DIGEST_SIZE], 0);
        assert_eq!(idxs[STATEMENT_ROOT_DIGEST_INPUT_ELTS - 1], 31);
        assert_eq!(STATEMENT_ROOT_DIGEST_INPUT_ELTS % STATEMENT_HASH_RATE, 0);
        assert_eq!(
            STATEMENT_ROOT_DIGEST_INPUT_ELTS / STATEMENT_HASH_RATE,
            STATEMENT_ROOT_DIGEST_BLOCKS
        );
    }

    #[test]
    fn hash_row_count_constants_match_m0_inputs() {
        assert_eq!(STATEMENT_VK_DIGEST_BLOCKS, 8);
        assert_eq!(NATIVE_RECURSION_NUM_PV_ELMS_TO_HASH.div_ceil(STATEMENT_HASH_RATE), 19);
        assert_eq!(STATEMENT_SELF_FINAL_INPUT_LANES, 8);
        assert_eq!(STATEMENT_SELF_CHAIN_RATE_LANES, 8);
        assert_eq!(STATEMENT_ROOT_BLOCK_SELECTORS, 6);
        assert_eq!(STATEMENT_ROOT_CHAIN_RATE_LANES, 8);
        assert_eq!(crate::statement_dt::NATIVE_RECURSION_NUM_PV_ELTS, 159);
    }

    #[test]
    fn self_digest_preimage_has_frozen_domain_and_delimiter() {
        let public: [F; crate::statement_dt::NATIVE_RECURSION_NUM_PV_ELTS] =
            core::array::from_fn(|idx| f(idx + 17));
        let input = statement_self_digest_hash_input(&public);
        assert_eq!(input.len(), 160);
        assert_eq!(input[0], F::from_canonical_u32(STATEMENT_SELF_DIGEST_DOMAIN[0]));
        assert_eq!(input[1], F::from_canonical_u32(STATEMENT_SELF_DIGEST_DOMAIN[1]));
        assert_eq!(&input[2..153], &public[..NATIVE_RECURSION_NUM_PV_ELMS_TO_HASH]);
        assert_eq!(&input[153..159], &[F::zero(); 6]);
        assert_eq!(input[159], F::one());

        let digest = poseidon2_hash_slice(&input);
        let mut mutated = public;
        mutated[NATIVE_PV_GLOBAL_INTERVAL_END] += F::one();
        assert_ne!(digest, poseidon2_hash_slice(&statement_self_digest_hash_input(&mutated)));
    }

    fn assert_committed_row(mode: StatementDigestMode, actual: &[F], semantic: &StatementHashRow) {
        match mode {
            StatementDigestMode::SelfDigest => {
                let cols: &StatementHashCols<F> = actual.borrow();
                assert_eq!(cols.proof_idx, f(semantic.proof_idx));
                assert_eq!(cols.is_final_digest, f_bool(semantic.is_self_digest()));
                assert_eq!(
                    cols.block_flags,
                    core::array::from_fn(|idx| f_bool(idx == semantic.block_idx))
                );
                assert_eq!(cols.chain_rate_carry, semantic.chain_rate_carry);
                assert_eq!(cols.perm_input, semantic.perm_input);
                assert_eq!(cols.perm_output, semantic.perm_output);
                assert_eq!(
                    cols.vk_final_block,
                    f_bool(semantic.is_vk_digest() && semantic.is_final_block())
                );
                assert_eq!(
                    cols.final_digest_block,
                    f_bool(semantic.is_self_digest() && semantic.is_final_block())
                );
            }
            StatementDigestMode::RootDigest => {
                let cols: &StatementHashRootCols<F> = actual.borrow();
                assert_eq!(cols.proof_idx, f(semantic.proof_idx));
                assert_eq!(cols.is_final_digest, f_bool(semantic.is_root_digest()));
                assert_eq!(
                    cols.block_flags,
                    core::array::from_fn(|idx| f_bool(idx == semantic.block_idx))
                );
                assert_eq!(cols.chain_rate_carry, semantic.chain_rate_carry);
                assert_eq!(cols.perm_input, semantic.perm_input);
                assert_eq!(cols.perm_output, semantic.perm_output);
                assert_eq!(
                    cols.vk_final_block,
                    f_bool(semantic.is_vk_digest() && semantic.is_final_block())
                );
                assert_eq!(
                    cols.final_digest_block,
                    f_bool(semantic.is_root_digest() && semantic.is_final_block())
                );
            }
        }
    }

    #[test]
    fn trace_rows_preserve_proof_block_lane_and_padding_contracts() {
        for mode in [StatementDigestMode::SelfDigest, StatementDigestMode::RootDigest] {
            for proof_count in [1usize, 11] {
                let proof_indices =
                    (0..proof_count).map(|idx| 100 + proof_count - idx).collect::<Vec<_>>();
                let record = fixture_record(mode, &proof_indices);
                let rows = statement_hash_rows_cached(&record, mode);
                let digest_blocks = match mode {
                    StatementDigestMode::SelfDigest => STATEMENT_SELF_DIGEST_BLOCKS,
                    StatementDigestMode::RootDigest => STATEMENT_ROOT_DIGEST_BLOCKS,
                };
                let vk_blocks =
                    StatementHashAir::new(mode).vk_digest_input_count.div_ceil(STATEMENT_HASH_RATE);
                let logical_rows = vk_blocks * proof_count + digest_blocks;
                assert_eq!(rows.len(), logical_rows);
                for (proof_ord, proof_idx) in proof_indices.iter().copied().enumerate() {
                    let start = proof_ord * vk_blocks;
                    for block_idx in 0..vk_blocks {
                        let row = &rows[start + block_idx];
                        assert_eq!(row.proof_idx, proof_idx);
                        assert!(row.is_vk_digest());
                        assert_eq!(row.block_idx, block_idx);
                    }
                    let remainder =
                        StatementHashAir::new(mode).vk_digest_input_count % STATEMENT_HASH_RATE;
                    if remainder != 0 {
                        assert!(rows[start + vk_blocks - 1].input_values()[remainder..]
                            .iter()
                            .all(|value| *value == F::zero()));
                    }
                }
                let final_rows = &rows[vk_blocks * proof_count..];
                for (block_idx, row) in final_rows.iter().enumerate() {
                    assert_eq!(row.proof_idx, 0);
                    assert_eq!(row.block_idx, block_idx);
                    assert_eq!(row.is_self_digest(), mode == StatementDigestMode::SelfDigest);
                    assert_eq!(row.is_root_digest(), mode == StatementDigestMode::RootDigest);
                    if block_idx > 0 {
                        assert_eq!(
                            &row.chain_rate_carry[statement_hash_chain_rate_start(mode)..],
                            &final_rows[block_idx - 1].perm_output
                                [statement_hash_chain_rate_start(mode)..STATEMENT_HASH_RATE]
                        );
                    }
                }
                if mode == StatementDigestMode::SelfDigest {
                    assert_eq!(
                        final_rows.last().expect("self final row").input_values()
                            [STATEMENT_SELF_FINAL_INPUT_LANES..],
                        [F::zero(); STATEMENT_HASH_RATE - STATEMENT_SELF_FINAL_INPUT_LANES]
                    );
                }

                let trace = StatementHashTraceGenerator::generate_trace_compressed(&record, mode);
                assert_eq!(trace.main.height(), logical_rows);
                assert_eq!(trace.total_height, logical_rows.max(1).next_power_of_two());
                assert_eq!(
                    trace.main.width(),
                    match mode {
                        StatementDigestMode::SelfDigest => NUM_STATEMENT_HASH_COLS,
                        StatementDigestMode::RootDigest => NUM_STATEMENT_HASH_ROOT_COLS,
                    }
                );
                for (row_idx, semantic) in rows.iter().enumerate() {
                    assert_committed_row(mode, trace.main.row_slice(row_idx).as_ref(), semantic);
                }
                for row_idx in logical_rows..trace.total_height {
                    assert!(
                        trace.row_slice(row_idx).iter().all(|value| *value == F::zero()),
                        "{mode:?} padding row {row_idx}"
                    );
                }
            }
        }

        for mode in [StatementDigestMode::SelfDigest, StatementDigestMode::RootDigest] {
            let empty = RecursionRecord::default();
            let trace = StatementHashTraceGenerator::generate_trace_compressed(&empty, mode);
            assert_eq!(trace.main.height(), 1);
            assert_eq!(trace.total_height, 1);
            assert!(trace.row_slice(0).iter().all(|value| *value == F::zero()));
        }
    }

    #[test]
    fn statement_hash_rows_are_record_authoritative_output_only_and_single_flight() {
        let mut record = RecursionRecord::default();
        record.statement_public_values = Some(Default::default());
        record.proof_records.push(fixture_proof(StatementDigestMode::SelfDigest, 0));

        let first = statement_hash_rows_cached(&record, StatementDigestMode::SelfDigest);
        let memo_after_first = record.poseidon2_memo.snapshot();
        assert!(memo_after_first.misses > 0);
        assert_eq!(record.poseidon2_tracegen.generated_rows(), 0);
        let second = statement_hash_rows_cached(&record, StatementDigestMode::SelfDigest);
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(record.poseidon2_memo.snapshot(), memo_after_first);
        assert_eq!(record.poseidon2_tracegen.generated_rows(), 0);
        assert_eq!(first.len(), STATEMENT_VK_DIGEST_BLOCKS + STATEMENT_SELF_DIGEST_BLOCKS);
        assert_eq!(
            StatementHashTraceGenerator::trace_height(&record, StatementDigestMode::SelfDigest),
            32
        );

        let mut deps = RecursionRecord::default();
        StatementHashAir::new(StatementDigestMode::SelfDigest)
            .generate_dependencies(&record, &mut deps);
        assert_eq!(deps.poseidon2.total_count_usize(), first.len());
        let after_dependencies =
            statement_hash_rows_cached(&record, StatementDigestMode::SelfDigest);
        assert!(Arc::ptr_eq(&first, &after_dependencies));

        let cloned = record.clone();
        assert_eq!(cloned.poseidon2_tracegen.generated_rows(), 0);
        let rebuilt = statement_hash_rows_cached(&cloned, StatementDigestMode::SelfDigest);
        assert!(!Arc::ptr_eq(&first, &rebuilt));
        assert_eq!(first.as_ref(), rebuilt.as_ref());
        assert_eq!(cloned.poseidon2_tracegen.generated_rows(), 0);
        assert!(cloned.poseidon2_memo.snapshot().misses > 0);

        let late_mutation = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = record.proof_record_mut(0);
        }));
        assert!(late_mutation.is_err(), "installed workspace artifacts must never be invalidated");
        let still_installed = statement_hash_rows_cached(&record, StatementDigestMode::SelfDigest);
        assert!(Arc::ptr_eq(&first, &still_installed));
    }

    #[test]
    fn root_statement_hash_cache_contains_only_root_instance() {
        let mut record = RecursionRecord::default();
        record.statement_public_values = Some(Default::default());
        record.proof_records.push(fixture_proof(StatementDigestMode::RootDigest, 0));

        let rows = statement_hash_rows_cached(&record, StatementDigestMode::RootDigest);
        let vk_blocks = StatementHashAir::new(StatementDigestMode::RootDigest)
            .vk_digest_input_count
            .div_ceil(STATEMENT_HASH_RATE);
        assert_eq!(rows.len(), vk_blocks + STATEMENT_ROOT_DIGEST_BLOCKS);
        assert_eq!(rows.iter().filter(|row| row.is_root_digest()).count(), 5);
        assert!(!rows.iter().any(StatementHashRow::is_self_digest));
        assert_eq!(
            StatementHashTraceGenerator::trace_height(&record, StatementDigestMode::RootDigest),
            16
        );
    }

    fn canonical_provider_inputs(record: &RecursionRecord) -> Vec<[u32; POSEIDON2_WIDTH]> {
        let mut inputs = record
            .poseidon2
            .requests()
            .flat_map(|request| {
                core::iter::repeat_n(
                    request.input.map(field_u32),
                    usize::try_from(request.count).expect("request count"),
                )
            })
            .collect::<Vec<_>>();
        inputs.sort_unstable();
        inputs
    }

    #[test]
    fn preimages_provider_dependencies_and_row_cache_are_field_exact() {
        for mode in [StatementDigestMode::SelfDigest, StatementDigestMode::RootDigest] {
            let mut record = fixture_record(mode, &[9, 4, 17]);
            let rows = statement_hash_rows_cached(&record, mode);
            let vk_digest_input_count = StatementHashAir::new(mode).vk_digest_input_count;
            let vk_blocks = vk_digest_input_count.div_ceil(STATEMENT_HASH_RATE);
            for (proof_ord, proof) in record.proof_records.iter().enumerate() {
                let start = proof_ord * vk_blocks;
                let flattened = rows[start..start + vk_blocks]
                    .iter()
                    .flat_map(StatementHashRow::input_values)
                    .take(vk_digest_input_count)
                    .collect::<Vec<_>>();
                assert_eq!(
                    flattened,
                    crate::statement_dt::child_vk_digest_input(&proof.proof_shape)
                );
            }
            let final_rows = &rows[record.proof_records.len() * vk_blocks..];
            let actual_final =
                final_rows.iter().flat_map(StatementHashRow::input_values).collect::<Vec<_>>();
            let public = record.statement_public_values.expect("fixture statement").as_array();
            let expected_final = match mode {
                StatementDigestMode::SelfDigest => {
                    statement_self_digest_hash_input(&public).to_vec()
                }
                StatementDigestMode::RootDigest => root_digest_hash_input(&public).to_vec(),
            };
            assert_eq!(&actual_final[..expected_final.len()], expected_final);
            assert!(actual_final[expected_final.len()..].iter().all(|value| *value == F::zero()));

            let row_inputs = rows.iter().map(|row| row.perm_input).collect::<Vec<_>>();
            let direct_inputs = statement_hash_poseidon2_inputs(&record, mode);
            assert_eq!(direct_inputs, row_inputs);
            let misses_after_rows = record.poseidon2_memo.snapshot().misses;

            let mut dependencies = RecursionRecord::default();
            StatementHashAir::new(mode).generate_dependencies(&record, &mut dependencies);
            let mut expected_canonical =
                direct_inputs.iter().map(|input| input.map(field_u32)).collect::<Vec<_>>();
            expected_canonical.sort_unstable();
            assert_eq!(canonical_provider_inputs(&dependencies), expected_canonical);

            finalize_provider_requests_at_source(&mut record, mode);
            assert_eq!(canonical_provider_inputs(&record), expected_canonical);
            assert_provider_requests_match_sources_for_test(&record, mode);
            assert_eq!(record.poseidon2_memo.snapshot().misses, misses_after_rows);
            let after_finalize = statement_hash_rows_cached(&record, mode);
            assert!(Arc::ptr_eq(&rows, &after_finalize));
        }
    }

    #[test]
    fn hash_chain_and_vk_digest_residuals_accept_honest_rows_and_reject_tampering() {
        for mode in [StatementDigestMode::SelfDigest, StatementDigestMode::RootDigest] {
            let record = fixture_record(mode, &[3, 8]);
            let rows = statement_hash_rows(&record, mode);
            assert!(statement_hash_chain_residual(&rows, mode).is_empty());
            assert!(statement_vk_digest_residual(&record, &rows).is_empty());
            assert!(statement_hash_bus_residual_report(&record, mode).is_empty());

            let mut broken_chain = rows.clone();
            if mode == StatementDigestMode::SelfDigest {
                broken_chain[1].perm_input[STATEMENT_HASH_RATE] += F::one();
            } else {
                broken_chain[1].chain_rate_carry[STATEMENT_ROOT_CHAIN_RATE_START] += F::one();
            }
            assert!(!statement_hash_chain_residual(&broken_chain, mode).is_empty());

            let mut broken_seed = rows.clone();
            if mode == StatementDigestMode::SelfDigest {
                broken_seed[0].perm_input[STATEMENT_HASH_RATE] += F::one();
            } else {
                broken_seed[0].chain_rate_carry[STATEMENT_ROOT_CHAIN_RATE_START] += F::one();
            }
            assert!(!statement_hash_chain_residual(&broken_seed, mode).is_empty());

            let mut broken_vk = rows;
            let vk_blocks =
                StatementHashAir::new(mode).vk_digest_input_count.div_ceil(STATEMENT_HASH_RATE);
            broken_vk[vk_blocks - 1].perm_output[2] += F::one();
            assert!(!statement_vk_digest_residual(&record, &broken_vk).is_empty());
        }
    }

    #[derive(Debug)]
    struct RowEvaluation {
        first: EF,
        nonfirst: EF,
        lookup_multiplicities: Vec<F>,
    }

    fn materialized_evaluations(
        mode: StatementDigestMode,
        main: &CompressedMatrix<F>,
        public: &[F],
    ) -> Vec<RowEvaluation> {
        let chip = Chip::<StatementHashAir, F, D_EF>::new(StatementHashAir::new(mode));
        let alpha = test_ext(211);
        let (beta_powers, beta_septix) = test_beta_data(223, chip.required_max_beta_power() + 1);
        let (precomputed, reserved, permutation, local_sum) = fused_precompute_reserved_permutation(
            &chip.air,
            None,
            main,
            public,
            alpha,
            &beta_powers,
            beta_septix,
            chip.num_precompute(),
            chip.reserved_poly(),
            chip.logup_batch_size(),
            chip.num_lookup(),
        );
        let reducers = (0..chip.num_alpha).map(|idx| test_ext(307 + idx * 13)).collect::<Vec<_>>();
        let reserved_ext = RowMajorMatrix::new(
            reserved.main.values.iter().copied().map(EF::from_base).collect(),
            reserved.main.width(),
        );
        let mut evaluations = Vec::with_capacity(reserved.stored_height());
        for row_idx in 0..reserved.stored_height() {
            let precomputed_row = precomputed.main.row_slice(row_idx);
            let reserved_row = reserved.main.row_slice(row_idx);
            let permutation_row = permutation.main.row_slice(row_idx);
            let mut first_accumulator = EF::zero();
            let mut first = ConstraintFolder::<F, F, EF> {
                public,
                alpha,
                beta_powers: &beta_powers,
                beta_septix,
                precomputed: RowMajorMatrixView::new_row(precomputed_row.as_ref()),
                reserved_poly: RowMajorMatrixView::new_row(reserved_row.as_ref()),
                is_first_row: F::zero(),
                is_last_row: F::zero(),
                local_sum,
                permutation: RowMajorMatrixView::new_row(permutation_row.as_ref()),
                multiplicitys: Vec::new(),
                batch_size: chip.logup_batch_size(),
                accumulator: &mut first_accumulator,
                constraint_reducer: &reducers,
                constraint_index: 0,
            };
            chip.air.eval(&mut first);
            chip.air.lookup(&mut first);
            let lookup_multiplicities = first.multiplicitys.clone();
            first.constrain_lookup();

            let reserved_ext_row = reserved_ext.row_slice(row_idx);
            let mut nonfirst_accumulator = EF::zero();
            let mut nonfirst = ConstraintFolder::<F, EF, EF> {
                public,
                alpha,
                beta_powers: &beta_powers,
                beta_septix,
                precomputed: RowMajorMatrixView::new_row(precomputed_row.as_ref()),
                reserved_poly: RowMajorMatrixView::new_row(reserved_ext_row.as_ref()),
                is_first_row: EF::zero(),
                is_last_row: EF::zero(),
                local_sum,
                permutation: RowMajorMatrixView::new_row(permutation_row.as_ref()),
                multiplicitys: Vec::new(),
                batch_size: chip.logup_batch_size(),
                accumulator: &mut nonfirst_accumulator,
                constraint_reducer: &reducers,
                constraint_index: 0,
            };
            chip.air.eval(&mut nonfirst);
            chip.air.lookup(&mut nonfirst);
            nonfirst.constrain_lookup();
            evaluations.push(RowEvaluation {
                first: first_accumulator,
                nonfirst: nonfirst_accumulator,
                lookup_multiplicities,
            });
        }

        if reserved.stored_height() < reserved.total_height {
            let row_idx = reserved.stored_height();
            let precomputed_row = precomputed.row_slice(row_idx);
            let reserved_row = reserved.row_slice(row_idx);
            let permutation_row = permutation.row_slice(row_idx);
            let mut accumulator = EF::zero();
            let mut padding = ConstraintFolder::<F, F, EF> {
                public,
                alpha,
                beta_powers: &beta_powers,
                beta_septix,
                precomputed: RowMajorMatrixView::new_row(precomputed_row.as_ref()),
                reserved_poly: RowMajorMatrixView::new_row(reserved_row.as_ref()),
                is_first_row: F::zero(),
                is_last_row: F::zero(),
                local_sum,
                permutation: RowMajorMatrixView::new_row(permutation_row.as_ref()),
                multiplicitys: Vec::new(),
                batch_size: chip.logup_batch_size(),
                accumulator: &mut accumulator,
                constraint_reducer: &reducers,
                constraint_index: 0,
            };
            chip.air.eval(&mut padding);
            chip.air.lookup(&mut padding);
            assert!(padding.multiplicitys.iter().all(|value| *value == F::zero()));
            padding.constrain_lookup();
            assert_eq!(accumulator, EF::zero(), "{mode:?} padding");
        }
        evaluations
    }

    fn expected_lookup_multiplicities(
        row: &StatementHashRow,
        vk_meta_value_count: usize,
    ) -> Vec<F> {
        let is_vk = row.is_vk_digest();
        let vk_final = is_vk && row.is_final_block();
        let mut expected =
            vec![-F::one(), f_bool(!row.is_final_block()), f_bool(row.is_first_block())];
        for lane in 0..STATEMENT_HASH_RATE {
            expected.push(-f_bool(
                is_vk && row.block_idx * STATEMENT_HASH_RATE + lane < vk_meta_value_count,
            ));
        }
        expected.push(-F::one());
        expected.push(f_bool(vk_final));
        expected
    }

    #[test]
    fn actual_trace_materialization_evaluates_honestly_and_exposes_exact_lookup_positions() {
        for mode in [StatementDigestMode::SelfDigest, StatementDigestMode::RootDigest] {
            let record = fixture_record(mode, &[12, 2]);
            let public =
                record.statement_public_values.expect("fixture statement").as_array().to_vec();
            let rows = statement_hash_rows(&record, mode);
            let vk_meta_value_count = StatementHashAir::new(mode).vk_meta_value_count;
            let main = StatementHashTraceGenerator::generate_trace_compressed(&record, mode);
            let evaluations = materialized_evaluations(mode, &main, &public);
            assert_eq!(evaluations.len(), rows.len());
            for (row_idx, (evaluation, row)) in evaluations.iter().zip(rows.iter()).enumerate() {
                assert_eq!(evaluation.first, EF::zero(), "{mode:?} first row {row_idx}");
                assert_eq!(evaluation.nonfirst, EF::zero(), "{mode:?} nonfirst row {row_idx}");
                assert_eq!(
                    evaluation.lookup_multiplicities,
                    expected_lookup_multiplicities(row, vk_meta_value_count),
                    "{mode:?} lookup row {row_idx}"
                );
            }
        }
    }

    #[test]
    fn selector_index_and_lane_tampering_fail_first_and_nonfirst_evaluation() {
        for mode in [StatementDigestMode::SelfDigest, StatementDigestMode::RootDigest] {
            let record = fixture_record(mode, &[5, 1]);
            let public =
                record.statement_public_values.expect("fixture statement").as_array().to_vec();
            let honest = StatementHashTraceGenerator::generate_trace_compressed(&record, mode);
            let width = honest.main.width();

            let mut bad_selector = honest.clone();
            let block_flags_offset = match mode {
                StatementDigestMode::SelfDigest => {
                    core::mem::offset_of!(StatementHashCols<u8>, block_flags)
                }
                StatementDigestMode::RootDigest => {
                    core::mem::offset_of!(StatementHashRootCols<u8>, block_flags)
                }
            };
            bad_selector.main.values[block_flags_offset + 1] = F::one();

            let mut bad_vk_final = honest.clone();
            let vk_final_offset = width - 1;
            bad_vk_final.main.values[vk_final_offset] = F::one();

            let mut bad_public = public.clone();
            let public_idx = match mode {
                StatementDigestMode::SelfDigest => 7 * STATEMENT_HASH_RATE + 6,
                StatementDigestMode::RootDigest => {
                    root_digest_input_pv_indices()[STATEMENT_ROOT_DIGEST_INPUT_ELTS - 1]
                }
            };
            bad_public[public_idx] += F::one();

            let mut bad_tail = honest.clone();
            let vk_blocks =
                StatementHashAir::new(mode).vk_digest_input_count.div_ceil(STATEMENT_HASH_RATE);
            let final_row = match mode {
                StatementDigestMode::SelfDigest => 2 * vk_blocks + STATEMENT_SELF_DIGEST_BLOCKS - 1,
                StatementDigestMode::RootDigest => 2 * vk_blocks + STATEMENT_ROOT_DIGEST_BLOCKS - 1,
            };
            let perm_input_offset = match mode {
                StatementDigestMode::SelfDigest => {
                    core::mem::offset_of!(StatementHashCols<u8>, perm_input)
                }
                StatementDigestMode::RootDigest => {
                    core::mem::offset_of!(StatementHashRootCols<u8>, perm_input)
                }
            };
            bad_tail.main.values
                [final_row * width + perm_input_offset + STATEMENT_HASH_RATE - 1] += F::one();

            for (label, tampered, tampered_public) in [
                ("selector", bad_selector, public.clone()),
                ("vk-final", bad_vk_final, public.clone()),
                ("public-only", honest.clone(), bad_public),
                ("final-input", bad_tail, public.clone()),
            ] {
                let evaluations = materialized_evaluations(mode, &tampered, &tampered_public);
                assert!(
                    evaluations.iter().any(|evaluation| evaluation.first != EF::zero()),
                    "{mode:?} {label} first"
                );
                assert!(
                    evaluations.iter().any(|evaluation| evaluation.nonfirst != EF::zero()),
                    "{mode:?} {label} nonfirst"
                );
            }
        }
    }
}
