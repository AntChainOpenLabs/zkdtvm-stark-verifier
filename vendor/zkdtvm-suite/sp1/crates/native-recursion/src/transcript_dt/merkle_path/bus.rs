use dt_stark::air::FullAirBuilder;
use p3_field::AbstractField;

use crate::{
    interaction_full_air_dt::RecursionFullAirBus,
    interaction_registry_dt::{
        MERKLE_COMMITMENT_ROOT_SCHEMA, MERKLE_DIGEST_CHAIN_SCHEMA, MERKLE_LEAF_BLOCK_SCHEMA,
        MERKLE_SPONGE_STATE_CHAIN_SCHEMA,
    },
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MerkleDigestChainBus {
    bus: RecursionFullAirBus,
}

impl MerkleDigestChainBus {
    pub const fn new() -> Self {
        Self { bus: RecursionFullAirBus::new(MERKLE_DIGEST_CHAIN_SCHEMA) }
    }

    pub const fn required_max_beta_power_floor(&self) -> usize {
        self.bus.required_max_beta_power_floor()
    }

    pub fn denominator<AB>(
        &self,
        builder: &AB,
        proof_idx: AB::VarMaybeExt,
        commit_id: AB::VarMaybeExt,
        level: AB::VarMaybeExt,
        digest: [AB::VarMaybeExt; 8],
        idx: AB::VarMaybeExt,
    ) -> AB::VarExt
    where
        AB: FullAirBuilder,
    {
        let mut values = Vec::with_capacity(11);
        values.push(commit_id);
        values.push(level);
        values.extend(digest);
        values.push(idx);
        self.bus.denominator_for_proof(builder, proof_idx, values)
    }
}

impl Default for MerkleDigestChainBus {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MerkleSpongeStateChainBus {
    bus: RecursionFullAirBus,
}

impl MerkleSpongeStateChainBus {
    pub const fn new() -> Self {
        Self { bus: RecursionFullAirBus::new(MERKLE_SPONGE_STATE_CHAIN_SCHEMA) }
    }

    pub const fn required_max_beta_power_floor(&self) -> usize {
        self.bus.required_max_beta_power_floor()
    }

    pub fn denominator<AB>(
        &self,
        builder: &AB,
        proof_idx: AB::VarMaybeExt,
        unit_key: AB::VarMaybeExt,
        idx: AB::VarMaybeExt,
        block_idx: AB::VarMaybeExt,
        state: [AB::VarMaybeExt; 16],
    ) -> AB::VarExt
    where
        AB: FullAirBuilder,
    {
        let mut values = Vec::with_capacity(19);
        values.push(unit_key);
        values.push(idx);
        values.push(block_idx);
        values.extend(state);
        self.bus.denominator_for_proof(builder, proof_idx, values)
    }
}

impl Default for MerkleSpongeStateChainBus {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MerkleCommitmentRootBus {
    bus: RecursionFullAirBus,
}

impl MerkleCommitmentRootBus {
    pub const fn new() -> Self {
        Self { bus: RecursionFullAirBus::new(MERKLE_COMMITMENT_ROOT_SCHEMA) }
    }

    pub const fn required_max_beta_power_floor(&self) -> usize {
        self.bus.required_max_beta_power_floor()
    }

    pub fn denominator<AB>(
        &self,
        builder: &AB,
        proof_idx: AB::VarMaybeExt,
        commit_id: AB::VarMaybeExt,
        root: [AB::VarMaybeExt; 8],
    ) -> AB::VarExt
    where
        AB: FullAirBuilder,
    {
        let mut values = Vec::with_capacity(9);
        values.push(commit_id);
        values.extend(root);
        self.bus.denominator_for_proof(builder, proof_idx, values)
    }
}

impl Default for MerkleCommitmentRootBus {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MerkleLeafBlockBus {
    bus: RecursionFullAirBus,
}

impl MerkleLeafBlockBus {
    pub const fn new() -> Self {
        Self { bus: RecursionFullAirBus::new(MERKLE_LEAF_BLOCK_SCHEMA) }
    }

    pub const fn required_max_beta_power_floor(&self) -> usize {
        self.bus.required_max_beta_power_floor()
    }

    pub fn denominator<AB>(
        &self,
        builder: &AB,
        proof_idx: AB::VarMaybeExt,
        commit_id: AB::VarMaybeExt,
        unit_key: AB::VarMaybeExt,
        idx: AB::VarMaybeExt,
        block_idx: AB::VarMaybeExt,
        mask: [AB::VarMaybeExt; 8],
        chunk: [AB::VarMaybeExt; 8],
    ) -> AB::VarExt
    where
        AB: FullAirBuilder,
    {
        // The 8 boolean mask slots fold into one affine bitmask
        // (sum 2^i * mask_i, degree-1); both sides build the payload through
        // this shared helper.
        // Note: the folding is injective only while the mask bits are
        // constrained boolean.
        let mut mask_bits = AB::zero_maybe();
        for (i, bit) in mask.into_iter().enumerate() {
            mask_bits =
                mask_bits + bit * AB::VarMaybeExt::from(AB::F::from_canonical_usize(1 << i));
        }
        self.denominator_with_mask_bitset(
            builder, proof_idx, commit_id, unit_key, idx, block_idx, mask_bits, chunk,
        )
    }

    /// Builds a leaf-block denominator from an already-folded mask.
    ///
    /// The caller must constrain `mask_bits` to the exact little-endian
    /// bitset of eight boolean element/limb masks before using this helper.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn denominator_with_mask_bitset<AB>(
        &self,
        builder: &AB,
        proof_idx: AB::VarMaybeExt,
        commit_id: AB::VarMaybeExt,
        unit_key: AB::VarMaybeExt,
        idx: AB::VarMaybeExt,
        block_idx: AB::VarMaybeExt,
        mask_bits: AB::VarMaybeExt,
        chunk: [AB::VarMaybeExt; 8],
    ) -> AB::VarExt
    where
        AB: FullAirBuilder,
    {
        let mut values = Vec::with_capacity(13);
        values.push(commit_id);
        values.push(unit_key);
        values.push(idx);
        values.push(block_idx);
        values.push(mask_bits);
        values.extend(chunk);
        self.bus.denominator_for_proof(builder, proof_idx, values)
    }
}

impl Default for MerkleLeafBlockBus {
    fn default() -> Self {
        Self::new()
    }
}
