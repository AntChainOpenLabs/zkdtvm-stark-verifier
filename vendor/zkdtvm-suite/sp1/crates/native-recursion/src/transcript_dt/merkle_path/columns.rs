use native_recursion_derive::AlignedBorrow;

use crate::config::{DIGEST_SIZE, POSEIDON2_WIDTH};

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct MerklePathCols<T> {
    pub proof_idx: T,
    pub is_valid: T,
    pub is_leaf_absorb: T,
    pub is_inject: T,
    pub is_last: T,
    pub is_leaf_first: T,
    pub is_leaf_last: T,
    /// Routing tag for the deduped leaf's block/sponge streams:
    /// `slot*32 + codeword_log_height`.
    /// Its product formula is constrained by the leaf producer, and
    /// `commit_id` binds the corresponding block to the authenticated tree.
    pub unit_key: T,
    pub commit_id: T,
    pub level: T,
    pub block_idx: T,
    pub idx: T,
    pub left_idx: T,
    pub left_cnt: T,
    pub right_cnt: T,
    pub root_cnt: T,
    /// Leaf-block recv count — balance-forced to the producer send
    /// multiset (1 for deduped stream units, m for m-query pair leaves).
    pub absorb_cnt: T,
    pub prev_state: [T; POSEIDON2_WIDTH],
    pub chunk: [T; DIGEST_SIZE],
    pub chunk_mask: [T; DIGEST_SIZE],
    pub input: [T; POSEIDON2_WIDTH],
    pub output: [T; POSEIDON2_WIDTH],
}

pub const NUM_MERKLE_PATH_COLS: usize = MerklePathCols::<u8>::width();
