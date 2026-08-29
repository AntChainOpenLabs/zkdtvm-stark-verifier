use native_recursion_derive::AlignedBorrow;

use crate::config::POSEIDON2_WIDTH;

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct TranscriptSpongeCols<T> {
    pub proof_idx: T,
    pub is_proof_start: T,
    pub is_proof_last: T,
    pub is_valid: T,
    pub tidx: T,
    pub prev_rate: [T; 8],
    pub input16: [T; POSEIDON2_WIDTH],
    pub output16: [T; POSEIDON2_WIDTH],
    pub absorb_mask: [T; 8],
    pub squeeze_mask: [T; POSEIDON2_WIDTH],
    pub prev_s_count: T,
}

pub const NUM_TRANSCRIPT_SPONGE_COLS: usize = TranscriptSpongeCols::<u8>::width();
