use std::mem::size_of;

use dt_derive::AlignedBorrow;

use crate::chips::poseidon2_skinny::WIDTH;

pub mod preprocessed;

/// Number of cells in one main-trace row of the skinny BabyBear Poseidon2 chip.
///
/// Layout: `state_in[16] | state_out[16]` -> 32 cells.
pub const NUM_POSEIDON2_COLS: usize = size_of::<Poseidon2<u8>>();

/// Main-trace columns (one round per row).
///
/// Each row holds the round's input state and the round's output state. Cross-row state
/// transitions are not constrained directly (the constraint system is single-row only);
/// instead, consecutive rounds are chained through memory lookups carried in the
/// preprocessed trace.
#[derive(AlignedBorrow, Clone, Copy)]
#[repr(C)]
pub struct Poseidon2<T: Copy> {
    /// Input state of this round.
    pub state_in: [T; WIDTH],
    /// Output state of this round.
    pub state_out: [T; WIDTH],
}
