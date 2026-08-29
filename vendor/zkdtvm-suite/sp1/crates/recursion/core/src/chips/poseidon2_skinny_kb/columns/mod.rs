use std::mem::size_of;

use dt_derive::AlignedBorrow;

use crate::chips::poseidon2_skinny_kb::{NUM_INTERNAL_ROUNDS, WIDTH};

pub mod preprocessed;

/// Number of cells in one main-trace row of the skinny KoalaBear Poseidon2 chip.
///
/// Layout: `state_in[16] | round_witness[19] | state_out[16]` = 51 cells.
pub const NUM_POSEIDON2_COLS: usize = size_of::<Poseidon2<u8>>();

/// Main-trace columns for the 5-row KoalaBear skinny layout.
///
/// All rows share the same width. Field semantics vary by row kind:
///
///   * **External row** (rows 0, 1, 3, 4):
///       - `state_in`       : this row's input state
///       - `round_witness`  : first `WIDTH` (16) cells carry the state after the first of the two
///         external rounds; remaining 3 cells unused (zero).
///       - `state_out`      : output state after 2 external rounds.
///
///   * **Internal-rounds row** (row 2):
///       - `state_in`       : input state
///       - `round_witness`  : `round_witness[k]` for k=0..18 holds the post-S-box value of
///         `state[0]` produced by internal round `k`. Round 19 (the last) is computed inline
///         without a witness.
///       - `state_out`      : state after all 20 internal rounds.
#[derive(AlignedBorrow, Clone, Copy)]
#[repr(C)]
pub struct Poseidon2<T: Copy> {
    pub state_in: [T; WIDTH],
    pub round_witness: [T; NUM_INTERNAL_ROUNDS - 1],
    pub state_out: [T; WIDTH],
}
