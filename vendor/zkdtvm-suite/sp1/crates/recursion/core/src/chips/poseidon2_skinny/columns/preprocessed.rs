use dt_derive::AlignedBorrow;

use crate::{
    chips::{mem::MemoryAccessColsChips, poseidon2_skinny::WIDTH},
    Address,
};

/// Preprocessed columns for the skinny BabyBear Poseidon2 chip (one round per row).
///
/// Layout (68 cells per row):
///   - `round_kind`        : 0 = external round, 1 = internal round
///   - `is_first_round`    : 1 only on the very first row of a permutation; selects an extra
///     `external_linear_layer` on the input state.
///   - `is_real`           : 1 on real rows, 0 on padding rows. AIR-side selector for transition
///     constraints.
///   - `state_in_neg_mult` : -1 on real rows, 0 on padding rows. Used as the `send_single`
///     multiplicity for `state_in` (a `send` with negative mult is equivalent to a `receive` with
///     positive mult; this matches the wide chip's "send-only" lookup convention so padding rows
///     never inject any net interaction into the memory bus).
///   - `round_constants`   : 16 round constants of this round.
///                            * external row: all 16 entries valid.
///                            * internal row: only `[0]` is the round constant; rest are 0.
///   - `state_in_addrs`    : addresses to fetch this row's input state from.
///   - `state_out_mem`     : addresses + write multiplicity for this row's output state.
///                            * intermediate row real:        mult = +1
///                            * last row real:                mult = +instr.mults[i]
///                            * padding row:                  mult = 0
#[derive(AlignedBorrow, Clone, Copy, Debug)]
#[repr(C)]
pub struct Poseidon2PreprocessedColsSkinny<T: Copy> {
    pub round_kind: T,
    pub is_first_round: T,
    pub is_real: T,
    pub state_in_neg_mult: T,
    pub round_constants: [T; WIDTH],
    pub state_in_addrs: [Address<T>; WIDTH],
    pub state_out_mem: [MemoryAccessColsChips<T>; WIDTH],
}

pub type Poseidon2PreprocessedCols<T> = Poseidon2PreprocessedColsSkinny<T>;
