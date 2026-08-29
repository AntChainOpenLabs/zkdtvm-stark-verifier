use dt_derive::AlignedBorrow;

use crate::{
    chips::{
        mem::MemoryAccessColsChips,
        poseidon2_skinny_kb::{ROWS_PER_PERMUTE, WIDTH},
    },
    Address,
};

/// Preprocessed columns for the 5-row KoalaBear skinny Poseidon2 chip layout.
///
/// One row per chip row, 5 rows per permutation. Field semantics:
///
///   - `is_round[r]`          : one-hot selector. `is_round[r] = 1` on the row that handles
///     round-group `r` of this permutation. On padding rows all are 0.
///                              * `is_round[0]` : external pair 0 (rounds 0-1 of first half)
///                              * `is_round[1]` : external pair 1 (rounds 2-3 of first half)
///                              * `is_round[2]` : internal-rounds row (all 20 internal rounds)
///                              * `is_round[3]` : external pair 2 (rounds 0-1 of second half)
///                              * `is_round[4]` : external pair 3 (rounds 2-3 of second half)
///
///                              Derived selectors:
///                              * `is_real = sum(is_round)` (gates all constraints)
///                              * `is_internal = is_round[2]`
///                              * `is_first_row = is_round[0]` (initial external_linear_layer)
///
///   - `state_in_addrs`       : addresses to fetch this row's input state from.
///   - `state_out_mem`        : addresses + write multiplicity for this row's output state.
///                              * intermediate row real: mult = +1
///                              * last row real:         mult = +instr.mults[i]
///                              * padding row:           mult = 0
///
/// Round constants are NOT stored here: they are public constants identical for every
/// permutation, inlined directly in the AIR based on `is_round` selectors.
#[derive(AlignedBorrow, Clone, Copy, Debug)]
#[repr(C)]
pub struct Poseidon2PreprocessedColsSkinnyKb<T: Copy> {
    pub is_round: [T; ROWS_PER_PERMUTE],
    pub state_in_addrs: [Address<T>; WIDTH],
    pub state_out_mem: [MemoryAccessColsChips<T>; WIDTH],
}

pub type Poseidon2PreprocessedCols<T> = Poseidon2PreprocessedColsSkinnyKb<T>;
