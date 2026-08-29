#pragma once

#include "poseidon2.hpp"
#include "prelude.hpp"

namespace dt_recursion_core_sys::poseidon2_skinny_kb {
using namespace constants;
using namespace poseidon2;

template <class F>
__DT_HOSTDEV__ void write_external_pair_row(F* row,
                                            F* state,
                                            size_t first_round,
                                            bool apply_initial_linear) {
  constexpr size_t WIDTH = POSEIDON2_WIDTH;
  constexpr size_t S0_OFFSET = WIDTH;
  constexpr size_t STATE_OUT_OFFSET = WIDTH + NUM_INTERNAL_ROUNDS_S0;
  constexpr size_t EXTERNAL_ROUNDS_PER_ROW = 2;

  for (size_t i = 0; i < WIDTH; i++) {
    row[i] = state[i];
  }
  // Remaining witness cells are unused on external rows.
  for (size_t k = 0; k < NUM_INTERNAL_ROUNDS_S0; k++) {
    row[S0_OFFSET + k] = F::zero();
  }

  F cur[WIDTH];
  for (size_t i = 0; i < WIDTH; i++) {
    cur[i] = state[i];
  }

  if (apply_initial_linear) {
    external_linear_layer<F>(cur);
  }

  for (size_t j = 0; j < EXTERNAL_ROUNDS_PER_ROW; j++) {
    for (size_t i = 0; i < WIDTH; i++) {
      cur[i] = cur[i] + F(F::to_monty(rc<F>(first_round + j, i)));
      cur[i] = poseidon2::sbox(cur[i]);
    }
    external_linear_layer<F>(cur);

    if (j == 0) {
      for (size_t i = 0; i < WIDTH; i++) {
        row[S0_OFFSET + i] = cur[i];
      }
    }
  }

  for (size_t i = 0; i < WIDTH; i++) {
    row[STATE_OUT_OFFSET + i] = cur[i];
    state[i] = cur[i];
  }
}

/// KoalaBear-only "5-row per permutation" variant of `event_to_row`.
///
/// Row layout (`ROWS_PER_PERMUTE = 5`, row width = `WIDTH + NUM_INTERNAL_ROUNDS_S0 + WIDTH`
/// = 16 + 19 + 16 = 51 cells):
///
///   row 0..1 : first half of external rounds (2 rounds per row)
///              cells: state_in[16] | round_witness[0..16]=mid state | state_out[16]
///
///   row 2    : single "internal-rounds" row, folds ALL `NUM_INTERNAL_ROUNDS` (= 20)
///              internal rounds.
///              cells: state_in[16]
///                   | round_witness[19]   <-- (state[0] + RC[k])^3 for k=0..18
///                   | state_out[16]       <-- state after the full 20-round sequence
///
///   row 3..4 : second half of external rounds (2 rounds per row)
///              cells: state_in[16] | round_witness[0..16]=mid state | state_out[16]
template <class F>
__DT_HOSTDEV__ void event_to_row(const Poseidon2Event<F>& event, F* dst) {
  constexpr size_t WIDTH = POSEIDON2_WIDTH;
  // 16 (state_in) + 19 (round_witness) + 16 (state_out) = 51
  constexpr size_t ROW_WIDTH = WIDTH + NUM_INTERNAL_ROUNDS_S0 + WIDTH;
  constexpr size_t S0_OFFSET = WIDTH;
  constexpr size_t STATE_OUT_OFFSET = WIDTH + NUM_INTERNAL_ROUNDS_S0;
  constexpr size_t EXTERNAL_ROUNDS_PER_ROW = 2;
  constexpr size_t HALF_EXTERNAL_ROWS =
      (NUM_EXTERNAL_ROUNDS / 2) / EXTERNAL_ROUNDS_PER_ROW;

  F state[WIDTH];
  for (size_t i = 0; i < WIDTH; i++) {
    state[i] = event.input[i];
  }

  size_t row_idx = 0;

  // ------------------------------------------------------------------
  // 1. First half of external rounds (2 rounds per row).
  // ------------------------------------------------------------------
  for (size_t r = 0; r < HALF_EXTERNAL_ROWS; r++) {
    write_external_pair_row<F>(
        dst + row_idx * ROW_WIDTH, state, r * EXTERNAL_ROUNDS_PER_ROW, r == 0);
    row_idx++;
  }

  // ------------------------------------------------------------------
  // 2. Single internal-rounds row: fold all 20 internal rounds.
  //    Write witnesses for rounds 0..18 only (round 19 is computed inline by AIR).
  // ------------------------------------------------------------------
  {
    F* row = dst + row_idx * ROW_WIDTH;

    // state_in
    for (size_t i = 0; i < WIDTH; i++) {
      row[i] = state[i];
    }

    F cur[WIDTH];
    for (size_t i = 0; i < WIDTH; i++) {
      cur[i] = state[i];
    }

    for (size_t k = 0; k < NUM_INTERNAL_ROUNDS; k++) {
      size_t round = k + NUM_EXTERNAL_ROUNDS / 2;

      F sbox_in = cur[0] + F(F::to_monty(rc<F>(round, 0)));
      F sbox_out = poseidon2::sbox(sbox_in);

      // Only write witness for rounds 0..18 (NUM_INTERNAL_ROUNDS_S0 = 19).
      if (k < NUM_INTERNAL_ROUNDS_S0) {
        row[S0_OFFSET + k] = sbox_out;
      }

      cur[0] = sbox_out;
      internal_linear_layer<F>(cur);
    }

    // state_out
    for (size_t i = 0; i < WIDTH; i++) {
      row[STATE_OUT_OFFSET + i] = cur[i];
      state[i] = cur[i];
    }

    row_idx++;
  }

  // ------------------------------------------------------------------
  // 3. Second half of external rounds (2 rounds per row).
  // ------------------------------------------------------------------
  for (size_t r = 0; r < HALF_EXTERNAL_ROWS; r++) {
    size_t first_round = NUM_EXTERNAL_ROUNDS / 2 + NUM_INTERNAL_ROUNDS +
                         r * EXTERNAL_ROUNDS_PER_ROW;
    write_external_pair_row<F>(dst + row_idx * ROW_WIDTH, state, first_round, false);
    row_idx++;
  }
}

}  // namespace dt_recursion_core_sys::poseidon2_skinny_kb
