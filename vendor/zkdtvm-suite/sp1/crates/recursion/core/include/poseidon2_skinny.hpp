#pragma once

#include "poseidon2.hpp"
#include "prelude.hpp"

namespace dt_recursion_core_sys::poseidon2_skinny {
using namespace constants;
using namespace poseidon2;

/// Single-row-per-round variant of event_to_row (used by both BabyBear and KoalaBear).
///
/// Layout per row: state_in[16] | state_out[16] (32 cells).
/// Total rows: NUM_ROUNDS (= NUM_EXTERNAL_ROUNDS + NUM_INTERNAL_ROUNDS).
///
/// The function re-runs the Poseidon2 permutation round-by-round and fills
/// each row with (state_in, state_out) for that round.
template <class F>
__DT_HOSTDEV__ void event_to_row(const Poseidon2Event<F>& event,
                                  F* dst) {
  constexpr size_t ROW_WIDTH = 2 * WIDTH;  // state_in[16] + state_out[16]

  F state[WIDTH];
  for (size_t i = 0; i < WIDTH; i++) {
    state[i] = event.input[i];
  }

  size_t row_idx = 0;

  // First half of external rounds (4 rounds).
  for (size_t r = 0; r < NUM_EXTERNAL_ROUNDS / 2; r++) {
    F* row = dst + row_idx * ROW_WIDTH;

    // Write state_in.
    for (size_t i = 0; i < WIDTH; i++) {
      row[i] = state[i];
    }

    F next[WIDTH];
    for (size_t i = 0; i < WIDTH; i++) {
      next[i] = state[i];
    }

    // The first round absorbs the initial linear layer.
    if (r == 0) {
      external_linear_layer<F>(next);
    }

    // Add round constants and apply sbox.
    for (size_t i = 0; i < WIDTH; i++) {
      next[i] = next[i] + F(F::to_monty(rc<F>(r, i)));
      next[i] = poseidon2::sbox(next[i]);
    }

    external_linear_layer<F>(next);

    // Write state_out.
    for (size_t i = 0; i < WIDTH; i++) {
      row[WIDTH + i] = next[i];
      state[i] = next[i];
    }

    row_idx++;
  }

  // Internal rounds (NUM_INTERNAL_ROUNDS = 20 rounds).
  for (size_t r = 0; r < NUM_INTERNAL_ROUNDS; r++) {
    F* row = dst + row_idx * ROW_WIDTH;

    // Write state_in.
    for (size_t i = 0; i < WIDTH; i++) {
      row[i] = state[i];
    }

    F next[WIDTH];
    for (size_t i = 0; i < WIDTH; i++) {
      next[i] = state[i];
    }

    size_t round = r + NUM_EXTERNAL_ROUNDS / 2;
    next[0] = next[0] + F(F::to_monty(rc<F>(round, 0)));
    next[0] = poseidon2::sbox(next[0]);
    internal_linear_layer<F>(next);

    // Write state_out.
    for (size_t i = 0; i < WIDTH; i++) {
      row[WIDTH + i] = next[i];
      state[i] = next[i];
    }

    row_idx++;
  }

  // Second half of external rounds (4 rounds).
  for (size_t r = 0; r < NUM_EXTERNAL_ROUNDS / 2; r++) {
    F* row = dst + row_idx * ROW_WIDTH;

    // Write state_in.
    for (size_t i = 0; i < WIDTH; i++) {
      row[i] = state[i];
    }

    F next[WIDTH];
    for (size_t i = 0; i < WIDTH; i++) {
      next[i] = state[i];
    }

    size_t round = r + NUM_EXTERNAL_ROUNDS / 2 + NUM_INTERNAL_ROUNDS;
    for (size_t i = 0; i < WIDTH; i++) {
      next[i] = next[i] + F(F::to_monty(rc<F>(round, i)));
      next[i] = poseidon2::sbox(next[i]);
    }

    external_linear_layer<F>(next);

    // Write state_out.
    for (size_t i = 0; i < WIDTH; i++) {
      row[WIDTH + i] = next[i];
      state[i] = next[i];
    }

    row_idx++;
  }
}

}  // namespace dt_recursion_core_sys::poseidon2_skinny
