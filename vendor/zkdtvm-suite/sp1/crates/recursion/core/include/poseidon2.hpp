#pragma once

#include "poseidon2_constants.hpp"
#include "prelude.hpp"

namespace dt_recursion_core_sys::poseidon2 {
using namespace constants;

/// Total number of Poseidon2 rounds (one row per round in the skinny chip).
constexpr size_t NUM_ROUNDS = NUM_EXTERNAL_ROUNDS + NUM_INTERNAL_ROUNDS;

constexpr size_t POSEIDON2_WIDTH = 16;

template <class F>
__DT_HOSTDEV__ __DT_INLINE__ void mdsLightPermutation4x4(F state[4]) {
  F t01 = state[0] + state[1];
  F t23 = state[2] + state[3];
  F t0123 = t01 + t23;
  F t01123 = t0123 + state[1];
  F t01233 = t0123 + state[3];
  state[3] = t01233 + operator<<(state[0], 1);
  state[1] = t01123 + operator<<(state[2], 1);
  state[0] = t01123 + t01;
  state[2] = t01233 + t23;
}

template <class F>
__DT_HOSTDEV__ __DT_INLINE__ void external_linear_layer(
    F state_var[POSEIDON2_WIDTH]) {
  for (size_t i = 0; i < POSEIDON2_WIDTH; i += 4) {
    mdsLightPermutation4x4(state_var + i);
  }

  F sums[4] = {F::zero(), F::zero(), F::zero(), F::zero()};
  for (size_t k = 0; k < 4; k++) {
    for (size_t j = 0; j < POSEIDON2_WIDTH; j += 4) {
      sums[k] = sums[k] + state_var[j + k];
    }
  }

  for (size_t j = 0; j < POSEIDON2_WIDTH; j++) {
    state_var[j] = state_var[j] + sums[j % 4];
  }
}

template <class F>
__DT_HOSTDEV__ __DT_INLINE__ void internal_linear_layer(
    F state[POSEIDON2_WIDTH]) {
  // Build matmul_constants from domain-independent canonical values.
  // The first element is F::MOD - 2, the rest are 1 << shift.
  F matmul_constants[POSEIDON2_WIDTH];
  matmul_constants[0] = F(F::to_monty(F::MOD - 2));
  for (size_t i = 1; i < POSEIDON2_WIDTH; i++) {
    matmul_constants[i] = F(F::to_monty(
        1u << constants::POSEIDON2_INTERNAL_MATRIX_DIAG_16_SHIFTS[i - 1]));
  }

  F sum = F::zero();
  for (size_t i = 0; i < POSEIDON2_WIDTH; i++) {
    sum = sum + state[i];
  }

  for (size_t i = 0; i < POSEIDON2_WIDTH; i++) {
    state[i] = state[i] * matmul_constants[i];
    state[i] = state[i] + sum;
  }

  F monty_inverse = F(F::to_monty(F::from_monty(1)));
  for (size_t i = 0; i < POSEIDON2_WIDTH; i++) {
    state[i] = state[i] * monty_inverse;
  }
}
/// Apply the Poseidon2 S-box to a single element.
/// For BabyBear (MOD = 0x78000001), SBOX_DEGREE = 7: x^7 = x^3 * x^3 * x.
/// For KoalaBear (MOD = 0x7f000001), SBOX_DEGREE = 3: x^3.
template <class F>
__DT_HOSTDEV__ __DT_INLINE__ F sbox(F x) {
  F x3 = x * x * x;
  if constexpr (F::MOD == 0x78000001u) {
    // BabyBear: x^7
    return x3 * x3 * x;
  } else {
    // KoalaBear: x^3
    return x3;
  }
}
}  // namespace dt_recursion_core_sys::poseidon2