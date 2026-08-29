#pragma once

#include "prelude.hpp"

namespace dt_recursion_core_sys::ext_exp_reverse_bits {
template <class F>
__DT_HOSTDEV__ void event_to_row(const ExtExpReverseBitsEventFFI<F>& event,
                                  size_t i, ExtExpReverseBitsCols<F>& cols) {
  cols.x = *event.base;
  cols.current_bit = event.exp_ptr[i];
  cols.prev_acc = event.prev_acc_ptr[i];
  cols.acc = event.acc_ptr[i];
  Block<F> one = {F::one(), F::zero(), F::zero(), F::zero()};
  cols.multiplier = (event.exp_ptr[i] == F::one()) ? *event.base : one;
}

template <class F>
__DT_HOSTDEV__ void instr_to_row(const ExtExpReverseBitsInstrFFI<F>& instr,
                                  size_t i, size_t len,
                                  ExtExpReverseBitsPreprocessedCols<F>& cols) {
  cols.is_real = F::one();
  cols.x_addr = *instr.base;
  cols.exponent_addr = instr.exp_ptr[i];
  cols.prev_acc_addr = instr.prev_acc_ptr[i];
  cols.acc_mem.addr = instr.acc_ptr[i];
  cols.acc_mem.mult = (i == len - 1) ? *instr.mult : F::one();
}
}  // namespace dt_recursion_core_sys::ext_exp_reverse_bits
