#pragma once

#include "prelude.hpp"

namespace dt_recursion_core_sys::poly_eval {
template <class F>
__DT_HOSTDEV__ void event_to_row(const PolyEvalEventFFI<F>& event,
                                  size_t i, PolyEvalCols<F>& cols) {
  cols.point = *event.point;
  cols.current_coeff = event.coeff_ptr[i];
}

template <class F>
__DT_HOSTDEV__ void instr_to_row(const PolyEvalInstrFFI<F>& instr,
                                  size_t i, size_t len,
                                  PolyEvalPreprocessedCols<F>& cols) {
  cols.is_real = F::one();
  cols.iteration_num = F::from_canonical_u32(i);
  cols.is_first = F::from_bool(i == 0);
  cols.is_last = F::from_bool(i == len - 1);

  cols.point_mem.addr = *instr.point;
  cols.point_mem.mult = F::zero() - F::from_bool(i == 0);

  cols.coeff_mem.addr = instr.coeff_ptr[i];
  cols.coeff_mem.mult = F::zero() - F::one();

  cols.out_mem.addr = *instr.out;
  cols.out_mem.mult = *instr.mult * F::from_bool(i == len - 1);
}
}  // namespace dt_recursion_core_sys::poly_eval
