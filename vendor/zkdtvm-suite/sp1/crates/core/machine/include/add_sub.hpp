#pragma once

#include <cstring>

#include "prelude.hpp"
#include "utils.hpp"

namespace dt_core_machine_sys::add_ops {

template<class F>
__DT_HOSTDEV__ __DT_INLINE__ void populate_add(AddOperation<F>& op, const uint32_t a_u32, const uint32_t b_u32) {
    const uint32_t expected = a_u32 + b_u32;
    write_word_from_u32_v2<F>(op.value, expected);
}

template<class F>
__DT_HOSTDEV__ __DT_INLINE__ void populate_sub(SubOperation<F>& op, const uint32_t a_u32, const uint32_t b_u32) {
    const uint32_t expected = a_u32 - b_u32;
    write_word_from_u32_v2<F>(op.value, expected);
}

template<class F>
__DT_HOSTDEV__ void add_event_to_row(const AluEventFfi& event, AddCols<F>& cols) {
    std::memset(&cols, 0, sizeof(cols));
    populate_add<F>(cols.add_operation, event.b, event.c);
    cols.is_real = F::one();
}

template<class F>
__DT_HOSTDEV__ void addi_event_to_row(const AluEventFfi& event, AddiCols<F>& cols) {
    std::memset(&cols, 0, sizeof(cols));
    populate_add<F>(cols.add_operation, event.b, event.c);
    cols.is_real = F::one();
}

template<class F>
__DT_HOSTDEV__ void sub_event_to_row(const AluEventFfi& event, SubCols<F>& cols) {
    std::memset(&cols, 0, sizeof(cols));
    populate_sub<F>(cols.add_operation, event.b, event.c);
    cols.is_real = F::one();
}

}  // namespace dt_core_machine_sys::add_ops
