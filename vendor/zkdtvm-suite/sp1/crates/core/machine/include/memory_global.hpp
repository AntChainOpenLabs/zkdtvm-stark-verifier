#pragma once

#include "prelude.hpp"
#include "utils.hpp"
#include "bb31_septic_extension_t.hpp"
#include "memory_local.hpp"

namespace dt_core_machine_sys::memory_global {
#ifdef USE_KOALABEAR
    constexpr uint8_t FIELD_ADDR_MSB_THRESHOLD = 0x7f;
#else
    constexpr uint8_t FIELD_ADDR_MSB_THRESHOLD = 0x78;
#endif

    template<class F, class EF7>
    __DT_HOSTDEV__ void event_to_row(const MemoryInitializeFinalizeEvent* event, const bool is_receive, MemoryInitCols<F>* cols) {
        [[maybe_unused]]MemoryRecord record;
        if (is_receive) {
            record.shard = event->shard;
            record.timestamp = event->timestamp;
            record.value = event->value;
        } else {
            record.shard = 0;
            record.timestamp = 0;
            record.value = event->value;
        }
        cols->addr = F::from_canonical_u32(event->addr);
        // Byte decomposition of addr (little-endian).
        cols->addr_word._0[0] = F::from_canonical_u32((event->addr) & 0xFF);
        cols->addr_word._0[1] = F::from_canonical_u32(((event->addr) >> 8) & 0xFF);
        cols->addr_word._0[2] = F::from_canonical_u32(((event->addr) >> 16) & 0xFF);
        cols->addr_word._0[3] = F::from_canonical_u32(((event->addr) >> 24) & 0xFF);
        cols->shard = F::from_canonical_u32(event->shard);
        cols->timestamp = F::from_canonical_u32(event->timestamp);
        // Byte decomposition of value (little-endian).
        cols->value._0[0] = F::from_canonical_u32((event->value) & 0xFF);
        cols->value._0[1] = F::from_canonical_u32(((event->value) >> 8) & 0xFF);
        cols->value._0[2] = F::from_canonical_u32(((event->value) >> 16) & 0xFF);
        cols->value._0[3] = F::from_canonical_u32(((event->value) >> 24) & 0xFF);
        cols->is_real = F::one();
        // is_addr_zero hint.
        cols->is_addr_zero = (event->addr == 0) ? F::one() : F::zero();
        // Field range check hint.
        uint8_t msb = (event->addr >> 24) & 0xFF;
        cols->is_addr_lt_threshold = (msb < FIELD_ADDR_MSB_THRESHOLD) ? F::one() : F::zero();
    }
}  // namespace dt::memory_local
