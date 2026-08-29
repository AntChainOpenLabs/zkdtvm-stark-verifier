#pragma once

#ifdef USE_KOALABEAR
    #include <cstdint>

namespace dt_core_machine_sys {
// cbindgen keeps the BabyBear-shaped FFI signatures when the Rust crate is
// compiled for KoalaBear, but it omits the private BabyBear definition. Both
// fields are repr(transparent) u32 wrappers, so provide the ABI-only shape.
struct BabyBearP3 {
    uint32_t value;
};
}  // namespace dt_core_machine_sys
#endif

#include "dt-core-machine-sys-cbindgen.hpp"

#ifndef __CUDACC__
    #define __DT_HOSTDEV__
    #define __DT_INLINE__ inline
    #include <array>

namespace dt_core_machine_sys {
template<class T, std::size_t N>
using array_t = std::array<T, N>;
}  // namespace dt
#else
    #define __DT_HOSTDEV__ __host__ __device__
    #define __DT_INLINE__ 
    #include <cuda/std/array>

namespace dt_core_machine_sys {
template<class T, std::size_t N>
using array_t = cuda::std::array<T, N>;
}  // namespace dt
#endif
