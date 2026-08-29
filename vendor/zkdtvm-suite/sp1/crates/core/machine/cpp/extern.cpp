#include "bb31_t.hpp"
#include "bb31_septic_extension_t.hpp"
#include "sys.hpp"

namespace dt_core_machine_sys {
extern void add_event_to_row_babybear(
    const AluEventFfi* event,
    AddCols<BabyBearP3>* cols
) {
    AddCols<bb31_t>* cols_bb31 = reinterpret_cast<AddCols<bb31_t>*>(cols);
    add_ops::add_event_to_row<bb31_t>(*event, *cols_bb31);
}

extern void addi_event_to_row_babybear(
    const AluEventFfi* event,
    AddiCols<BabyBearP3>* cols
) {
    AddiCols<bb31_t>* cols_bb31 = reinterpret_cast<AddiCols<bb31_t>*>(cols);
    add_ops::addi_event_to_row<bb31_t>(*event, *cols_bb31);
}

extern void sub_event_to_row_babybear(
    const AluEventFfi* event,
    SubCols<BabyBearP3>* cols
) {
    SubCols<bb31_t>* cols_bb31 = reinterpret_cast<SubCols<bb31_t>*>(cols);
    add_ops::sub_event_to_row<bb31_t>(*event, *cols_bb31);
}

extern void memory_local_event_to_row_babybear(const MemoryLocalEvent* event, SingleMemoryLocal<BabyBearP3>* cols) {
    SingleMemoryLocal<bb31_t>* cols_bb31 = reinterpret_cast<SingleMemoryLocal<bb31_t>*>(cols);
    memory_local::event_to_row<bb31_t, bb31_septic_extension_t>(event, cols_bb31);
}

extern void memory_global_event_to_row_babybear(const MemoryInitializeFinalizeEvent* event, const bool is_receive, MemoryInitCols<BabyBearP3>* cols) {
    MemoryInitCols<bb31_t>* cols_bb31 = reinterpret_cast<MemoryInitCols<bb31_t>*>(cols);
    memory_global::event_to_row<bb31_t, bb31_septic_extension_t>(event, is_receive, cols_bb31);
}

extern void syscall_event_to_row_babybear(const SyscallEventFfi* event, const bool is_receive, SyscallCols<BabyBearP3>* cols) {
    SyscallCols<bb31_t>* cols_bb31 = reinterpret_cast<SyscallCols<bb31_t>*>(cols);
    syscall::event_to_row<bb31_t, bb31_septic_extension_t>(event, is_receive, cols_bb31);
}
} // namespace dt_core_machine_sys
