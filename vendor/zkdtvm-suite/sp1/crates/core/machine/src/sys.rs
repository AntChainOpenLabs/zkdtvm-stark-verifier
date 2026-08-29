use crate::{
    alu::{
        AddCols, AddiCols, BitwiseCols, LtCols, MulCols, ShiftLeftCols, ShiftRightCols, SubCols,
    },
    memory::{MemoryInitCols, SingleMemoryLocal},
    syscall::chip::SyscallCols,
};
use p3_baby_bear::BabyBear;

use dt_core_executor::{
    events::{
        AluEvent, MemoryInitializeFinalizeEvent, MemoryLocalEvent, MemoryReadRecord,
        MemoryRecordEnum, MemoryWriteRecord, SyscallEvent,
    },
    Opcode,
};

/// FFI-safe version of [`AluEvent`] for C++ interop.
///
/// Contains only the flat, primitive fields that C++ code needs.
/// The problematic `Option<MemoryRecordEnum>` fields
/// is intentionally excluded because cbindgen cannot generate valid C++ for it.
#[derive(Debug, Copy, Clone)]
#[repr(C)]
pub struct AluEventFfi {
    /// The program counter.
    pub pc: u32,
    /// The opcode.
    pub opcode: Opcode,
    /// The first operand value.
    pub a: u32,
    /// The second operand value.
    pub b: u32,
    /// The third operand value.
    pub c: u32,
    /// Whether the first operand is register 0.
    pub op_a_0: bool,
}

impl From<&AluEvent> for AluEventFfi {
    fn from(event: &AluEvent) -> Self {
        Self {
            pc: event.pc,
            opcode: event.opcode,
            a: event.a,
            b: event.b,
            c: event.c,
            op_a_0: event.op_a_0,
        }
    }
}

/// FFI-safe version of [`SyscallEvent`] for C++ interop.
///
/// Contains only the flat, primitive fields that C++ code needs.
/// The problematic `MemoryWriteRecord` and `SyscallCode` fields
/// are intentionally excluded because cbindgen cannot generate valid C++ for them.
#[derive(Debug, Copy, Clone)]
#[repr(C)]
pub struct SyscallEventFfi {
    /// The shard number.
    pub shard: u32,
    /// The clock cycle.
    pub clk: u32,
    /// The syscall id.
    pub syscall_id: u32,
    /// The first operand value (`op_b`).
    pub arg1: u32,
    /// The second operand value (`op_c`).
    pub arg2: u32,
}

impl From<&SyscallEvent> for SyscallEventFfi {
    fn from(event: &SyscallEvent) -> Self {
        Self {
            shard: event.shard,
            clk: event.clk,
            syscall_id: event.syscall_id,
            arg1: event.arg1,
            arg2: event.arg2,
        }
    }
}

#[link(name = "dt-core-machine-sys", kind = "static")]
extern "C-unwind" {
    pub fn add_event_to_row_babybear(event: &AluEventFfi, cols: &mut AddCols<BabyBear>);
    pub fn addi_event_to_row_babybear(event: &AluEventFfi, cols: &mut AddiCols<BabyBear>);
    pub fn sub_event_to_row_babybear(event: &AluEventFfi, cols: &mut SubCols<BabyBear>);
    pub fn mul_event_to_row_babybear(event: &AluEventFfi, cols: &mut MulCols<BabyBear>);
    pub fn bitwise_event_to_row_babybear(event: &AluEventFfi, cols: &mut BitwiseCols<BabyBear>);
    pub fn lt_event_to_row_babybear(event: &AluEventFfi, cols: &mut LtCols<BabyBear>);
    pub fn sll_event_to_row_babybear(event: &AluEventFfi, cols: &mut ShiftLeftCols<BabyBear>);
    pub fn sr_event_to_row_babybear(event: &AluEventFfi, cols: &mut ShiftRightCols<BabyBear>);
    pub fn memory_local_event_to_row_babybear(
        event: &MemoryLocalEvent,
        cols: &mut SingleMemoryLocal<BabyBear>,
    );
    pub fn memory_global_event_to_row_babybear(
        event: &MemoryInitializeFinalizeEvent,
        is_receive: bool,
        cols: &mut MemoryInitCols<BabyBear>,
    );
    pub fn syscall_event_to_row_babybear(
        event: &SyscallEventFfi,
        is_receive: bool,
        cols: &mut SyscallCols<BabyBear>,
    );
}

/// An alternative to `Option<MemoryRecordEnum>` that is FFI-safe.
///
/// See [`MemoryRecordEnum`].
#[derive(Debug, Copy, Clone)]
#[repr(C)]
pub enum OptionMemoryRecordEnum {
    /// Read.
    Read(MemoryReadRecord),
    /// Write.
    Write(MemoryWriteRecord),
    None,
}

impl From<Option<MemoryRecordEnum>> for OptionMemoryRecordEnum {
    fn from(value: Option<MemoryRecordEnum>) -> Self {
        match value {
            Some(MemoryRecordEnum::Read(r)) => Self::Read(r),
            Some(MemoryRecordEnum::Write(r)) => Self::Write(r),
            None => Self::None,
        }
    }
}

impl From<OptionMemoryRecordEnum> for Option<MemoryRecordEnum> {
    fn from(value: OptionMemoryRecordEnum) -> Self {
        match value {
            OptionMemoryRecordEnum::Read(r) => Some(MemoryRecordEnum::Read(r)),
            OptionMemoryRecordEnum::Write(r) => Some(MemoryRecordEnum::Write(r)),
            OptionMemoryRecordEnum::None => None,
        }
    }
}
