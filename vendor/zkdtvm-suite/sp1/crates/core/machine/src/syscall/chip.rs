use crate::utils::{next_power_of_two, padded_rows_threshold};
use core::fmt;
use dt_core_executor::{events::SyscallEvent, ExecutionRecord, Program};
use dt_derive::AlignedBorrow;
use dt_stark::{
    air::{AirInteraction, DTAirBuilder, InteractionScope, MachineAir},
    sumcheck::trace::{CompressedMatrix, PaddingRow},
    InteractionKind,
};
use p3_air::{Air, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::{IntoParallelRefIterator, ParallelBridge, ParallelIterator};
use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};
/// The number of main trace columns for `SyscallChip`.
pub const NUM_SYSCALL_COLS: usize = size_of::<SyscallCols<u8>>();

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SyscallShardKind {
    Core,
    Precompile,
}

/// A chip that stores the syscall invocations.
pub struct SyscallChip {
    shard_kind: SyscallShardKind,
}

impl SyscallChip {
    pub const fn new(shard_kind: SyscallShardKind) -> Self {
        Self { shard_kind }
    }

    pub const fn core() -> Self {
        Self::new(SyscallShardKind::Core)
    }

    pub const fn precompile() -> Self {
        Self::new(SyscallShardKind::Precompile)
    }

    pub fn shard_kind(&self) -> SyscallShardKind {
        self.shard_kind
    }
}

/// The column layout for the chip.
#[derive(AlignedBorrow, Clone, Copy)]
#[repr(C)]
pub struct SyscallCols<T: Copy> {
    /// The shard number of the syscall.
    pub shard: T,

    /// The clk of the syscall.
    pub clk: T,

    /// The syscall_id of the syscall.
    pub syscall_id: T,

    /// The arg1.
    pub arg1: T,

    /// The arg2.
    pub arg2: T,

    pub is_real: T,
}

impl<F: Field> MachineAir<F> for SyscallChip {
    type Record = ExecutionRecord;

    type Program = Program;

    fn name(&self) -> String {
        format!("Syscall{}", self.shard_kind).to_string()
    }

    fn generate_dependencies(&self, _input: &ExecutionRecord, _output: &mut ExecutionRecord) {}

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows = match self.shard_kind() {
            SyscallShardKind::Core => input.syscall_events.len(),
            SyscallShardKind::Precompile => input.precompile_events.all_precompile_events().count(),
        };
        let size_log2 = input.fixed_log2_rows::<F, _>(self);
        let mut padded_nb_rows = next_power_of_two(nb_rows, size_log2);
        padded_nb_rows = padded_rows_threshold(padded_nb_rows);
        Some(padded_nb_rows)
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _output: &mut ExecutionRecord,
    ) -> CompressedMatrix<F> {
        let row_fn = |syscall_event: &SyscallEvent, _: bool| {
            let mut row = [F::zero(); NUM_SYSCALL_COLS];
            let cols: &mut SyscallCols<F> = row.as_mut_slice().borrow_mut();

            cols.shard = F::from_canonical_u32(syscall_event.shard);
            cols.clk = F::from_canonical_u32(syscall_event.clk);
            cols.syscall_id = F::from_canonical_u32(syscall_event.syscall_code.syscall_id());
            cols.arg1 = F::from_canonical_u32(syscall_event.arg1);
            cols.arg2 = F::from_canonical_u32(syscall_event.arg2);
            cols.is_real = F::one();
            row
        };

        let rows = match self.shard_kind {
            SyscallShardKind::Core => input
                .syscall_events
                .par_iter()
                .map(|(_, e)| e)
                .filter(|event| event.syscall_code.should_send() == 1)
                .map(|event| row_fn(event, false))
                .collect::<Vec<_>>(),
            SyscallShardKind::Precompile => input
                .precompile_events
                .all_precompile_events()
                .map(|(event, _)| event)
                .par_bridge()
                .map(|event| row_fn(event, true))
                .collect::<Vec<_>>(),
        };

        let padded_nb_rows = <SyscallChip as MachineAir<F>>::num_rows(self, input).unwrap();
        let main =
            RowMajorMatrix::new(rows.into_iter().flatten().collect::<Vec<_>>(), NUM_SYSCALL_COLS);
        CompressedMatrix::new(main, PaddingRow::Zero { width: NUM_SYSCALL_COLS }, padded_nb_rows)
    }

    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            match self.shard_kind {
                SyscallShardKind::Core => {
                    shard
                        .syscall_events
                        .iter()
                        .map(|(_, e)| e)
                        .filter(|e| e.syscall_code.should_send() == 1)
                        .take(1)
                        .count()
                        > 0
                }
                SyscallShardKind::Precompile => {
                    !shard.precompile_events.is_precompile_empty()
                        && shard.cpu_events == 0
                        && shard.global_memory_initialize_events.is_empty()
                        && shard.global_memory_finalize_events.is_empty()
                }
            }
        }
    }

    fn commit_scope(&self) -> InteractionScope {
        InteractionScope::Local
    }
}

impl<AB> Air<AB> for SyscallChip
where
    AB: DTAirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &SyscallCols<AB::Var> = (*local).borrow();

        // Constrain that `local.is_real` is boolean.
        builder.assert_bool(local.is_real);

        builder.assert_eq(
            local.is_real * local.is_real * local.is_real,
            local.is_real * local.is_real * local.is_real,
        );

        match self.shard_kind {
            SyscallShardKind::Core => {
                builder.receive_syscall(
                    local.shard,
                    local.clk,
                    local.syscall_id,
                    local.arg1,
                    local.arg2,
                    local.is_real,
                    InteractionScope::Local,
                );

                // Send the "send interaction" to the global table.
                builder.send(
                    AirInteraction::new(
                        vec![
                            local.shard.into(),
                            local.clk.into(),
                            local.syscall_id.into(),
                            local.arg1.into(),
                            local.arg2.into(),
                            AB::Expr::zero(),
                            AB::Expr::zero(),
                            AB::Expr::one(),
                            AB::Expr::zero(),
                            AB::Expr::from_canonical_u8(InteractionKind::Syscall as u8),
                        ],
                        local.is_real.into(),
                        InteractionKind::Global,
                    ),
                    InteractionScope::Local,
                );
            }
            SyscallShardKind::Precompile => {
                builder.send_syscall(
                    local.shard,
                    local.clk,
                    local.syscall_id,
                    local.arg1,
                    local.arg2,
                    local.is_real,
                    InteractionScope::Local,
                );

                // Send the "receive interaction" to the global table.
                builder.send(
                    AirInteraction::new(
                        vec![
                            local.shard.into(),
                            local.clk.into(),
                            local.syscall_id.into(),
                            local.arg1.into(),
                            local.arg2.into(),
                            AB::Expr::zero(),
                            AB::Expr::zero(),
                            AB::Expr::zero(),
                            AB::Expr::one(),
                            AB::Expr::from_canonical_u8(InteractionKind::Syscall as u8),
                        ],
                        local.is_real.into(),
                        InteractionKind::Global,
                    ),
                    InteractionScope::Local,
                );
            }
        }
    }
}

impl<F> BaseAir<F> for SyscallChip {
    fn width(&self) -> usize {
        NUM_SYSCALL_COLS
    }
}

impl fmt::Display for SyscallShardKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SyscallShardKind::Core => write!(f, "Core"),
            SyscallShardKind::Precompile => write!(f, "Precompile"),
        }
    }
}
