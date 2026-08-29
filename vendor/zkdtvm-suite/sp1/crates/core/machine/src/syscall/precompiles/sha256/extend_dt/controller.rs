use crate::utils::{next_power_of_two, padded_rows_threshold};
use dt_core_executor::{
    events::ShaExtendEvent,
    syscalls::SyscallCode,
    ExecutionRecord, Program,
};
use dt_derive::AlignedBorrow;
use dt_stark::{
    air::{AirInteraction, DTAirBuilder, InteractionScope, MachineAir},
    sumcheck::trace::{CompressedMatrix, PaddingRow},
    InteractionKind,
};
use itertools::Itertools;
use p3_air::{Air, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::{ParallelBridge, ParallelIterator};
use std::{
    borrow::{Borrow, BorrowMut},
    iter::once,
    mem::size_of,
};

pub struct ShaExtendControllerChip {}

impl ShaExtendControllerChip {
    pub fn new() -> Self {
        Self {}
    }
}

/// The number of main trace columns for `ShaExtendControllerCols`.
pub const NUM_SHA_EXTEND_CONTROLLER_COLS: usize = size_of::<ShaExtendControllerCols<u8>>();

/// The column layout for the chip.
#[derive(AlignedBorrow, Clone, Copy)]
#[repr(C)]
pub struct ShaExtendControllerCols<T: Copy> {
    /// The shard number of the syscall.
    pub shard: T,

    /// The clk of the syscall.
    pub clk: T,

    /// The arg1.
    pub w_ptr: T,

    pub is_real: T,
}

impl<F: Field> MachineAir<F> for ShaExtendControllerChip {
    type Record = ExecutionRecord;

    type Program = Program;

    fn name(&self) -> String {
        "ShaExtendController".to_string()
    }

    fn generate_dependencies(&self, _input: &ExecutionRecord, _output: &mut ExecutionRecord) {}

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows = input.precompile_events.sha_extend_events().collect_vec().len();
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
        let row_fn = |syscall_event: &ShaExtendEvent| {
            let mut row = [F::zero(); NUM_SHA_EXTEND_CONTROLLER_COLS];
            let cols: &mut ShaExtendControllerCols<F> = row.as_mut_slice().borrow_mut();

            cols.shard = F::from_canonical_u32(syscall_event.shard);
            cols.clk = F::from_canonical_u32(syscall_event.clk);
            cols.w_ptr = F::from_canonical_u32(syscall_event.w_ptr);
            cols.is_real = F::one();

            row
        };

        let rows = input
            .precompile_events
            .sha_extend_events()
            .par_bridge()
            .map(row_fn)
            .collect::<Vec<_>>();

        let padded_nb_rows =
            <ShaExtendControllerChip as MachineAir<F>>::num_rows(self, input).unwrap();
        let main = RowMajorMatrix::new(
            rows.into_iter().flatten().collect::<Vec<_>>(),
            NUM_SHA_EXTEND_CONTROLLER_COLS,
        );
        CompressedMatrix::new(
            main,
            PaddingRow::Zero { width: NUM_SHA_EXTEND_CONTROLLER_COLS },
            padded_nb_rows,
        )
    }

    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            !shard.precompile_events.is_sha_extend_empty() &&
                shard.cpu_events == 0 &&
                shard.global_memory_initialize_events.is_empty() &&
                shard.global_memory_finalize_events.is_empty()
        }
    }

    fn commit_scope(&self) -> InteractionScope {
        InteractionScope::Local
    }
}

impl<AB> Air<AB> for ShaExtendControllerChip
where
    AB: DTAirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &ShaExtendControllerCols<AB::Var> = (*local).borrow();

        // Constrain that `local.is_real` is boolean.
        builder.assert_bool(local.is_real);

        builder.assert_eq(
            local.is_real * local.is_real * local.is_real,
            local.is_real * local.is_real * local.is_real,
        );

        let send_values = once(local.shard.into())
            .chain(once(local.clk.into()))
            .chain(once(local.w_ptr.into()))
            .chain(once(AB::Expr::from_canonical_u32(16)))
            .collect::<Vec<_>>();
        builder.send(
            AirInteraction::new(send_values, local.is_real.into(), InteractionKind::ShaExtend),
            InteractionScope::Local,
        );
        let receive_values = once(local.shard.into())
            .chain(once(local.clk.into()))
            .chain(once(local.w_ptr.into()))
            .chain(once(AB::Expr::from_canonical_u32(64)))
            .collect::<Vec<_>>();
        builder.receive(
            AirInteraction::new(receive_values, local.is_real.into(), InteractionKind::ShaExtend),
            InteractionScope::Local,
        );

        // Send the "receive interaction" to the global table.
        builder.send(
            AirInteraction::new(
                vec![
                    local.shard.into(),
                    local.clk.into(),
                    AB::Expr::from_canonical_u32(SyscallCode::SHA_EXTEND.syscall_id()),
                    local.w_ptr.into(),
                    AB::Expr::zero(),
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

impl<F> BaseAir<F> for ShaExtendControllerChip {
    fn width(&self) -> usize {
        NUM_SHA_EXTEND_CONTROLLER_COLS
    }
}
