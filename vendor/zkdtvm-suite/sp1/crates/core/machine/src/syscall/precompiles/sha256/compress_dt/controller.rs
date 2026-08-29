use crate::{
    air::MemoryAirBuilder,
    memory::MemoryWriteCols,
    operations_dt::{AddNOperationWithoutResult, CompactWord},
    utils::{next_power_of_two, padded_rows_threshold},
};
use dt_core_executor::{
    events::{ByteRecord, ShaCompressEvent},
    syscalls::SyscallCode,
    ExecutionRecord, Program,
};
use dt_derive::AlignedBorrow;
use dt_stark::{
    air::{AirInteraction, DTAirBuilder, InteractionScope, MachineAir},
    sumcheck::trace::{CompressedMatrix, PaddingRow},
    InteractionKind,
};
use hashbrown::HashMap;
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
use typenum::U2;

pub struct ShaCompressControllerChip {}

impl ShaCompressControllerChip {
    pub fn new() -> Self {
        Self {}
    }
}

/// The number of main trace columns for `ShaCompressControllerCols`.
pub const NUM_SHA_COMPRESS_CONTROLLER_COLS: usize = size_of::<ShaCompressControllerCols<u8>>();

/// The column layout for the chip.
#[derive(AlignedBorrow, Clone)]
#[repr(C)]
pub struct ShaCompressControllerCols<T: Clone> {
    /// The shard number of the syscall.
    pub shard: T,

    /// The clk of the syscall.
    pub clk: T,

    /// The arg1.
    pub w_ptr: T,

    /// The arg2.
    pub h_ptr: T,

    pub is_real: T,

    pub h_access: [MemoryWriteCols<T>; 8],
    pub h_finalize: [CompactWord<T>; 8],
    pub final_add: [AddNOperationWithoutResult<T, U2>; 8],
}

impl<F: Field> MachineAir<F> for ShaCompressControllerChip {
    type Record = ExecutionRecord;

    type Program = Program;

    fn name(&self) -> String {
        "ShaCompressController".to_string()
    }

    fn generate_dependencies(&self, input: &ExecutionRecord, output: &mut ExecutionRecord) {
        for event in input.precompile_events.sha_compress_events() {
            self.event_to_row(
                event,
                &mut [F::zero(); NUM_SHA_COMPRESS_CONTROLLER_COLS],
                output,
            );
        }
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows = input.precompile_events.sha_compress_events().collect_vec().len();
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
        let rows = input
            .precompile_events
            .sha_compress_events()
            .par_bridge()
            .map(|event| {
                let mut row = [F::zero(); NUM_SHA_COMPRESS_CONTROLLER_COLS];
                self.event_to_row(event, &mut row, &mut HashMap::new());
                row
            })
            .collect::<Vec<_>>();

        let padded_nb_rows =
            <ShaCompressControllerChip as MachineAir<F>>::num_rows(self, input).unwrap();
        let main = RowMajorMatrix::new(
            rows.into_iter().flatten().collect::<Vec<_>>(),
            NUM_SHA_COMPRESS_CONTROLLER_COLS,
        );
        CompressedMatrix::new(
            main,
            PaddingRow::Zero { width: NUM_SHA_COMPRESS_CONTROLLER_COLS },
            padded_nb_rows,
        )
    }

    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            !shard.precompile_events.is_sha_compress_empty() &&
                shard.cpu_events == 0 &&
                shard.global_memory_initialize_events.is_empty() &&
                shard.global_memory_finalize_events.is_empty()
        }
    }

    fn commit_scope(&self) -> InteractionScope {
        InteractionScope::Local
    }
}

impl<AB> Air<AB> for ShaCompressControllerChip
where
    AB: DTAirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &ShaCompressControllerCols<AB::Var> = (*local).borrow();

        // Constrain that `local.is_real` is boolean.
        builder.assert_bool(local.is_real);

        builder.assert_eq(
            local.is_real * local.is_real * local.is_real,
            local.is_real * local.is_real * local.is_real,
        );

        let h_initialize: [CompactWord<AB::Expr>; 8] = local.h_access.map(|h| h.prev_value.into());
        let h_finalize: [CompactWord<AB::Expr>; 8] =
            local.h_finalize.map(|h| CompactWord(h.0.map(|h| h.into())));

        let send_values = once(local.shard.into())
            .chain(once(local.clk.into()))
            .chain(once(local.w_ptr.into()))
            .chain(once(AB::Expr::from_canonical_u32(0)))
            .chain(h_initialize.iter().flat_map(|h| h.0.clone().into_iter()))
            .collect::<Vec<_>>();
        builder.send(
            AirInteraction::new(send_values, local.is_real.into(), InteractionKind::ShaCompress),
            InteractionScope::Local,
        );
        let receive_values = once(local.shard.into())
            .chain(once(local.clk.into()))
            .chain(once(local.w_ptr.into()))
            .chain(once(AB::Expr::from_canonical_u32(64)))
            .chain(h_finalize.iter().flat_map(|h| h.0.clone().into_iter()))
            .collect::<Vec<_>>();
        builder.receive(
            AirInteraction::new(receive_values, local.is_real.into(), InteractionKind::ShaCompress),
            InteractionScope::Local,
        );

        // lookup for k
        // for i in 0..64 {
        //     let receive_values = once(local.shard.into())
        //         .chain(once(local.clk.into()))
        //         .chain(once(AB::Expr::from_canonical_u32(i as u32)))
        //         .chain(once(AB::Expr::from_canonical_u32(SHA_COMPRESS_K[i])))
        //         .collect::<Vec<_>>();
        //     builder.receive(
        //         AirInteraction::new(
        //             receive_values,
        //             local.is_real.into(),
        //             InteractionKind::ShaCompress,
        //         ),
        //         InteractionScope::Local,
        //     );
        // }

        let write_clk = local.clk + AB::F::one();
        for i in 0..8 {
            AddNOperationWithoutResult::<AB::F, U2>::eval(
                builder,
                [h_initialize[i].clone(), h_finalize[i].clone()].into_iter(),
                local.h_access[i].access.value.into(),
                local.is_real,
            );

            builder.eval_memory_access(
                local.shard,
                write_clk.clone(),
                local.h_ptr + AB::F::from_canonical_u32((i * size_of::<u32>()) as u32),
                &local.h_access[i],
                local.is_real,
            );
        }

        // Send the "receive interaction" to the global table.
        builder.send(
            AirInteraction::new(
                vec![
                    local.shard.into(),
                    local.clk.into(),
                    AB::Expr::from_canonical_u32(SyscallCode::SHA_COMPRESS.syscall_id()),
                    local.w_ptr.into(),
                    local.h_ptr.into(),
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

impl<F> BaseAir<F> for ShaCompressControllerChip {
    fn width(&self) -> usize {
        NUM_SHA_COMPRESS_CONTROLLER_COLS
    }
}

impl ShaCompressControllerChip {
    fn event_to_row<F: Field>(
        &self,
        event: &ShaCompressEvent,
        row: &mut [F; NUM_SHA_COMPRESS_CONTROLLER_COLS],
        blu: &mut impl ByteRecord,
    ) {
        let cols: &mut ShaCompressControllerCols<F> = row.as_mut_slice().borrow_mut();

        cols.shard = F::from_canonical_u32(event.shard);
        cols.clk = F::from_canonical_u32(event.clk);
        cols.w_ptr = F::from_canonical_u32(event.w_ptr);
        cols.h_ptr = F::from_canonical_u32(event.h_ptr);
        cols.is_real = F::one();

        for i in 0..8 {
            cols.h_access[i].populate(event.h_write_records[i], blu);

            let prev_value = event.h_write_records[i].prev_value;
            let curr_value = event.h_write_records[i].value;
            let h_finalize = curr_value.wrapping_sub(prev_value);

            cols.h_finalize[i] = h_finalize.into();
            AddNOperationWithoutResult::<F, U2>::populate(
                blu,
                [prev_value, h_finalize].into_iter(),
            );
        }
    }
}
