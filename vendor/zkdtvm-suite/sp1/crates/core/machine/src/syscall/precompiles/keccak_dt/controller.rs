use crate::{
    air::MemoryAirBuilder,
    memory::MemoryWriteCols,
    operations_dt::CompactWord,
    syscall::precompiles::keccak_dt::STATE_NUM_WORDS,
    utils::{next_power_of_two, padded_rows_threshold},
};
use dt_core_executor::{
    events::{ByteRecord, KeccakPermuteEvent},
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

pub struct KeccakControllerChip {}

impl KeccakControllerChip {
    pub fn new() -> Self {
        Self {}
    }
}

/// The number of main trace columns for `KeccakControllerCols`.
pub const NUM_KECCAK_CONTROLLER_COLS: usize = size_of::<KeccakControllerCols<u8>>();

/// The column layout for the chip.
#[derive(AlignedBorrow, Clone)]
#[repr(C)]
pub struct KeccakControllerCols<T: Clone> {
    /// The shard number of the syscall.
    pub shard: T,

    /// The clk of the syscall.
    pub clk: T,

    /// The arg1.
    pub state_ptr: T,

    pub is_real: T,

    pub state_access: [MemoryWriteCols<T>; STATE_NUM_WORDS],
}

impl<F: Field> MachineAir<F> for KeccakControllerChip {
    type Record = ExecutionRecord;

    type Program = Program;

    fn name(&self) -> String {
        "KeccakController".to_string()
    }

    fn generate_dependencies(&self, input: &ExecutionRecord, output: &mut ExecutionRecord) {
        for event in input.precompile_events.keccak_events() {
            self.event_to_row(event, &mut [F::zero(); NUM_KECCAK_CONTROLLER_COLS], output);
        }
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows = input.precompile_events.keccak_events().collect_vec().len();
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
            .keccak_events()
            .par_bridge()
            .map(|event| {
                let mut row = [F::zero(); NUM_KECCAK_CONTROLLER_COLS];
                self.event_to_row(event, &mut row, &mut HashMap::new());
                row
            })
            .collect::<Vec<_>>();

        let padded_nb_rows =
            <KeccakControllerChip as MachineAir<F>>::num_rows(self, input).unwrap();
        let main = RowMajorMatrix::new(
            rows.into_iter().flatten().collect::<Vec<_>>(),
            NUM_KECCAK_CONTROLLER_COLS,
        );
        CompressedMatrix::new(
            main,
            PaddingRow::Zero { width: NUM_KECCAK_CONTROLLER_COLS },
            padded_nb_rows,
        )
    }

    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            !shard.precompile_events.is_keccak_empty() &&
                shard.cpu_events == 0 &&
                shard.global_memory_initialize_events.is_empty() &&
                shard.global_memory_finalize_events.is_empty()
        }
    }

    fn commit_scope(&self) -> InteractionScope {
        InteractionScope::Local
    }
}

impl<AB> Air<AB> for KeccakControllerChip
where
    AB: DTAirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &KeccakControllerCols<AB::Var> = (*local).borrow();

        // Constrain that `local.is_real` is boolean.
        builder.assert_bool(local.is_real);

        builder.assert_eq(
            local.is_real * local.is_real * local.is_real,
            local.is_real * local.is_real * local.is_real,
        );

        let send_values = once(local.shard.into())
            .chain(once(local.clk.into()))
            .chain(once(AB::Expr::from_canonical_u32(0)))
            .chain(local.state_access.iter().flat_map(|s| {
                let s: CompactWord<AB::Expr> = s.prev_value.into();
                s.0.into_iter()
            }))
            .collect::<Vec<_>>();
        builder.send(
            AirInteraction::new(send_values, local.is_real.into(), InteractionKind::Keccak),
            InteractionScope::Local,
        );
        let receive_values = once(local.shard.into())
            .chain(once(local.clk.into()))
            .chain(once(AB::Expr::from_canonical_u32(24)))
            .chain(local.state_access.iter().flat_map(|s| {
                let s: CompactWord<AB::Expr> = s.access.value.into();
                s.0.into_iter()
            }))
            .collect::<Vec<_>>();
        builder.receive(
            AirInteraction::new(receive_values, local.is_real.into(), InteractionKind::Keccak),
            InteractionScope::Local,
        );

        let write_clk = local.clk + AB::F::one();
        for i in 0..STATE_NUM_WORDS {
            builder.eval_memory_access(
                local.shard,
                write_clk.clone(),
                local.state_ptr + AB::F::from_canonical_u32((i * size_of::<u32>()) as u32),
                &local.state_access[i],
                local.is_real,
            );
        }

        // Send the "receive interaction" to the global table.
        builder.send(
            AirInteraction::new(
                vec![
                    local.shard.into(),
                    local.clk.into(),
                    AB::Expr::from_canonical_u32(SyscallCode::KECCAK_PERMUTE.syscall_id()),
                    local.state_ptr.into(),
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

impl<F> BaseAir<F> for KeccakControllerChip {
    fn width(&self) -> usize {
        NUM_KECCAK_CONTROLLER_COLS
    }
}

impl KeccakControllerChip {
    fn event_to_row<F: Field>(
        &self,
        event: &KeccakPermuteEvent,
        row: &mut [F; NUM_KECCAK_CONTROLLER_COLS],
        blu: &mut impl ByteRecord,
    ) {
        let cols: &mut KeccakControllerCols<F> = row.as_mut_slice().borrow_mut();

        cols.shard = F::from_canonical_u32(event.shard);
        cols.clk = F::from_canonical_u32(event.clk);
        cols.state_ptr = F::from_canonical_u32(event.state_addr);
        cols.is_real = F::one();

        for i in 0..STATE_NUM_WORDS {
            cols.state_access[i].populate(event.state_write_records[i], blu);
        }
    }
}
