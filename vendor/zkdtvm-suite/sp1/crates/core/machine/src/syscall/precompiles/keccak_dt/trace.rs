use std::borrow::BorrowMut;

use dt_core_executor::{
    events::{ByteLookupEvent, ByteRecord, KeccakPermuteEvent, PrecompileEvent},
    syscalls::SyscallCode,
    ExecutionRecord, Program,
};
use dt_stark::{
    air::MachineAir,
    sumcheck::trace::{CompressedMatrix, PaddingRow},
};
use hashbrown::HashMap;
use itertools::Itertools;
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::{ParallelIterator, ParallelSlice};

use crate::{
    syscall::precompiles::keccak_dt::columns::{KeccakPermuteCols, NUM_KECCAK_PERMUTE_COLS},
    utils::{next_power_of_two, padded_rows_threshold},
};

use super::KeccakPermuteChip;

impl<F: Field> MachineAir<F> for KeccakPermuteChip {
    type Record = ExecutionRecord;

    type Program = Program;

    fn name(&self) -> String {
        "KeccakPermute".to_string()
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _: &mut ExecutionRecord,
    ) -> CompressedMatrix<F> {
        let events = input.get_precompile_events(SyscallCode::KECCAK_PERMUTE);
        let real_nb_rows = events.len() * 24;

        let mut rows = vec![[F::zero(); NUM_KECCAK_PERMUTE_COLS]; real_nb_rows];

        for (i, (_, event)) in events.iter().enumerate() {
            let event = if let PrecompileEvent::KeccakPermute(event) = event {
                event
            } else {
                unreachable!()
            };
            self.event_to_rows(
                event,
                unsafe {
                    &mut *(rows.as_mut_ptr() as *mut [[F; NUM_KECCAK_PERMUTE_COLS]; 24]).add(i)
                },
                &mut HashMap::new(),
            );
        }

        let size_log2 = input.fixed_log2_rows::<F, _>(self);
        let mut padded_nb_rows = next_power_of_two(real_nb_rows, size_log2);
        padded_nb_rows = padded_rows_threshold(padded_nb_rows);

        let main = RowMajorMatrix::new(
            rows.into_iter().flatten().collect::<Vec<_>>(),
            NUM_KECCAK_PERMUTE_COLS,
        );
        CompressedMatrix::new(
            main,
            PaddingRow::Zero { width: NUM_KECCAK_PERMUTE_COLS },
            padded_nb_rows,
        )
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        let events = input.get_precompile_events(SyscallCode::KECCAK_PERMUTE);
        let chunk_size = std::cmp::max(events.len() / num_cpus::get(), 1);

        let blu_batches = events
            .par_chunks(chunk_size)
            .map(|events| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                events.iter().for_each(|(_, event)| {
                    let event = if let PrecompileEvent::KeccakPermute(event) = event {
                        event
                    } else {
                        unreachable!()
                    };
                    self.event_to_rows::<F>(
                        event,
                        &mut [[F::zero(); NUM_KECCAK_PERMUTE_COLS]; 24],
                        &mut blu,
                    );
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_batches.iter().collect_vec());
    }

    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            !shard.get_precompile_events(SyscallCode::KECCAK_PERMUTE).is_empty()
        }
    }
}

impl KeccakPermuteChip {
    fn event_to_rows<F: Field>(
        &self,
        event: &KeccakPermuteEvent,
        rows: &mut [[F; NUM_KECCAK_PERMUTE_COLS]; 24],
        blu: &mut impl ByteRecord,
    ) {
        let mut state = event.pre_state;

        for i in 0..24usize {
            let cols: &mut KeccakPermuteCols<F> = rows[i].as_mut_slice().borrow_mut();

            cols.shard = F::from_canonical_u32(event.shard);
            cols.clk = F::from_canonical_u32(event.clk);

            cols.keccak.populate(blu, i, &mut state);

            cols.is_real = F::one();
        }
    }
}
