use std::borrow::BorrowMut;

use dt_core_executor::{
    events::{ByteLookupEvent, Poseidon2PermuteEvent, PrecompileEvent, SyscallEvent},
    syscalls::SyscallCode,
    ExecutionRecord, Program,
};
use dt_stark::{
    air::MachineAir,
    sumcheck::trace::{CompressedMatrix, PaddingRow},
};
use p3_field::Field;
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::{ParallelBridge, ParallelIterator, ParallelSlice};

use super::{
    columns::{Poseidon2MemCols, NUM_POSEIDON2_MEM_COLS},
    Poseidon2PermuteChip, WIDTH,
};
use crate::{
    syscall::precompiles::poseidon_permute::poseidon2_inner::{generate_trace_rows, num_cols},
    utils::{padded_rows_threshold, zeroed_f_vec},
};
use dt_core_executor::events::ByteRecord;

impl<F: Field> MachineAir<F> for Poseidon2PermuteChip<F> {
    type Record = ExecutionRecord;
    type Program = Program;

    fn name(&self) -> String {
        "Poseidon2Permute".to_string()
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        let events = input.get_precompile_events(SyscallCode::POSEIDON2_PERMUTE);
        let chunk_size = std::cmp::max(events.len() / num_cpus::get(), 1);
        let blu_events: Vec<Vec<ByteLookupEvent>> = events
            .par_chunks(chunk_size)
            .map(|ops: &[(SyscallEvent, PrecompileEvent)]| {
                let mut blu = Vec::new();
                let mut chunk = zeroed_f_vec::<F>(NUM_POSEIDON2_MEM_COLS);
                ops.iter().for_each(|(_, op)| {
                    if let PrecompileEvent::Poseidon2Permute(event) = op {
                        self.populate_chunk(event, &mut chunk, &mut blu);
                    } else {
                        unreachable!();
                    }
                });
                blu
            })
            .collect();
        for blu in blu_events {
            output.add_byte_lookup_events(blu);
        }
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _: &mut ExecutionRecord,
    ) -> CompressedMatrix<F> {
        let events = input.get_precompile_events(SyscallCode::POSEIDON2_PERMUTE);
        let real_nb_rows = events.len();
        let target_log2_size = input.fixed_log2_rows(self).expect("No shape for Poseidon2Permute");
        let mut padded_nb_rows = 1 << target_log2_size;
        padded_nb_rows = padded_rows_threshold(padded_nb_rows);

        let values = vec![0u32; real_nb_rows * NUM_POSEIDON2_MEM_COLS];
        // SAFETY: F is a 4-byte field where zero bytes represent the zero element.
        // Vec<u32> and Vec<F> have identical memory layout.
        let mut values = unsafe { std::mem::transmute::<Vec<u32>, Vec<F>>(values) };

        let dummy_poseidon2_trace = generate_trace_rows::<F>(
            vec![[F::zero(); WIDTH]],
            &self.p3_poseidon2_permute.constants,
        );
        let dummy_poseidon2_inner_row = dummy_poseidon2_trace.row_slice(0);
        let mut dummy_poseidon2_row: Vec<F> = vec![F::zero(); NUM_POSEIDON2_MEM_COLS];
        dummy_poseidon2_row[..num_cols()].copy_from_slice(&dummy_poseidon2_inner_row);

        values.chunks_mut(NUM_POSEIDON2_MEM_COLS).enumerate().par_bridge().for_each(
            |(index, row)| {
                let mut new_byte_lookup_events = Vec::new();
                if let PrecompileEvent::Poseidon2Permute(event) = &events[index].1 {
                    self.populate_chunk(event, row, &mut new_byte_lookup_events);
                } else {
                    unreachable!();
                }
            },
        );

        let main = RowMajorMatrix::new(values, NUM_POSEIDON2_MEM_COLS);
        CompressedMatrix::new(main, PaddingRow::General(dummy_poseidon2_row), padded_nb_rows)
    }

    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            !shard.get_precompile_events(SyscallCode::POSEIDON2_PERMUTE).is_empty()
        }
    }
}

impl<F: Field> Poseidon2PermuteChip<F> {
    pub fn populate_chunk(
        &self,
        event: &Poseidon2PermuteEvent,
        chunk: &mut [F],
        new_byte_lookup_events: &mut Vec<ByteLookupEvent>,
    ) {
        let start_clk = event.clk;
        let shard = event.shard;

        let input_state: [F; WIDTH] = event.pre_state.map(|x| F::from_canonical_u32(x));

        let poseidon2_inner_trace =
            generate_trace_rows::<F>(vec![input_state], &self.p3_poseidon2_permute.constants);
        let poseidon2_inner_row = poseidon2_inner_trace.row_slice(0);
        let row = &mut chunk[0..NUM_POSEIDON2_MEM_COLS];
        row[..num_cols()].copy_from_slice(&poseidon2_inner_row);
        let cols: &mut Poseidon2MemCols<F> = row.borrow_mut();
        cols.shard = F::from_canonical_u32(shard);
        cols.clk = F::from_canonical_u32(start_clk);
        cols.state_addr = F::from_canonical_u32(event.state_addr);
        cols.is_real = F::one();
        for (j, read_record) in event.state_read_records.iter().enumerate() {
            cols.state_mem_read[j].populate_read(*read_record, new_byte_lookup_events);
            new_byte_lookup_events.add_u8_range_checks(&read_record.value.to_le_bytes());
        }
        for (j, write_record) in event.state_write_records.iter().enumerate() {
            cols.state_mem_write[j].populate_write(*write_record, new_byte_lookup_events);
            new_byte_lookup_events.add_u8_range_checks(&write_record.value.to_le_bytes());
        }
    }
}
