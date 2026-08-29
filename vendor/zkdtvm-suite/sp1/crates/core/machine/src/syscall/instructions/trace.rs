use std::borrow::BorrowMut;

use dt_core_executor::{
    events::{ByteLookupEvent, ByteRecord, SyscallEvent},
    syscalls::SyscallCode,
    ExecutionRecord, Program, RTypeRecord,
};
use dt_stark::{
    air::MachineAir,
    sumcheck::trace::{CompressedMatrix, PaddingRow},
};
use hashbrown::HashMap;
use itertools::Itertools;
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;
use rayon::iter::{ParallelBridge, ParallelIterator};

use crate::utils::{next_power_of_two, padded_rows_threshold, zeroed_f_vec};

use super::{
    columns::{SyscallInstrColumns, NUM_SYSCALL_INSTR_COLS},
    SyscallInstrsChip,
};

impl<F: Field> MachineAir<F> for SyscallInstrsChip {
    type Record = ExecutionRecord;

    type Program = Program;

    fn name(&self) -> String {
        "SyscallInstrs".to_string()
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        output: &mut ExecutionRecord,
    ) -> CompressedMatrix<F> {
        let real_nb_rows = input.syscall_events.len();
        let chunk_size = std::cmp::max(real_nb_rows / num_cpus::get(), 1);
        let size_log2 = input.fixed_log2_rows::<F, _>(self);
        let mut padded_nb_rows = next_power_of_two(real_nb_rows, size_log2);
        padded_nb_rows = padded_rows_threshold(padded_nb_rows);
        let mut values = zeroed_f_vec(real_nb_rows * NUM_SYSCALL_INSTR_COLS);

        let blu_events = values
            .chunks_mut(chunk_size * NUM_SYSCALL_INSTR_COLS)
            .enumerate()
            .par_bridge()
            .map(|(i, rows)| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                rows.chunks_mut(NUM_SYSCALL_INSTR_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut SyscallInstrColumns<F> = row.borrow_mut();
                    let (record, event) = &input.syscall_events[idx];
                    self.event_to_row(record, event, cols, &mut blu);
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_events.iter().collect_vec());

        let main = RowMajorMatrix::new(values, NUM_SYSCALL_INSTR_COLS);
        CompressedMatrix::new(
            main,
            PaddingRow::Zero { width: NUM_SYSCALL_INSTR_COLS },
            padded_nb_rows,
        )
    }

    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            !shard.syscall_events.is_empty()
        }
    }
}

impl SyscallInstrsChip {
    pub(super) fn event_to_row<F: Field>(
        &self,
        record: &RTypeRecord,
        event: &SyscallEvent,
        cols: &mut SyscallInstrColumns<F>,
        blu: &mut impl ByteRecord,
    ) {
        cols.is_real = F::one();
        cols.cpu_state.populate(blu, event.clk, event.pc, event.shard);
        cols.mem_ops.populate(blu, *record);

        cols.next_pc = F::from_canonical_u32(event.next_pc);
        let op_a_prev_value = record.a.previous_record().value;
        // let op_a_value = rtype_record.op_a_value();
        let op_a_prev_bytes = op_a_prev_value.to_le_bytes();
        // let op_a_bytes = op_a_value.to_le_bytes();
        // let syscall_id = cols.op_a_access.prev_value[0];
        let syscall_id = F::from_canonical_u8(op_a_prev_bytes[0]);
        let num_cycles = F::from_canonical_u8(op_a_prev_bytes[2]);
        // let num_cycles = cols.op_a_access.prev_value[2];

        cols.num_extra_cycles = num_cycles;
        cols.is_halt =
            F::from_bool(syscall_id == F::from_canonical_u32(SyscallCode::HALT.syscall_id()));

        // Populate `is_enter_unconstrained`.
        cols.is_enter_unconstrained.populate_from_field_element(
            syscall_id - F::from_canonical_u32(SyscallCode::ENTER_UNCONSTRAINED.syscall_id()),
        );

        // Populate `is_hint_len`.
        cols.is_hint_len.populate_from_field_element(
            syscall_id - F::from_canonical_u32(SyscallCode::HINT_LEN.syscall_id()),
        );

        // Populate `is_halt`.
        cols.is_halt_check.populate_from_field_element(
            syscall_id - F::from_canonical_u32(SyscallCode::HALT.syscall_id()),
        );

        // Populate `is_commit`.
        cols.is_commit.populate_from_field_element(
            syscall_id - F::from_canonical_u32(SyscallCode::COMMIT.syscall_id()),
        );

        // Populate `is_commit_deferred_proofs`.
        cols.is_commit_deferred_proofs.populate_from_field_element(
            syscall_id - F::from_canonical_u32(SyscallCode::COMMIT_DEFERRED_PROOFS.syscall_id()),
        );

        // If the syscall is `COMMIT` or `COMMIT_DEFERRED_PROOFS`, set the index bitmap and
        // digest word.
        if syscall_id == F::from_canonical_u32(SyscallCode::COMMIT.syscall_id()) ||
            syscall_id == F::from_canonical_u32(SyscallCode::COMMIT_DEFERRED_PROOFS.syscall_id())
        {
            let digest_idx = record.op_b_value() as usize;
            cols.index_bitmap[digest_idx] = F::one();
        }

        // For halt and commit deferred proofs syscalls, we need to baby bear range check one of
        // it's operands.
        if cols.is_halt == F::one() {
            cols.operand_to_check = event.arg1.into();
            cols.operand_range_check_cols.populate(cols.operand_to_check, blu);
            cols.ecall_range_check_operand = F::one();
        }

        if syscall_id == F::from_canonical_u32(SyscallCode::COMMIT_DEFERRED_PROOFS.syscall_id()) {
            cols.operand_to_check = event.arg2.into();
            cols.operand_range_check_cols.populate(cols.operand_to_check, blu);
            cols.ecall_range_check_operand = F::one();
        }
    }
}
