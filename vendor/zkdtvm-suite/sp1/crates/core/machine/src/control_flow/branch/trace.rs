use std::borrow::BorrowMut;

use dt_core_executor::{
    events::{BranchEvent, ByteLookupEvent, ByteRecord},
    BTypeRecord, ExecutionRecord, Opcode, Program,
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

use super::{BranchChip, BranchColumns, NUM_BRANCH_COLS};

impl<F: Field> MachineAir<F> for BranchChip {
    type Record = ExecutionRecord;

    type Program = Program;

    fn name(&self) -> String {
        "Branch".to_string()
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        output: &mut ExecutionRecord,
    ) -> CompressedMatrix<F> {
        let nb_rows = input.branch_events.len();
        let size_log2 = input.fixed_log2_rows::<F, _>(self);
        let padded_nb_rows = padded_rows_threshold(next_power_of_two(nb_rows, size_log2));
        let chunk_size = std::cmp::max(nb_rows / num_cpus::get(), 1);
        let mut values = zeroed_f_vec(nb_rows * NUM_BRANCH_COLS);
        let shard = input.execution_shard();
        let blu_events = values
            .chunks_mut(chunk_size * NUM_BRANCH_COLS)
            .enumerate()
            .par_bridge()
            .map(|(i, rows)| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                rows.chunks_mut(NUM_BRANCH_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut BranchColumns<F> = row.borrow_mut();
                    let (record, event) = &input.branch_events[idx];
                    self.event_to_row(record, event, cols, &mut blu, shard);
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_events.iter().collect_vec());

        let main = RowMajorMatrix::new(values, NUM_BRANCH_COLS);
        CompressedMatrix::new(main, PaddingRow::Zero { width: NUM_BRANCH_COLS }, padded_nb_rows)
    }

    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            !shard.branch_events.is_empty()
        }
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl BranchChip {
    /// Create a row from an event.
    pub(crate) fn event_to_row<F: Field>(
        &self,
        record: &BTypeRecord,
        event: &BranchEvent,
        cols: &mut BranchColumns<F>,
        blu: &mut HashMap<ByteLookupEvent, usize>,
        shard: u32,
    ) {
        cols.is_beq = F::from_bool(matches!(event.opcode, Opcode::BEQ));
        cols.is_bne = F::from_bool(matches!(event.opcode, Opcode::BNE));
        cols.is_blt = F::from_bool(matches!(event.opcode, Opcode::BLT));
        cols.is_bge = F::from_bool(matches!(event.opcode, Opcode::BGE));
        cols.is_bltu = F::from_bool(matches!(event.opcode, Opcode::BLTU));
        cols.is_bgeu = F::from_bool(matches!(event.opcode, Opcode::BGEU));

        cols.cpu_state.populate(blu, record.clk, event.pc, shard);
        cols.mem_ops.populate(blu, *record);

        let a_eq_b = event.a == event.b;

        let use_signed_comparison = matches!(event.opcode, Opcode::BLT | Opcode::BGE);

        let a_lt_b = if use_signed_comparison {
            (event.a as i32) < (event.b as i32)
        } else {
            event.a < event.b
        };
        let a_gt_b = if use_signed_comparison {
            (event.a as i32) > (event.b as i32)
        } else {
            event.a > event.b
        };

        cols.a_eq_b = F::from_bool(a_eq_b);
        cols.a_lt_b = F::from_bool(a_lt_b);
        cols.a_gt_b = F::from_bool(a_gt_b);

        let branching = match event.opcode {
            Opcode::BEQ => a_eq_b,
            Opcode::BNE => !a_eq_b,
            Opcode::BLT | Opcode::BLTU => a_lt_b,
            Opcode::BGE | Opcode::BGEU => a_eq_b || a_gt_b,
            _ => unreachable!(),
        };

        cols.pc = event.pc.into();

        cols.pc_range_checker.populate(cols.pc, blu);
        //add op: eval uses is_branching as mult, so only add byte lookups when branching
        let offset = if branching { event.c } else { 4 };
        if branching {
            cols.add_op.populate(blu, event.pc, offset);
        } else {
            let mut dummy_blu: Vec<dt_core_executor::events::ByteLookupEvent> = vec![];
            cols.add_op.populate(&mut dummy_blu, event.pc, offset);
        }
        // Range check next_pc (add_op.value) — must use the actual computed value.
        cols.next_pc_range_checker.populate(cols.add_op.value, blu);
        //lt op
        cols.compare_operation.populate(
            blu,
            a_lt_b as u32,
            event.a,
            event.b,
            use_signed_comparison,
        );

        if branching {
            cols.is_branching = F::one();
        }
    }
}
