use std::borrow::BorrowMut;

use dt_core_executor::{ByteOpcode, ExecutionRecord, Program};
use dt_stark::{air::MachineAir, sumcheck::trace::CompressedMatrix};
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;

use crate::{bytes::get_index_from_range_value, utils::zeroed_f_vec};

use super::{
    columns::{ByteMultCols, NUM_BYTE_MULT_COLS, NUM_BYTE_PREPROCESSED_COLS},
    ByteChip,
};

pub const NUM_ROWS: usize = 1 << 16;

impl<F: Field> MachineAir<F> for ByteChip<F> {
    type Record = ExecutionRecord;

    type Program = Program;

    fn name(&self) -> String {
        "Byte".to_string()
    }

    fn preprocessed_width(&self) -> usize {
        NUM_BYTE_PREPROCESSED_COLS
    }

    fn generate_preprocessed_trace(&self, _program: &Self::Program) -> Option<CompressedMatrix<F>> {
        let trace = Self::trace();
        Some(CompressedMatrix::from_full_matrix_no_padding(trace))
    }

    fn generate_dependencies(&self, _input: &ExecutionRecord, _output: &mut ExecutionRecord) {
        // Do nothing since this chip has no dependencies.
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _output: &mut ExecutionRecord,
    ) -> CompressedMatrix<F> {
        let mut trace =
            RowMajorMatrix::new(zeroed_f_vec(NUM_BYTE_MULT_COLS * NUM_ROWS), NUM_BYTE_MULT_COLS);

        for (lookup, mult) in input.byte_lookups.iter() {
            let (index, row) = match lookup.opcode {
                ByteOpcode::BitRange => {
                    if lookup.a2 >= 16 {
                        (ByteOpcode::U16Range as usize, lookup.a1 as usize)
                    } else {
                        (
                            ByteOpcode::BitRange as usize,
                            get_index_from_range_value(lookup.a1, lookup.a2),
                        )
                    }
                }
                ByteOpcode::U16Range => (ByteOpcode::U16Range as usize, lookup.a1 as usize),
                _ => {
                    (lookup.opcode as usize, (((lookup.b as u16) << 8) + lookup.c as u16) as usize)
                }
            };

            let cols: &mut ByteMultCols<F> = trace.row_mut(row).borrow_mut();
            cols.multiplicities[index] += F::from_canonical_usize(*mult);
        }

        CompressedMatrix::from_full_matrix_no_padding(trace)
    }

    fn included(&self, _shard: &Self::Record) -> bool {
        true
    }
}
