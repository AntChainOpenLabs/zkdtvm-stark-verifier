pub mod program_polyair;

use core::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};
use std::collections::HashMap;

use crate::{
    air::ProgramAirBuilder,
    utils::{next_power_of_two, padded_rows_threshold, zeroed_f_vec},
};
use dt_core_executor::{ExecutionRecord, Program};
use dt_derive::AlignedBorrow;
use dt_stark::{
    air::{DTAirBuilder, MachineAir},
    sumcheck::trace::{CompressedMatrix, PaddingRow},
};
use p3_air::{Air, BaseAir, PairBuilder};
use p3_field::Field;
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::{ParallelBridge, ParallelIterator};

use crate::adapter::instruction::InstructionCols;

/// The number of preprocessed program columns.
pub const NUM_PROGRAM_PREPROCESSED_COLS: usize = size_of::<ProgramPreprocessedCols<u8>>();

/// The number of columns for the program multiplicities.
pub const NUM_PROGRAM_MULT_COLS: usize = size_of::<ProgramMultiplicityCols<u8>>();

/// The column layout for the chip.
#[derive(AlignedBorrow, Clone, Copy, Default)]
#[repr(C)]
pub struct ProgramPreprocessedCols<T> {
    pub pc: T,
    pub instruction: InstructionCols<T>,
}

/// The column layout for the chip.
#[derive(AlignedBorrow, Clone, Copy, Default)]
#[repr(C)]
pub struct ProgramMultiplicityCols<T> {
    pub multiplicity: T,
}

/// A chip that implements addition for the opcodes ADD and ADDI.
#[derive(Default)]
pub struct ProgramChip;

impl ProgramChip {
    pub const fn new() -> Self {
        Self {}
    }
}

impl<F: Field> MachineAir<F> for ProgramChip {
    type Record = ExecutionRecord;

    type Program = Program;

    fn name(&self) -> String {
        "Program".to_string()
    }

    fn preprocessed_width(&self) -> usize {
        NUM_PROGRAM_PREPROCESSED_COLS
    }

    fn generate_preprocessed_trace(&self, program: &Self::Program) -> Option<CompressedMatrix<F>> {
        debug_assert!(
            !program.instructions.is_empty() || program.preprocessed_shape.is_some(),
            "empty program"
        );
        let nb_rows = program.instructions.len();
        let size_log2 = program.fixed_log2_rows::<F, _>(self);
        let padded_nb_rows = padded_rows_threshold(next_power_of_two(nb_rows, size_log2));

        let mut values = zeroed_f_vec(nb_rows * NUM_PROGRAM_PREPROCESSED_COLS);
        let chunk_size = std::cmp::max((nb_rows + 1) / num_cpus::get(), 1);

        values
            .chunks_mut(chunk_size * NUM_PROGRAM_PREPROCESSED_COLS)
            .enumerate()
            .par_bridge()
            .for_each(|(i, rows)| {
                rows.chunks_mut(NUM_PROGRAM_PREPROCESSED_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut ProgramPreprocessedCols<F> = row.borrow_mut();
                    let instruction = &program.instructions[idx];
                    let pc = program.pc_base + (idx as u32 * 4);
                    cols.pc = F::from_canonical_u32(pc);
                    cols.instruction.populate(instruction);
                });
            });

        let main = RowMajorMatrix::new(values, NUM_PROGRAM_PREPROCESSED_COLS);
        let padding_row = if nb_rows > 0 {
            PaddingRow::General(main.row(0).collect::<Vec<_>>())
        } else {
            PaddingRow::Zero { width: NUM_PROGRAM_PREPROCESSED_COLS }
        };
        Some(CompressedMatrix::new(main, padding_row, padded_nb_rows))
    }

    fn generate_dependencies(&self, _input: &ExecutionRecord, _output: &mut ExecutionRecord) {
        // Do nothing since this chip has no dependencies.
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _output: &mut ExecutionRecord,
    ) -> CompressedMatrix<F> {
        // Generate the trace rows for each event.

        // Collect the number of times each instruction is called from the cpu events.
        // Store it as a map of PC -> count.
        let mut instruction_counts = HashMap::new();
        input.all_event_pcs().for_each(|pc| {
            instruction_counts.entry(pc).and_modify(|count| *count += 1).or_insert(1);
        });

        let rows = input
            .program
            .instructions
            .clone()
            .into_iter()
            .enumerate()
            .map(|(i, _)| {
                let pc = input.program.pc_base + (i as u32 * 4);
                let mut row = [F::zero(); NUM_PROGRAM_MULT_COLS];
                let cols: &mut ProgramMultiplicityCols<F> = row.as_mut_slice().borrow_mut();
                cols.multiplicity =
                    F::from_canonical_usize(*instruction_counts.get(&pc).unwrap_or(&0));
                row
            })
            .collect::<Vec<_>>();

        let real_nb_rows = rows.len();
        let size_log2 = input.fixed_log2_rows::<F, _>(self);
        let padded_nb_rows = padded_rows_threshold(next_power_of_two(real_nb_rows, size_log2));

        let main = RowMajorMatrix::new(
            rows.into_iter().flatten().collect::<Vec<_>>(),
            NUM_PROGRAM_MULT_COLS,
        );
        CompressedMatrix::new(
            main,
            PaddingRow::Zero { width: NUM_PROGRAM_MULT_COLS },
            padded_nb_rows,
        )
    }

    fn included(&self, _: &Self::Record) -> bool {
        true
    }
}

impl<F> BaseAir<F> for ProgramChip {
    fn width(&self) -> usize {
        NUM_PROGRAM_MULT_COLS
    }
}

impl<AB> Air<AB> for ProgramChip
where
    AB: DTAirBuilder + PairBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let preprocessed = builder.preprocessed();

        let prep_local = preprocessed.row_slice(0);
        let prep_local: &ProgramPreprocessedCols<AB::Var> = (*prep_local).borrow();
        let mult_local = main.row_slice(0);
        let mult_local: &ProgramMultiplicityCols<AB::Var> = (*mult_local).borrow();

        // Constrain the interaction with CPU table
        builder.receive_program(prep_local.pc, prep_local.instruction, mult_local.multiplicity);
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::print_stdout)]

    use std::sync::Arc;

    use p3_baby_bear::BabyBear;

    use dt_core_executor::{ExecutionRecord, Instruction, Opcode, Program};
    use dt_stark::{air::MachineAir, sumcheck::trace::CompressedMatrix};
    use p3_matrix::dense::RowMajorMatrix;

    use crate::program::ProgramChip;

    #[test]
    fn generate_trace() {
        // main:
        //     addi x29, x0, 5
        //     addi x30, x0, 37
        //     add x31, x30, x29
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, 5, false, true),
            Instruction::new(Opcode::ADD, 30, 0, 37, false, true),
            Instruction::new(Opcode::ADD, 31, 30, 29, false, false),
        ];
        let shard = ExecutionRecord {
            program: Arc::new(Program::new(instructions, 0, 0)),
            ..Default::default()
        };
        let chip = ProgramChip::new();
        let trace: CompressedMatrix<BabyBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default());
        println!("{:?}", trace)
    }
}
