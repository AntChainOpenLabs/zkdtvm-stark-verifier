use crate::{builder::DTRecursionAirBuilder, *};
use core::borrow::Borrow;
use instruction::{HintAddCurveInstr, HintBitsInstr, HintExt2FeltsInstr, HintInstr};
use p3_air::{Air, BaseAir, PairBuilder};
use p3_field::Field;

use super::{MemoryAccessCols, NUM_MEM_ACCESS_COLS};
use dt_core_machine::utils::{next_power_of_two, padded_rows_threshold};
use dt_derive::AlignedBorrow;
use dt_stark::{
    air::MachineAir,
    sumcheck::trace::{CompressedMatrix, PaddingRow},
};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::*;
use std::{borrow::BorrowMut, iter::zip, marker::PhantomData};

pub const NUM_VAR_MEM_ENTRIES_PER_ROW: usize = 2;

#[derive(Default)]
pub struct MemoryChip<F> {
    _marker: PhantomData<F>,
}

pub const NUM_MEM_INIT_COLS: usize = core::mem::size_of::<MemoryCols<u8>>();

#[derive(AlignedBorrow, Debug, Clone, Copy)]
#[repr(C)]
pub struct MemoryCols<F: Copy> {
    values: [Block<F>; NUM_VAR_MEM_ENTRIES_PER_ROW],
}

pub const NUM_MEM_PREPROCESSED_INIT_COLS: usize =
    core::mem::size_of::<MemoryPreprocessedCols<u8>>();

#[derive(AlignedBorrow, Debug, Clone, Copy)]
#[repr(C)]
pub struct MemoryPreprocessedCols<F: Copy> {
    accesses: [MemoryAccessCols<F>; NUM_VAR_MEM_ENTRIES_PER_ROW],
}

impl<F: Send + Sync> BaseAir<F> for MemoryChip<F> {
    fn width(&self) -> usize {
        NUM_MEM_INIT_COLS
    }
}

impl<F: Field> MachineAir<F> for MemoryChip<F> {
    type Record = crate::ExecutionRecord<F>;

    type Program = crate::RecursionProgram<F>;

    fn name(&self) -> String {
        "MemoryVar".to_string()
    }
    fn preprocessed_width(&self) -> usize {
        NUM_MEM_PREPROCESSED_INIT_COLS
    }

    fn generate_preprocessed_trace(&self, program: &Self::Program) -> Option<CompressedMatrix<F>> {
        // Allocating an intermediate `Vec` is faster.
        let accesses = program
            .inner
            .iter()
            // .par_bridge() // Using `rayon` here provides a big speedup. TODO put rayon back
            .flat_map(|instruction| match instruction {
                Instruction::Hint(HintInstr { output_addrs_mults }) |
                Instruction::HintBits(HintBitsInstr {
                    output_addrs_mults,
                    input_addr: _, // No receive interaction for the hint operation
                }) => output_addrs_mults.iter().collect(),
                Instruction::HintExt2Felts(HintExt2FeltsInstr {
                    output_addrs_mults,
                    input_addr: _, // No receive interaction for the hint operation
                }) => output_addrs_mults.iter().collect(),
                Instruction::HintAddCurve(instr) => {
                    let HintAddCurveInstr {
                        output_x_addrs_mults,
                        output_y_addrs_mults, .. // No receive interaction for the hint operation
                    } = instr.as_ref();
                    output_x_addrs_mults.iter().chain(output_y_addrs_mults.iter()).collect()
                }
                _ => vec![],
            })
            .collect::<Vec<_>>();

        let real_nb_rows = accesses.len().div_ceil(NUM_VAR_MEM_ENTRIES_PER_ROW);
        let mut total_height = match program.fixed_log2_rows(self) {
            Some(log2_rows) => 1 << log2_rows,
            None => next_power_of_two(real_nb_rows, None),
        };
        total_height = padded_rows_threshold(total_height);
        let mut values = vec![F::zero(); real_nb_rows * NUM_MEM_PREPROCESSED_INIT_COLS];

        let populate_len = accesses.len() * NUM_MEM_ACCESS_COLS;
        values[..populate_len]
            .par_chunks_mut(NUM_MEM_ACCESS_COLS)
            .zip_eq(accesses)
            .for_each(|(row, &(addr, mult))| *row.borrow_mut() = MemoryAccessCols { addr, mult });

        let main = RowMajorMatrix::new(values, NUM_MEM_PREPROCESSED_INIT_COLS);
        Some(CompressedMatrix::new(
            main,
            PaddingRow::Zero { width: NUM_MEM_PREPROCESSED_INIT_COLS },
            total_height,
        ))
    }

    fn generate_dependencies(&self, _: &Self::Record, _: &mut Self::Record) {
        // This is a no-op.
    }

    fn generate_trace(&self, input: &Self::Record, _: &mut Self::Record) -> CompressedMatrix<F> {
        let real_nb_rows = input.mem_var_events.len().div_ceil(NUM_VAR_MEM_ENTRIES_PER_ROW);
        let mut total_height = next_power_of_two(real_nb_rows, input.fixed_log2_rows(self));
        total_height = padded_rows_threshold(total_height);

        let rows: Vec<_> = input
            .mem_var_events
            .chunks(NUM_VAR_MEM_ENTRIES_PER_ROW)
            .map(|row_events| {
                let mut row = [F::zero(); NUM_MEM_INIT_COLS];
                let cols: &mut MemoryCols<_> = row.as_mut_slice().borrow_mut();
                for (cell, vals) in zip(&mut cols.values, row_events) {
                    *cell = vals.inner;
                }
                row
            })
            .collect();

        let main =
            RowMajorMatrix::new(rows.into_iter().flatten().collect::<Vec<_>>(), NUM_MEM_INIT_COLS);
        CompressedMatrix::new(main, PaddingRow::Zero { width: NUM_MEM_INIT_COLS }, total_height)
    }

    fn included(&self, _record: &Self::Record) -> bool {
        true
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl<AB> Air<AB> for MemoryChip<AB::F>
where
    AB: DTRecursionAirBuilder + PairBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &MemoryCols<AB::Var> = (*local).borrow();
        let prep = builder.preprocessed();
        let prep_local = prep.row_slice(0);
        let prep_local: &MemoryPreprocessedCols<AB::Var> = (*prep_local).borrow();

        for (value, access) in zip(local.values, prep_local.accesses) {
            builder.send_block(access.addr, value, access.mult);
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::print_stdout)]

    use p3_baby_bear::BabyBear;
    use p3_field::AbstractField;
    use p3_matrix::dense::RowMajorMatrix;

    use super::*;

    #[test]
    pub fn generate_trace() {
        let shard = ExecutionRecord::<BabyBear> {
            mem_var_events: vec![
                MemEvent { inner: BabyBear::one().into() },
                MemEvent { inner: BabyBear::one().into() },
            ],
            ..Default::default()
        };
        let chip = MemoryChip::default();
        let trace = chip.generate_trace(&shard, &mut ExecutionRecord::default());
        println!("{:?}", trace.main.values)
    }
}
