use crate::{builder::DTRecursionAirBuilder, *};
use core::borrow::Borrow;
use itertools::Itertools;
use p3_air::{Air, BaseAir, PairBuilder};
use p3_field::Field;

use crate::utils::{next_power_of_two, padded_rows_threshold};
use dt_derive::AlignedBorrow;
use dt_stark::air::MachineAir;
use dt_stark::sumcheck::trace::{CompressedMatrix, PaddingRow};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use std::{borrow::BorrowMut, iter::zip, marker::PhantomData};

use super::MemoryAccessCols;

pub const NUM_CONST_MEM_ENTRIES_PER_ROW: usize = 2;

#[derive(Default)]
pub struct MemoryChip<F> {
    _marker: PhantomData<F>,
}

pub const NUM_MEM_INIT_COLS: usize = core::mem::size_of::<MemoryCols<u8>>();

#[derive(AlignedBorrow, Debug, Clone, Copy)]
#[repr(C)]
pub struct MemoryCols<F: Copy> {
    // At least one column is required, otherwise a bunch of things break.
    _nothing: F,
}

pub const NUM_MEM_PREPROCESSED_INIT_COLS: usize =
    core::mem::size_of::<MemoryPreprocessedCols<u8>>();

#[derive(AlignedBorrow, Debug, Clone, Copy)]
#[repr(C)]
pub struct MemoryPreprocessedCols<F: Copy> {
    values_and_accesses: [(Block<F>, MemoryAccessCols<F>); NUM_CONST_MEM_ENTRIES_PER_ROW],
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
        "MemoryConst".to_string()
    }
    fn preprocessed_width(&self) -> usize {
        NUM_MEM_PREPROCESSED_INIT_COLS
    }

    fn generate_preprocessed_trace(&self, program: &Self::Program) -> Option<CompressedMatrix<F>> {
        let rows: Vec<_> = program
            .inner
            .iter()
            .filter_map(|instruction| match instruction {
                Instruction::Mem(MemInstr { addrs, vals, mult, kind }) => {
                    let mult = mult.to_owned();
                    let mult = match kind {
                        MemAccessKind::Read => -mult,
                        MemAccessKind::Write => mult,
                    };

                    Some((vals.inner, MemoryAccessCols { addr: addrs.inner, mult }))
                }
                _ => None,
            })
            .chunks(NUM_CONST_MEM_ENTRIES_PER_ROW)
            .into_iter()
            .map(|row_vs_as| {
                let mut row = [F::zero(); NUM_MEM_PREPROCESSED_INIT_COLS];
                let cols: &mut MemoryPreprocessedCols<_> = row.as_mut_slice().borrow_mut();
                for (cell, access) in zip(&mut cols.values_and_accesses, row_vs_as) {
                    *cell = access;
                }
                row
            })
            .collect();

        let real_nb_rows = rows.len();
        let mut total_height = next_power_of_two(real_nb_rows, program.fixed_log2_rows(self));
        total_height = padded_rows_threshold(total_height);

        let main = RowMajorMatrix::new(
            rows.into_iter().flatten().collect::<Vec<_>>(),
            NUM_MEM_PREPROCESSED_INIT_COLS,
        );
        Some(CompressedMatrix::new(
            main,
            PaddingRow::Zero { width: NUM_MEM_PREPROCESSED_INIT_COLS },
            total_height,
        ))
    }

    fn generate_dependencies(&self, _: &Self::Record, _: &mut Self::Record) {
        // This is a no-op.
    }

    #[allow(clippy::manual_repeat_n)]
    fn generate_trace(&self, input: &Self::Record, _: &mut Self::Record) -> CompressedMatrix<F> {
        let real_nb_rows = input
            .mem_const_count
            .checked_sub(1)
            .map(|x| x / NUM_CONST_MEM_ENTRIES_PER_ROW + 1)
            .unwrap_or_default();
        let mut total_height = next_power_of_two(real_nb_rows, input.fixed_log2_rows(self));
        total_height = padded_rows_threshold(total_height);

        let rows = std::iter::repeat([F::zero(); NUM_MEM_INIT_COLS])
            .take(real_nb_rows)
            .collect::<Vec<_>>();
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
        let prep = builder.preprocessed();
        let prep_local = prep.row_slice(0);
        let prep_local: &MemoryPreprocessedCols<AB::Var> = (*prep_local).borrow();

        for (value, access) in prep_local.values_and_accesses {
            builder.send_block(access.addr, value, access.mult);
        }
    }
}
