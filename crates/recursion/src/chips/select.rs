use crate::{builder::DTRecursionAirBuilder, *};
use core::borrow::Borrow;
use crate::utils::{next_power_of_two, padded_rows_threshold};
use dt_derive::AlignedBorrow;
use dt_stark::air::MachineAir;
use dt_stark::sumcheck::trace::{CompressedMatrix, PaddingRow};
use p3_air::{Air, AirBuilder, BaseAir, PairBuilder};
use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, Field};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::*;
use std::borrow::BorrowMut;

#[derive(Default)]
pub struct SelectChip;

pub const SELECT_COLS: usize = core::mem::size_of::<SelectCols<u8>>();

#[derive(AlignedBorrow, Debug, Clone, Copy)]
#[repr(C)]
pub struct SelectCols<F: Copy> {
    pub vals: SelectIo<F>,
}

pub const SELECT_PREPROCESSED_COLS: usize = core::mem::size_of::<SelectPreprocessedCols<u8>>();

#[derive(AlignedBorrow, Debug, Clone, Copy)]
#[repr(C)]
pub struct SelectPreprocessedCols<F: Copy> {
    pub is_real: F,
    pub addrs: SelectIo<Address<F>>,
    pub mult1: F,
    pub mult2: F,
}

impl<F: Field> BaseAir<F> for SelectChip {
    fn width(&self) -> usize {
        SELECT_COLS
    }
}

impl<F: Field> MachineAir<F> for SelectChip {
    type Record = ExecutionRecord<F>;

    type Program = crate::RecursionProgram<F>;

    fn name(&self) -> String {
        "Select".to_string()
    }

    fn preprocessed_width(&self) -> usize {
        SELECT_PREPROCESSED_COLS
    }

    fn preprocessed_num_rows(&self, program: &Self::Program, instrs_len: usize) -> Option<usize> {
        let fixed_log2_rows = program.fixed_log2_rows(self);
        let nb_rows = match fixed_log2_rows {
            Some(log2_rows) => 1 << log2_rows,
            None => next_power_of_two(instrs_len, None),
        };
        Some(padded_rows_threshold(nb_rows))
    }

    fn generate_preprocessed_trace(&self, program: &Self::Program) -> Option<CompressedMatrix<F>> {
        assert!(
            std::mem::size_of::<F>() == std::mem::size_of::<BabyBear>(),
            "generate_preprocessed_trace only supports 32-bit prime fields (BabyBear/KoalaBear)"
        );

        let instrs = unsafe {
            std::mem::transmute::<Vec<&SelectInstr<F>>, Vec<&SelectInstr<BabyBear>>>(
                program
                    .inner
                    .iter()
                    .filter_map(|instruction| match instruction {
                        Instruction::Select(x) => Some(x),
                        _ => None,
                    })
                    .collect::<Vec<_>>(),
            )
        };
        let padded_nb_rows = self.preprocessed_num_rows(program, instrs.len()).unwrap();
        let real_nb_rows = instrs.len();
        let mut values = vec![BabyBear::zero(); real_nb_rows * SELECT_PREPROCESSED_COLS];

        let populate_len = instrs.len() * SELECT_PREPROCESSED_COLS;
        values[..populate_len].par_chunks_mut(SELECT_PREPROCESSED_COLS).zip_eq(instrs).for_each(
            |(row, instr)| {
                let cols: &mut SelectPreprocessedCols<_> = row.borrow_mut();
                unsafe {
                    cfg_if::cfg_if! {
                        if #[cfg(feature = "koalabear")] {
                            { #[cfg(feature = "sys")] { crate::sys::select_instr_to_row_koalabear(instr, cols); } #[cfg(not(feature = "sys"))] { let _ = (instr, cols); } }
                        } else {
                            { #[cfg(feature = "sys")] { crate::sys::select_instr_to_row_babybear(instr, cols); } #[cfg(not(feature = "sys"))] { let _ = (instr, cols); } }
                        }
                    }
                }
            },
        );

        let main = RowMajorMatrix::new(
            unsafe { std::mem::transmute::<Vec<BabyBear>, Vec<F>>(values) },
            SELECT_PREPROCESSED_COLS,
        );
        Some(CompressedMatrix::new(
            main,
            PaddingRow::Zero { width: SELECT_PREPROCESSED_COLS },
            padded_nb_rows,
        ))
    }

    fn generate_dependencies(&self, _: &Self::Record, _: &mut Self::Record) {
        // This is a no-op.
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let events = &input.select_events;
        let nb_rows = next_power_of_two(events.len(), input.fixed_log2_rows(self));
        Some(padded_rows_threshold(nb_rows))
    }

    fn generate_trace(&self, input: &Self::Record, _: &mut Self::Record) -> CompressedMatrix<F> {
        assert!(
            std::mem::size_of::<F>() == std::mem::size_of::<BabyBear>(),
            "generate_trace only supports 32-bit prime fields (BabyBear/KoalaBear)"
        );

        let events = unsafe {
            std::mem::transmute::<&Vec<SelectIo<F>>, &Vec<SelectIo<BabyBear>>>(&input.select_events)
        };
        let padded_nb_rows = self.num_rows(input).unwrap();
        let real_nb_rows = events.len();
        let mut values = vec![BabyBear::zero(); real_nb_rows * SELECT_COLS];

        let populate_len = events.len() * SELECT_COLS;
        values[..populate_len].par_chunks_mut(SELECT_COLS).zip_eq(events).for_each(
            |(row, &vals)| {
                let cols: &mut SelectCols<_> = row.borrow_mut();
                unsafe {
                    cfg_if::cfg_if! {
                        if #[cfg(feature = "koalabear")] {
                            { #[cfg(feature = "sys")] { crate::sys::select_event_to_row_koalabear(&vals, cols); } #[cfg(not(feature = "sys"))] { let _ = (&vals, cols); } }
                        } else {
                            { #[cfg(feature = "sys")] { crate::sys::select_event_to_row_babybear(&vals, cols); } #[cfg(not(feature = "sys"))] { let _ = (&vals, cols); } }
                        }
                    }
                }
            },
        );

        let main = RowMajorMatrix::new(
            unsafe { std::mem::transmute::<Vec<BabyBear>, Vec<_>>(values) },
            SELECT_COLS,
        );
        CompressedMatrix::new(main, PaddingRow::Zero { width: SELECT_COLS }, padded_nb_rows)
    }

    fn included(&self, _record: &Self::Record) -> bool {
        true
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl<AB> Air<AB> for SelectChip
where
    AB: DTRecursionAirBuilder + PairBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &SelectCols<AB::Var> = (*local).borrow();
        let prep = builder.preprocessed();
        let prep_local = prep.row_slice(0);
        let prep_local: &SelectPreprocessedCols<AB::Var> = (*prep_local).borrow();

        builder.receive_single(prep_local.addrs.bit, local.vals.bit, prep_local.is_real);
        builder.receive_single(prep_local.addrs.in1, local.vals.in1, prep_local.is_real);
        builder.receive_single(prep_local.addrs.in2, local.vals.in2, prep_local.is_real);
        builder.send_single(prep_local.addrs.out1, local.vals.out1, prep_local.mult1);
        builder.send_single(prep_local.addrs.out2, local.vals.out2, prep_local.mult2);
        builder.assert_eq(
            local.vals.out1,
            local.vals.in1 + local.vals.bit * (local.vals.in2 - local.vals.in1),
        );
        builder
            .when(prep_local.is_real)
            .assert_eq(local.vals.out1 + local.vals.out2, local.vals.in1 + local.vals.in2);
    }
}
