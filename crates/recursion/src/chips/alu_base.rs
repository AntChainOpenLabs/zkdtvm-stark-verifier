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
use std::{borrow::BorrowMut, iter::zip};

pub const NUM_BASE_ALU_ENTRIES_PER_ROW: usize = 4;

#[derive(Default)]
pub struct BaseAluChip;

pub const NUM_BASE_ALU_COLS: usize = core::mem::size_of::<BaseAluCols<u8>>();

#[derive(AlignedBorrow, Debug, Clone, Copy)]
#[repr(C)]
pub struct BaseAluCols<F: Copy> {
    pub values: [BaseAluValueCols<F>; NUM_BASE_ALU_ENTRIES_PER_ROW],
}

pub const NUM_BASE_ALU_VALUE_COLS: usize = core::mem::size_of::<BaseAluValueCols<u8>>();

#[derive(AlignedBorrow, Debug, Clone, Copy)]
#[repr(C)]
pub struct BaseAluValueCols<F: Copy> {
    pub vals: BaseAluIo<F>,
}

pub const NUM_BASE_ALU_PREPROCESSED_COLS: usize =
    core::mem::size_of::<BaseAluPreprocessedCols<u8>>();

#[derive(AlignedBorrow, Debug, Clone, Copy)]
#[repr(C)]
pub struct BaseAluPreprocessedCols<F: Copy> {
    pub accesses: [BaseAluAccessCols<F>; NUM_BASE_ALU_ENTRIES_PER_ROW],
}

pub const NUM_BASE_ALU_ACCESS_COLS: usize = core::mem::size_of::<BaseAluAccessCols<u8>>();

#[derive(AlignedBorrow, Debug, Clone, Copy)]
#[repr(C)]
pub struct BaseAluAccessCols<F: Copy> {
    pub addrs: BaseAluIo<Address<F>>,
    pub is_add: F,
    pub is_sub: F,
    pub is_mul: F,
    pub is_div: F,
    pub mult: F,
}

impl<F: Field> BaseAir<F> for BaseAluChip {
    fn width(&self) -> usize {
        NUM_BASE_ALU_COLS
    }
}

impl<F: Field> MachineAir<F> for BaseAluChip {
    type Record = ExecutionRecord<F>;

    type Program = crate::RecursionProgram<F>;

    fn name(&self) -> String {
        "BaseAlu".to_string()
    }

    fn preprocessed_width(&self) -> usize {
        NUM_BASE_ALU_PREPROCESSED_COLS
    }

    fn preprocessed_num_rows(&self, program: &Self::Program, instrs_len: usize) -> Option<usize> {
        let nb_rows = instrs_len.div_ceil(NUM_BASE_ALU_ENTRIES_PER_ROW);
        let fixed_log2_rows = program.fixed_log2_rows(self);
        let nb_rows = match fixed_log2_rows {
            Some(log2_rows) => 1 << log2_rows,
            None => next_power_of_two(nb_rows, None),
        };
        Some(padded_rows_threshold(nb_rows))
    }

    fn generate_preprocessed_trace(&self, program: &Self::Program) -> Option<CompressedMatrix<F>> {
        assert!(
            std::mem::size_of::<F>() == std::mem::size_of::<BabyBear>(),
            "generate_preprocessed_trace only supports 32-bit prime fields (BabyBear/KoalaBear)"
        );

        let instrs = unsafe {
            std::mem::transmute::<Vec<&BaseAluInstr<F>>, Vec<&BaseAluInstr<BabyBear>>>(
                program
                    .inner
                    .iter()
                    .filter_map(|instruction| match instruction {
                        Instruction::BaseAlu(x) => Some(x),
                        _ => None,
                    })
                    .collect::<Vec<_>>(),
            )
        };
        let padded_nb_rows = self.preprocessed_num_rows(program, instrs.len()).unwrap();
        let real_nb_rows = instrs.len().div_ceil(NUM_BASE_ALU_ENTRIES_PER_ROW);
        let mut values = vec![BabyBear::zero(); real_nb_rows * NUM_BASE_ALU_PREPROCESSED_COLS];

        // Generate the trace rows (only real rows).
        let populate_len = instrs.len() * NUM_BASE_ALU_ACCESS_COLS;
        values[..populate_len].par_chunks_mut(NUM_BASE_ALU_ACCESS_COLS).zip_eq(instrs).for_each(
            |(row, instr)| {
                let access: &mut BaseAluAccessCols<_> = row.borrow_mut();
                unsafe {
                    cfg_if::cfg_if! {
                        if #[cfg(feature = "koalabear")] {
                            { #[cfg(feature = "sys")] { crate::sys::alu_base_instr_to_row_koalabear(instr, access); } #[cfg(not(feature = "sys"))] { let _ = (instr, access); } }
                        } else {
                            { #[cfg(feature = "sys")] { crate::sys::alu_base_instr_to_row_babybear(instr, access); } #[cfg(not(feature = "sys"))] { let _ = (instr, access); } }
                        }
                    }
                }
            },
        );

        let main = RowMajorMatrix::new(
            unsafe { std::mem::transmute::<Vec<BabyBear>, Vec<F>>(values) },
            NUM_BASE_ALU_PREPROCESSED_COLS,
        );
        Some(CompressedMatrix::new(
            main,
            PaddingRow::Zero { width: NUM_BASE_ALU_PREPROCESSED_COLS },
            padded_nb_rows,
        ))
    }

    fn generate_dependencies(&self, _: &Self::Record, _: &mut Self::Record) {
        // This is a no-op.
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows = input.base_alu_events.len().div_ceil(NUM_BASE_ALU_ENTRIES_PER_ROW);
        let fixed_log2_rows = input.fixed_log2_rows(self);
        let nb_rows = match fixed_log2_rows {
            Some(log2_rows) => 1 << log2_rows,
            None => next_power_of_two(nb_rows, None),
        };
        Some(padded_rows_threshold(nb_rows))
    }

    fn generate_trace(&self, input: &Self::Record, _: &mut Self::Record) -> CompressedMatrix<F> {
        assert!(
            std::mem::size_of::<F>() == std::mem::size_of::<BabyBear>(),
            "generate_trace only supports 32-bit prime fields (BabyBear/KoalaBear)"
        );

        let events = unsafe {
            std::mem::transmute::<&Vec<BaseAluIo<F>>, &Vec<BaseAluIo<BabyBear>>>(
                &input.base_alu_events,
            )
        };
        let padded_nb_rows = self.num_rows(input).unwrap();
        let real_nb_rows = events.len().div_ceil(NUM_BASE_ALU_ENTRIES_PER_ROW);
        let mut values = vec![BabyBear::zero(); real_nb_rows * NUM_BASE_ALU_COLS];

        // Generate only real rows; padding represented by PaddingRow::Zero.
        let populate_len = events.len() * NUM_BASE_ALU_VALUE_COLS;
        values[..populate_len].par_chunks_mut(NUM_BASE_ALU_VALUE_COLS).zip_eq(events).for_each(
            |(row, &vals)| {
                let cols: &mut BaseAluValueCols<_> = row.borrow_mut();
                unsafe {
                    cfg_if::cfg_if! {
                        if #[cfg(feature = "koalabear")] {
                            { #[cfg(feature = "sys")] { crate::sys::alu_base_event_to_row_koalabear(&vals, cols); } #[cfg(not(feature = "sys"))] { let _ = (&vals, cols); } }
                        } else {
                            { #[cfg(feature = "sys")] { crate::sys::alu_base_event_to_row_babybear(&vals, cols); } #[cfg(not(feature = "sys"))] { let _ = (&vals, cols); } }
                        }
                    }
                }
            },
        );

        let main = RowMajorMatrix::new(
            unsafe { std::mem::transmute::<Vec<BabyBear>, Vec<F>>(values) },
            NUM_BASE_ALU_COLS,
        );
        CompressedMatrix::new(main, PaddingRow::Zero { width: NUM_BASE_ALU_COLS }, padded_nb_rows)
    }

    fn included(&self, _record: &Self::Record) -> bool {
        true
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl<AB> Air<AB> for BaseAluChip
where
    AB: DTRecursionAirBuilder + PairBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &BaseAluCols<AB::Var> = (*local).borrow();
        let prep = builder.preprocessed();
        let prep_local = prep.row_slice(0);
        let prep_local: &BaseAluPreprocessedCols<AB::Var> = (*prep_local).borrow();

        for (
            BaseAluValueCols { vals: BaseAluIo { out, in1, in2 } },
            BaseAluAccessCols { addrs, is_add, is_sub, is_mul, is_div, mult },
        ) in zip(local.values, prep_local.accesses)
        {
            // Check exactly one flag is enabled.
            let is_real = is_add + is_sub + is_mul + is_div;
            builder.assert_bool(is_real.clone());

            builder.when(is_add).assert_eq(in1 + in2, out);
            builder.when(is_sub).assert_eq(in1, in2 + out);
            builder.when(is_mul).assert_eq(out, in1 * in2);
            builder.when(is_div).assert_eq(in2 * out, in1);

            builder.receive_single(addrs.in1, in1, is_real.clone());

            builder.receive_single(addrs.in2, in2, is_real);

            builder.send_single(addrs.out, out, mult);
        }
    }
}
