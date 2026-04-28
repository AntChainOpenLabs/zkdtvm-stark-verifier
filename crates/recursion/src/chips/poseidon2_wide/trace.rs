use super::{columns::preprocessed::Poseidon2PreprocessedColsWide, Poseidon2WideChip};
use crate::{
    instruction::Instruction::Poseidon2, ExecutionRecord, Poseidon2Io, Poseidon2SkinnyInstr,
};
use crate::{
    operations::poseidon2::WIDTH,
    utils::{next_power_of_two, padded_rows_threshold},
};
use dt_stark::{
    air::MachineAir,
    sumcheck::trace::{CompressedMatrix, PaddingRow},
};
use p3_air::BaseAir;
use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, Field};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use std::{borrow::BorrowMut, mem::size_of};
use tracing::instrument;

pub const PREPROCESSED_POSEIDON2_WIDTH: usize = size_of::<Poseidon2PreprocessedColsWide<u8>>();

impl<F: Field, const DEGREE: usize> MachineAir<F> for Poseidon2WideChip<DEGREE> {
    type Record = ExecutionRecord<F>;

    type Program = crate::RecursionProgram<F>;

    fn name(&self) -> String {
        format!("Poseidon2WideDeg{DEGREE}")
    }

    fn generate_dependencies(&self, _: &Self::Record, _: &mut Self::Record) {
        // This is a no-op.
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let events = &input.poseidon2_events;
        let nb_rows = match input.fixed_log2_rows(self) {
            Some(log2_rows) => 1 << log2_rows,
            None => next_power_of_two(events.len(), None),
        };
        Some(padded_rows_threshold(nb_rows))
    }

    #[instrument(name = "generate poseidon2 wide trace", level = "debug", skip_all, fields(rows = input.poseidon2_events.len()))]
    fn generate_trace(
        &self,
        input: &ExecutionRecord<F>,
        _output: &mut ExecutionRecord<F>,
    ) -> CompressedMatrix<F> {
        assert!(
            std::mem::size_of::<F>() == std::mem::size_of::<BabyBear>(),
            "generate_trace only supports 32-bit prime fields (BabyBear/KoalaBear)"
        );

        let events = unsafe {
            std::mem::transmute::<&Vec<Poseidon2Io<F>>, &Vec<Poseidon2Io<BabyBear>>>(
                &input.poseidon2_events,
            )
        };
        let total_height = self.num_rows(input).unwrap();
        let num_columns = <Self as BaseAir<F>>::width(self);
        let real_nb_rows = events.len();
        let mut values = vec![BabyBear::zero(); real_nb_rows * num_columns];

        let populate_perm_ffi = |input: &[BabyBear; WIDTH], input_row: &mut [BabyBear]| unsafe {
            #[cfg(feature = "sys")]
            {
            crate::sys::poseidon2_wide_event_to_row_babybear(
                input.as_ptr(),
                input_row.as_mut_ptr(),
                DEGREE == 3,
            )
;
            }
            #[cfg(not(feature = "sys"))]
            {
                unimplemented!("sys feature required for trace generation")
            }
        };

        values
            .par_chunks_mut(num_columns)
            .zip_eq(events)
            .for_each(|(row, event)| populate_perm_ffi(&event.input, row));

        let mut dummy_row = vec![BabyBear::zero(); num_columns];
        populate_perm_ffi(&[BabyBear::zero(); WIDTH], &mut dummy_row);

        let main = RowMajorMatrix::new(
            unsafe { std::mem::transmute::<Vec<BabyBear>, Vec<F>>(values) },
            num_columns,
        );
        CompressedMatrix::new(
            main,
            PaddingRow::General(unsafe { std::mem::transmute::<Vec<BabyBear>, Vec<F>>(dummy_row) }),
            total_height,
        )
    }

    fn included(&self, _record: &Self::Record) -> bool {
        true
    }

    fn local_only(&self) -> bool {
        true
    }

    fn preprocessed_width(&self) -> usize {
        PREPROCESSED_POSEIDON2_WIDTH
    }

    fn preprocessed_num_rows(&self, program: &Self::Program, instrs_len: usize) -> Option<usize> {
        let nb_rows = match program.fixed_log2_rows(self) {
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

        let instrs: Vec<&Poseidon2SkinnyInstr<BabyBear>> =
            program
                .inner
                .iter()
                .filter_map(|instruction| match instruction {
                    Poseidon2(instr) => Some(unsafe {
                        std::mem::transmute::<
                            &Poseidon2SkinnyInstr<F>,
                            &Poseidon2SkinnyInstr<BabyBear>,
                        >(instr.as_ref())
                    }),
                    _ => None,
                })
                .collect::<Vec<_>>();
        let total_height = self.preprocessed_num_rows(program, instrs.len()).unwrap();
        let real_nb_rows = instrs.len();
        let mut values = vec![BabyBear::zero(); real_nb_rows * PREPROCESSED_POSEIDON2_WIDTH];

        values.par_chunks_mut(PREPROCESSED_POSEIDON2_WIDTH).zip_eq(instrs).for_each(
            |(row, instr)| {
                let cols: &mut Poseidon2PreprocessedColsWide<_> = row.borrow_mut();
                unsafe {
                    #[cfg(feature = "sys")]
                    {
                        #[cfg(feature = "sys")]
                        {
                        crate::sys::poseidon2_wide_instr_to_row_babybear(instr, cols);
                        }
                        #[cfg(not(feature = "sys"))]
                        {
                            unimplemented!("sys feature required for trace generation")
                        }
                    }
                    #[cfg(not(feature = "sys"))]
                    {
                        // sys call omitted in verifier build
                    }
                }
            },
        );

        let main = RowMajorMatrix::new(
            unsafe { std::mem::transmute::<Vec<BabyBear>, Vec<F>>(values) },
            PREPROCESSED_POSEIDON2_WIDTH,
        );
        Some(CompressedMatrix::new(
            main,
            PaddingRow::Zero { width: PREPROCESSED_POSEIDON2_WIDTH },
            total_height,
        ))
    }
}
