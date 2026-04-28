use super::columns::preprocessed::Poseidon2PreprocessedCols;
use crate::{
    chips::poseidon2_skinny_kb::{
        columns::NUM_POSEIDON2_COLS, Poseidon2SkinnyKbChip, NUM_EXTERNAL_ROUNDS,
    },
    instruction::Instruction::Poseidon2,
    ExecutionRecord, Poseidon2Io, Poseidon2SkinnyInstr,
};
use crate::utils::{next_power_of_two, padded_rows_threshold};
use dt_stark::air::MachineAir;
use dt_stark::sumcheck::trace::{CompressedMatrix, PaddingRow};
use itertools::Itertools;
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;
use std::{borrow::BorrowMut, mem::size_of};
use tracing::instrument;

const PREPROCESSED_POSEIDON2_WIDTH: usize = size_of::<Poseidon2PreprocessedCols<u8>>();
pub const OUTPUT_ROUND_IDX: usize = NUM_EXTERNAL_ROUNDS + 2;

impl<F: Field, const DEGREE: usize> MachineAir<F> for Poseidon2SkinnyKbChip<DEGREE> {
    type Record = ExecutionRecord<F>;

    type Program = crate::RecursionProgram<F>;

    fn name(&self) -> String {
        format!("Poseidon2SkinnyKbDeg{DEGREE}")
    }

    fn generate_dependencies(&self, _: &Self::Record, _: &mut Self::Record) {
        // This is a no-op.
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let events = &input.poseidon2_events;
        let nb_rows =
            next_power_of_two(events.len() * (OUTPUT_ROUND_IDX + 1), input.fixed_log2_rows(self));
        Some(padded_rows_threshold(nb_rows))
    }

    #[instrument(name = "generate poseidon2 skinny kb trace", level = "debug", skip_all, fields(rows = input.poseidon2_events.len()))]
    fn generate_trace(
        &self,
        input: &ExecutionRecord<F>,
        _output: &mut ExecutionRecord<F>,
    ) -> CompressedMatrix<F> {
        assert!(
            std::mem::size_of::<F>() == std::mem::size_of::<BabyBear>(),
            "generate_trace only supports 32-bit prime fields (BabyBear/KoalaBear)"
        );

        let mut rows = Vec::new();

        let events = unsafe {
            std::mem::transmute::<&Vec<Poseidon2Io<F>>, &Vec<Poseidon2Io<BabyBear>>>(
                &input.poseidon2_events,
            )
        };
        for event in events {
            let mut row_add = [[BabyBear::zero(); NUM_POSEIDON2_COLS]; NUM_EXTERNAL_ROUNDS + 3];
            unsafe {
                #[cfg(feature = "sys")]
                {
                    #[cfg(feature = "sys")]
                    {
                    crate::sys::poseidon2_skinny_event_to_row_koalabear(
                    event,
                    row_add.as_mut_ptr() as *mut u8,
                );
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
            rows.extend(row_add.into_iter());
        }

        let total_height = self.num_rows(input).unwrap();

        let main = RowMajorMatrix::new(
            unsafe {
                std::mem::transmute::<Vec<BabyBear>, Vec<F>>(
                    rows.into_iter().flatten().collect::<Vec<BabyBear>>(),
                )
            },
            NUM_POSEIDON2_COLS,
        );
        CompressedMatrix::new(main, PaddingRow::Zero { width: NUM_POSEIDON2_COLS }, total_height)
    }

    fn included(&self, _record: &Self::Record) -> bool {
        true
    }

    fn preprocessed_width(&self) -> usize {
        PREPROCESSED_POSEIDON2_WIDTH
    }

    fn preprocessed_num_rows(&self, program: &Self::Program, instrs_len: usize) -> Option<usize> {
        let nb_rows = next_power_of_two(instrs_len, program.fixed_log2_rows(self));
        Some(padded_rows_threshold(nb_rows))
    }

    fn generate_preprocessed_trace(&self, program: &Self::Program) -> Option<CompressedMatrix<F>> {
        assert!(
            std::mem::size_of::<F>() == std::mem::size_of::<BabyBear>(),
            "generate_preprocessed_trace only supports 32-bit prime fields (BabyBear/KoalaBear)"
        );

        let instructions = program.inner.iter().filter_map(|instruction| match instruction {
            Poseidon2(instr) => Some(unsafe {
                std::mem::transmute::<
                    &Box<Poseidon2SkinnyInstr<F>>,
                    &Box<Poseidon2SkinnyInstr<BabyBear>>,
                >(instr)
            }),
            _ => None,
        });

        let num_instructions =
            program.inner.iter().filter(|instr| matches!(instr, Poseidon2(_))).count();
        let mut rows = vec![
            [BabyBear::zero(); PREPROCESSED_POSEIDON2_WIDTH];
            num_instructions * (NUM_EXTERNAL_ROUNDS + 3)
        ];
        instructions.zip_eq(&rows.iter_mut().chunks(NUM_EXTERNAL_ROUNDS + 3)).for_each(
            |(instruction, row_add)| {
                row_add.into_iter().enumerate().for_each(|(i, row)| {
                    let cols: &mut Poseidon2PreprocessedCols<_> =
                        (*row).as_mut_slice().borrow_mut();
                    unsafe {
                        #[cfg(feature = "sys")]
                        {
                            #[cfg(feature = "sys")]
                            {
                            crate::sys::poseidon2_skinny_instr_to_row_koalabear(
                            instruction,
                            i,
                            cols as *mut Poseidon2PreprocessedCols<_> as *mut u8,
                        );
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
                });
            },
        );

        let total_height = self.preprocessed_num_rows(program, rows.len()).unwrap();

        let main = RowMajorMatrix::new(
            unsafe {
                std::mem::transmute::<Vec<BabyBear>, Vec<F>>(
                    rows.into_iter().flatten().collect::<Vec<BabyBear>>(),
                )
            },
            PREPROCESSED_POSEIDON2_WIDTH,
        );
        Some(CompressedMatrix::new(
            main,
            PaddingRow::Zero { width: PREPROCESSED_POSEIDON2_WIDTH },
            total_height,
        ))
    }
}
