use super::columns::preprocessed::Poseidon2PreprocessedCols;
use crate::{
    chips::{
        mem::MemoryAccessColsChips,
        poseidon2_skinny_kb::{columns::NUM_POSEIDON2_COLS, Poseidon2SkinnyKbChip, WIDTH},
    },
    instruction::Instruction::Poseidon2Skinny,
    ExecutionRecord, Poseidon2Io, Poseidon2SkinnyInstr,
};
use crate::utils::{next_power_of_two, padded_rows_threshold};
use dt_stark::air::MachineAir;
use dt_stark::sumcheck::trace::{CompressedMatrix, PaddingRow};
use p3_baby_bear::BabyBear;
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use std::borrow::BorrowMut;
use std::mem::size_of;
use tracing::instrument;

const PREPROCESSED_POSEIDON2_WIDTH: usize = size_of::<Poseidon2PreprocessedCols<u8>>();

/// Number of rows used per Poseidon2 permutation in this chip.
pub const ROWS_PER_PERMUTE: usize = super::ROWS_PER_PERMUTE;

impl<F: Field, const DEGREE: usize> MachineAir<F> for Poseidon2SkinnyKbChip<DEGREE> {
    type Record = ExecutionRecord<F>;

    type Program = crate::RecursionProgram<F>;

    fn name(&self) -> String {
        format!("Poseidon2SkinnyKbDeg{DEGREE}")
    }

    fn generate_dependencies(&self, _: &Self::Record, _: &mut Self::Record) {}

    fn local_only(&self) -> bool {
        true
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let events = &input.poseidon2_skinny_events;
        let nb_rows =
            next_power_of_two(events.len() * ROWS_PER_PERMUTE, input.fixed_log2_rows(self));
        Some(padded_rows_threshold(nb_rows))
    }

    #[instrument(
        name = "generate poseidon2 skinny kb trace",
        level = "debug",
        skip_all,
        fields(rows = input.poseidon2_skinny_events.len())
    )]
    fn generate_trace(
        &self,
        input: &ExecutionRecord<F>,
        _output: &mut ExecutionRecord<F>,
    ) -> CompressedMatrix<F> {
        assert!(
            std::mem::size_of::<F>() == std::mem::size_of::<BabyBear>(),
            "generate_trace only supports 32-bit prime fields (BabyBear/KoalaBear)"
        );

        let total_height = self.num_rows(input).unwrap();
        let events = &input.poseidon2_skinny_events;
        let real_rows = events.len() * ROWS_PER_PERMUTE;
        let mut values = vec![F::zero(); real_rows * NUM_POSEIDON2_COLS];

        values
            .par_chunks_mut(ROWS_PER_PERMUTE * NUM_POSEIDON2_COLS)
            .zip(events.par_iter())
            .for_each(|(perm_rows, event)| {
                let event_bb =
                    unsafe { &*(event as *const Poseidon2Io<F> as *const Poseidon2Io<BabyBear>) };
                #[cfg(feature = "sys")]
                unsafe {
                    crate::sys::poseidon2_skinny_event_to_row_koalabear(
                        event_bb,
                        perm_rows.as_mut_ptr() as *mut u8,
                    );
                }
                #[cfg(not(feature = "sys"))]
                let _ = (event_bb, perm_rows);
            });

        let main = RowMajorMatrix::new(values, NUM_POSEIDON2_COLS);
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

        let instrs: Vec<&Poseidon2SkinnyInstr<F>> = program
            .inner
            .iter()
            .filter_map(|instruction| match instruction {
                Poseidon2Skinny(instr) => Some(instr.as_ref()),
                _ => None,
            })
            .collect();

        let real_rows = instrs.len() * ROWS_PER_PERMUTE;
        let total_height = self.preprocessed_num_rows(program, real_rows).unwrap();
        let mut values = vec![F::zero(); real_rows * PREPROCESSED_POSEIDON2_WIDTH];

        values
            .par_chunks_mut(ROWS_PER_PERMUTE * PREPROCESSED_POSEIDON2_WIDTH)
            .zip(instrs.par_iter())
            .for_each(|(perm_rows, instr)| {
                fill_prep_rows_for_instr::<F>(instr, perm_rows);
            });

        let main = RowMajorMatrix::new(values, PREPROCESSED_POSEIDON2_WIDTH);
        Some(CompressedMatrix::new(
            main,
            PaddingRow::Zero { width: PREPROCESSED_POSEIDON2_WIDTH },
            total_height,
        ))
    }
}

/// Fill `ROWS_PER_PERMUTE` (= 5) consecutive preprocessed-trace rows for one Poseidon2
/// permutation instruction.
fn fill_prep_rows_for_instr<F: Field>(instr: &Poseidon2SkinnyInstr<F>, dst: &mut [F]) {
    debug_assert_eq!(dst.len(), ROWS_PER_PERMUTE * PREPROCESSED_POSEIDON2_WIDTH);
    debug_assert_eq!(
        instr.scratch_addrs.len(),
        ROWS_PER_PERMUTE - 1,
        "expected ROWS_PER_PERMUTE - 1 (= 4) scratch groups for the 5-row KoalaBear layout"
    );

    let one = F::one();

    for r in 0..ROWS_PER_PERMUTE {
        let row_slice =
            &mut dst[r * PREPROCESSED_POSEIDON2_WIDTH..(r + 1) * PREPROCESSED_POSEIDON2_WIDTH];
        let cols: &mut Poseidon2PreprocessedCols<F> = row_slice.borrow_mut();

        // One-hot row selector: is_round[r] = 1 for this row.
        cols.is_round[r] = one;

        // state_in_addrs: row 0 reads from instr.input; others read from previous scratch.
        for i in 0..WIDTH {
            cols.state_in_addrs[i] =
                if r == 0 { instr.addrs.input[i] } else { instr.scratch_addrs[r - 1][i] };
        }

        // state_out_mem: last row writes to instr.output with instr.mults;
        // others write to next scratch with mult = +1.
        for i in 0..WIDTH {
            let (addr, mult) = if r == ROWS_PER_PERMUTE - 1 {
                (instr.addrs.output[i], instr.mults[i])
            } else {
                (instr.scratch_addrs[r][i], one)
            };
            cols.state_out_mem[i] = MemoryAccessColsChips { addr, mult };
        }
    }
}
