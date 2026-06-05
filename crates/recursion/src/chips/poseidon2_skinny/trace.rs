use super::columns::preprocessed::Poseidon2PreprocessedCols;
use crate::{
    chips::{
        mem::MemoryAccessColsChips,
        poseidon2_skinny::{
            columns::NUM_POSEIDON2_COLS,
            Poseidon2SkinnyChip,
            NUM_EXTERNAL_ROUNDS, NUM_INTERNAL_ROUNDS, NUM_ROUNDS, WIDTH,
        },
    },
    instruction::Instruction::Poseidon2Skinny,
    ExecutionRecord, Poseidon2Io, Poseidon2SkinnyInstr,
};
use crate::utils::{next_power_of_two, padded_rows_threshold};
use dt_primitives::RC_16_30_U32;
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

/// Number of rows used per Poseidon2 permutation in this chip (one row per round).
pub const ROWS_PER_PERMUTE: usize = NUM_ROUNDS;

impl<F: Field, const DEGREE: usize> MachineAir<F> for Poseidon2SkinnyChip<DEGREE> {
    type Record = ExecutionRecord<F>;

    type Program = crate::RecursionProgram<F>;

    fn name(&self) -> String {
        format!("Poseidon2SkinnyDeg{DEGREE}")
    }

    fn generate_dependencies(&self, _: &Self::Record, _: &mut Self::Record) {
        // No-op: events are produced directly by the runtime.
    }

    fn local_only(&self) -> bool {
        // All transition constraints are evaluated on a single row; rounds are chained via
        // memory lookups carried in the preprocessed trace.
        true
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let events = &input.poseidon2_skinny_events;
        let nb_rows =
            next_power_of_two(events.len() * ROWS_PER_PERMUTE, input.fixed_log2_rows(self));
        Some(padded_rows_threshold(nb_rows))
    }

    #[instrument(
        name = "generate poseidon2 skinny trace",
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
                let event_bb = unsafe {
                    &*(event as *const Poseidon2Io<F> as *const Poseidon2Io<BabyBear>)
                };
                #[cfg(feature = "sys")]
                unsafe {
                    crate::sys::poseidon2_skinny_event_to_row_babybear(
                        event_bb,
                        perm_rows.as_mut_ptr() as *mut u8,
                    );
                }
                #[cfg(not(feature = "sys"))]
                let _ = (event_bb, perm_rows);
            });

        let main = RowMajorMatrix::new(values, NUM_POSEIDON2_COLS);
        CompressedMatrix::new(
            main,
            PaddingRow::Zero { width: NUM_POSEIDON2_COLS },
            total_height,
        )
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

/// Fill `ROWS_PER_PERMUTE` consecutive preprocessed-trace rows for one instruction.
fn fill_prep_rows_for_instr<F: Field>(instr: &Poseidon2SkinnyInstr<F>, dst: &mut [F]) {
    debug_assert_eq!(dst.len(), ROWS_PER_PERMUTE * PREPROCESSED_POSEIDON2_WIDTH);
    debug_assert_eq!(
        instr.scratch_addrs.len(),
        ROWS_PER_PERMUTE - 1,
        "expected NUM_ROUNDS - 1 scratch groups"
    );

    let half_ext = NUM_EXTERNAL_ROUNDS / 2;
    let one = F::one();

    for r in 0..ROWS_PER_PERMUTE {
        let row_slice = &mut dst[r * PREPROCESSED_POSEIDON2_WIDTH
            ..(r + 1) * PREPROCESSED_POSEIDON2_WIDTH];
        let cols: &mut Poseidon2PreprocessedCols<F> = row_slice.borrow_mut();

        // Selectors.
        let is_external = r < half_ext || r >= half_ext + NUM_INTERNAL_ROUNDS;
        cols.round_kind = if is_external { F::zero() } else { F::one() };
        cols.is_first_round = if r == 0 { one } else { F::zero() };
        cols.is_real = one;
        cols.state_in_neg_mult = -one;

        // Round constants from RC_16_30_U32[30][16].
        if r < half_ext {
            // First half external rounds: round index = r.
            for i in 0..WIDTH {
                cols.round_constants[i] = F::from_wrapped_u32(RC_16_30_U32[r][i]);
            }
        } else if r < half_ext + NUM_INTERNAL_ROUNDS {
            // Internal rounds: round index = half_ext + (r - half_ext) = r.
            // Only state[0] gets a round constant.
            let round = r;
            cols.round_constants[0] = F::from_wrapped_u32(RC_16_30_U32[round][0]);
            for i in 1..WIDTH {
                cols.round_constants[i] = F::zero();
            }
        } else {
            // Second half external rounds: round index = r - half_ext - NUM_INTERNAL_ROUNDS
            // mapped to RC table at half_ext + NUM_INTERNAL_ROUNDS + offset = r.
            let round = r;
            for i in 0..WIDTH {
                cols.round_constants[i] = F::from_wrapped_u32(RC_16_30_U32[round][i]);
            }
        }

        // state_in_addrs: row 0 -> instr.input addr ; otherwise -> previous round's scratch addr.
        for i in 0..WIDTH {
            cols.state_in_addrs[i] = if r == 0 {
                instr.addrs.input[i]
            } else {
                instr.scratch_addrs[r - 1][i]
            };
        }

        // state_out_mem: last row -> instr.output addr with instr.mults[i] ;
        // otherwise -> next scratch addr with mult = +1.
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

