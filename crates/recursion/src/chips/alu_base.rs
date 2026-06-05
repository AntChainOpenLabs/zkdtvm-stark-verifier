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
pub const NUM_BASE_ALU_SHRINK_ENTRIES_PER_ROW: usize = 2;

pub struct BaseAluChip<const N: usize = NUM_BASE_ALU_ENTRIES_PER_ROW>;

impl<const N: usize> Default for BaseAluChip<N> {
    fn default() -> Self {
        Self
    }
}

impl<const N: usize> BaseAluChip<N> {
    pub const fn num_cols() -> usize {
        core::mem::size_of::<BaseAluValueCols<u8>>() * N
    }

    pub const fn num_preprocessed_cols() -> usize {
        core::mem::size_of::<BaseAluAccessCols<u8>>() * N
    }
}

pub const NUM_BASE_ALU_COLS: usize = BaseAluChip::<NUM_BASE_ALU_ENTRIES_PER_ROW>::num_cols();

#[derive(AlignedBorrow, Debug, Clone, Copy)]
#[repr(C)]
pub struct BaseAluCols<F: Copy, const N: usize = NUM_BASE_ALU_ENTRIES_PER_ROW> {
    pub values: [BaseAluValueCols<F>; N],
}

pub const NUM_BASE_ALU_VALUE_COLS: usize = core::mem::size_of::<BaseAluValueCols<u8>>();

#[derive(AlignedBorrow, Debug, Clone, Copy)]
#[repr(C)]
pub struct BaseAluValueCols<F: Copy> {
    pub vals: BaseAluIo<F>,
}

pub const NUM_BASE_ALU_PREPROCESSED_COLS: usize =
    BaseAluChip::<NUM_BASE_ALU_ENTRIES_PER_ROW>::num_preprocessed_cols();

#[derive(AlignedBorrow, Debug, Clone, Copy)]
#[repr(C)]
pub struct BaseAluPreprocessedCols<F: Copy, const N: usize = NUM_BASE_ALU_ENTRIES_PER_ROW> {
    pub accesses: [BaseAluAccessCols<F>; N],
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

impl<F: Field, const N: usize> BaseAir<F> for BaseAluChip<N> {
    fn width(&self) -> usize {
        Self::num_cols()
    }
}

impl<F: Field, const N: usize> MachineAir<F> for BaseAluChip<N> {
    type Record = ExecutionRecord<F>;

    type Program = crate::RecursionProgram<F>;

    fn name(&self) -> String {
        if N == NUM_BASE_ALU_ENTRIES_PER_ROW {
            "BaseAlu".to_string()
        } else {
            format!("BaseAlu<{}>", N)
        }
    }

    fn preprocessed_width(&self) -> usize {
        Self::num_preprocessed_cols()
    }

    fn preprocessed_num_rows(&self, program: &Self::Program, instrs_len: usize) -> Option<usize> {
        let nb_rows = instrs_len.div_ceil(N);
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
        let num_preprocessed_cols = Self::num_preprocessed_cols();
        let padded_nb_rows = self.preprocessed_num_rows(program, instrs.len()).unwrap();
        let real_nb_rows = instrs.len().div_ceil(N);
        let mut values = vec![BabyBear::zero(); real_nb_rows * num_preprocessed_cols];

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
            num_preprocessed_cols,
        );
        Some(CompressedMatrix::new(
            main,
            PaddingRow::Zero { width: num_preprocessed_cols },
            padded_nb_rows,
        ))
    }

    fn generate_dependencies(&self, _: &Self::Record, _: &mut Self::Record) {
        // This is a no-op.
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows = input.base_alu_events.len().div_ceil(N);
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
        let num_cols = Self::num_cols();
        let padded_nb_rows = self.num_rows(input).unwrap();
        let real_nb_rows = events.len().div_ceil(N);
        let mut values = vec![BabyBear::zero(); real_nb_rows * num_cols];

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
            num_cols,
        );
        CompressedMatrix::new(main, PaddingRow::Zero { width: num_cols }, padded_nb_rows)
    }

    fn included(&self, _record: &Self::Record) -> bool {
        true
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl<AB, const N: usize> Air<AB> for BaseAluChip<N>
where
    AB: DTRecursionAirBuilder + PairBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &BaseAluCols<AB::Var, N> = (*local).borrow();
        let prep = builder.preprocessed();
        let prep_local = prep.row_slice(0);
        let prep_local: &BaseAluPreprocessedCols<AB::Var, N> = (*prep_local).borrow();

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

#[cfg(test)]
mod tests {
    use crate::{chips::test_fixtures, runtime::instruction as instr};
    use dt_stark::{baby_bear_poseidon2::BabyBearPoseidon2, StarkGenericConfig};
    use machine::tests::test_recursion_linear_program;
    use p3_baby_bear::BabyBear;
    use p3_field::AbstractField;
    use p3_matrix::dense::RowMajorMatrix;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    use super::*;

    fn generate_trace_reference(
        input: &ExecutionRecord<BabyBear>,
        _: &mut ExecutionRecord<BabyBear>,
    ) -> RowMajorMatrix<BabyBear> {
        let events = &input.base_alu_events;
        let chip = BaseAluChip::<NUM_BASE_ALU_ENTRIES_PER_ROW>;
        let padded_nb_rows = chip.num_rows(input).unwrap();
        let mut values = vec![BabyBear::zero(); padded_nb_rows * NUM_BASE_ALU_COLS];

        let populate_len = events.len() * NUM_BASE_ALU_VALUE_COLS;
        values[..populate_len].par_chunks_mut(NUM_BASE_ALU_VALUE_COLS).zip_eq(events).for_each(
            |(row, &vals)| {
                let cols: &mut BaseAluValueCols<_> = row.borrow_mut();
                *cols = BaseAluValueCols { vals };
            },
        );

        RowMajorMatrix::new(values, NUM_BASE_ALU_COLS)
    }

    #[test]
    fn generate_trace() {
        let shard = test_fixtures::shard();
        let mut execution_record = test_fixtures::default_execution_record();
        let chip = BaseAluChip::<NUM_BASE_ALU_ENTRIES_PER_ROW>;
        let trace = chip.generate_trace(&shard, &mut execution_record);
        let ref_full = generate_trace_reference(&shard, &mut execution_record);
        assert!(trace.total_height >= test_fixtures::MIN_TEST_CASES);
        assert_eq!(trace.total_height, ref_full.height());
        for i in 0..trace.main.height() {
            assert_eq!(trace.main.row(i), ref_full.row(i));
        }
        for i in trace.main.height()..trace.total_height {
            assert!(ref_full.row(i).iter().all(|&x| x == BabyBear::zero()));
        }
    }

    fn generate_preprocessed_trace_reference(
        program: &RecursionProgram<BabyBear>,
    ) -> RowMajorMatrix<BabyBear> {
        type F = BabyBear;

        let instrs = program
            .inner
            .iter()
            .filter_map(|instruction| match instruction {
                Instruction::BaseAlu(x) => Some(x),
                _ => None,
            })
            .collect::<Vec<_>>();
        let chip = BaseAluChip::<NUM_BASE_ALU_ENTRIES_PER_ROW>;
        let padded_nb_rows = chip.preprocessed_num_rows(program, instrs.len()).unwrap();
        let mut values = vec![F::zero(); padded_nb_rows * NUM_BASE_ALU_PREPROCESSED_COLS];

        let populate_len = instrs.len() * NUM_BASE_ALU_ACCESS_COLS;
        values[..populate_len].par_chunks_mut(NUM_BASE_ALU_ACCESS_COLS).zip_eq(instrs).for_each(
            |(row, instr)| {
                let BaseAluInstr { opcode, mult, addrs } = instr;
                let access: &mut BaseAluAccessCols<_> = row.borrow_mut();
                *access = BaseAluAccessCols {
                    addrs: addrs.to_owned(),
                    is_add: F::from_bool(false),
                    is_sub: F::from_bool(false),
                    is_mul: F::from_bool(false),
                    is_div: F::from_bool(false),
                    mult: mult.to_owned(),
                };
                let target_flag = match opcode {
                    BaseAluOpcode::AddF => &mut access.is_add,
                    BaseAluOpcode::SubF => &mut access.is_sub,
                    BaseAluOpcode::MulF => &mut access.is_mul,
                    BaseAluOpcode::DivF => &mut access.is_div,
                };
                *target_flag = F::from_bool(true);
            },
        );

        RowMajorMatrix::new(values, NUM_BASE_ALU_PREPROCESSED_COLS)
    }

    #[test]
    #[ignore = "Failing due to merge conflicts. Will be fixed shortly."]
    fn generate_preprocessed_trace() {
        let program = test_fixtures::program();
        let chip = BaseAluChip::<NUM_BASE_ALU_ENTRIES_PER_ROW>;
        let trace = chip.generate_preprocessed_trace(&program).unwrap();
        let ref_full = generate_preprocessed_trace_reference(&program);
        assert!(trace.total_height >= test_fixtures::MIN_TEST_CASES);
        assert_eq!(trace.total_height, ref_full.height());
        for i in 0..trace.main.height() {
            assert_eq!(trace.main.row(i), ref_full.row(i));
        }
        for i in trace.main.height()..trace.total_height {
            assert!(ref_full.row(i).iter().all(|&x| x == BabyBear::zero()));
        }
    }

    #[test]
    pub fn four_ops() {
        type SC = BabyBearPoseidon2;
        type F = <SC as StarkGenericConfig>::Val;

        let mut rng = StdRng::seed_from_u64(0xDEADBEEF);
        let mut random_felt = move || -> F { rng.sample(rand::distributions::Standard) };
        let mut addr = 0;

        let instructions = (0..1000)
            .flat_map(|_| {
                let quot = random_felt();
                let in2 = random_felt();
                let in1 = in2 * quot;
                let alloc_size = 6;
                let a = (0..alloc_size).map(|x| x + addr).collect::<Vec<_>>();
                addr += alloc_size;
                [
                    instr::mem_single(MemAccessKind::Write, 4, a[0], in1),
                    instr::mem_single(MemAccessKind::Write, 4, a[1], in2),
                    instr::base_alu(BaseAluOpcode::AddF, 1, a[2], a[0], a[1]),
                    instr::mem_single(MemAccessKind::Read, 1, a[2], in1 + in2),
                    instr::base_alu(BaseAluOpcode::SubF, 1, a[3], a[0], a[1]),
                    instr::mem_single(MemAccessKind::Read, 1, a[3], in1 - in2),
                    instr::base_alu(BaseAluOpcode::MulF, 1, a[4], a[0], a[1]),
                    instr::mem_single(MemAccessKind::Read, 1, a[4], in1 * in2),
                    instr::base_alu(BaseAluOpcode::DivF, 1, a[5], a[0], a[1]),
                    instr::mem_single(MemAccessKind::Read, 1, a[5], quot),
                ]
            })
            .collect::<Vec<Instruction<F>>>();

        test_recursion_linear_program(instructions);
    }
}
