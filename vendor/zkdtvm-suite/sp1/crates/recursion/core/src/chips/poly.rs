#![allow(clippy::needless_range_loop)]

use crate::{builder::DTRecursionAirBuilder, runtime::ExecutionRecord, Instruction, PolyEvalInstr};
use core::borrow::Borrow;
use dt_core_machine::utils::{next_power_of_two, padded_rows_threshold};
use dt_derive::AlignedBorrow;
use dt_stark::{
    air::{BaseAirBuilder, DTAirBuilder, ExtensionAirBuilder, MachineAir},
    sumcheck::trace::{CompressedMatrix, PaddingRow},
};
use p3_air::{Air, AirBuilder, BaseAir, PairBuilder};
use p3_field::Field;
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use std::borrow::BorrowMut;
use tracing::instrument;

use super::mem::{MemoryAccessCols, MemoryAccessColsChips};

pub const NUM_POLY_EVAL_COLS: usize = size_of::<PolyEvalCols<u8>>();
pub const NUM_POLY_EVAL_PREPROCESS_COLS: usize = size_of::<PolyEvalPreprocessedCols<u8>>();

#[derive(Clone, Debug, Copy, Default)]
pub struct PolyEvalChip<const DEGREE: usize>;

#[derive(AlignedBorrow, Clone, Copy, Debug)]
#[repr(C)]
pub struct PolyEvalPreprocessedCols<T: Copy> {
    pub point_mem: MemoryAccessColsChips<T>,
    pub coeff_mem: MemoryAccessColsChips<T>,
    pub out_mem: MemoryAccessColsChips<T>,
    pub iteration_num: T,
    pub is_first: T,
    pub is_last: T,
    pub is_real: T,
    pub chain_accum_out: MemoryAccessColsChips<T>,
    pub chain_accum_in: MemoryAccessColsChips<T>,
}

#[derive(AlignedBorrow, Debug, Clone, Copy)]
#[repr(C)]
pub struct PolyEvalCols<T: Copy> {
    /// The eval point of the polynomial.
    pub point: T,

    /// The current coefficient of the polynomial. This is read from memory.
    pub current_coeff: T,

    /// accum * point.
    pub accum_mul_point: T,

    /// The accumulator of the current iteration.
    pub accum: T,

    /// The previous row's accum_mul_point, for local-only chaining.
    pub prev_accum_mul_point: T,
}

impl<F, const DEGREE: usize> BaseAir<F> for PolyEvalChip<DEGREE> {
    fn width(&self) -> usize {
        NUM_POLY_EVAL_COLS
    }
}

impl<F: Field, const DEGREE: usize> MachineAir<F> for PolyEvalChip<DEGREE> {
    type Record = ExecutionRecord<F>;

    type Program = crate::RecursionProgram<F>;

    fn name(&self) -> String {
        "PolyEval".to_string()
    }

    #[instrument(name = "generate poly eval trace", level = "debug", skip_all, fields(rows = input.poly_eval_events.len()))]
    fn generate_trace(
        &self,
        input: &ExecutionRecord<F>,
        _: &mut ExecutionRecord<F>,
    ) -> CompressedMatrix<F> {
        let mut overall_rows = Vec::new();

        input.poly_eval_events.iter().for_each(|event| {
            let mut rows = vec![vec![F::zero(); NUM_POLY_EVAL_COLS]; event.coeff.len()];
            let mut prev_accum_mul_point = F::zero();

            rows.iter_mut().enumerate().for_each(|(i, row)| {
                let cols: &mut PolyEvalCols<F> = row.as_mut_slice().borrow_mut();

                cols.point = event.point;
                cols.current_coeff = event.coeff[i];
                cols.prev_accum_mul_point = prev_accum_mul_point;

                let accum = if i == 0 {
                    cols.current_coeff
                } else {
                    prev_accum_mul_point + cols.current_coeff
                };

                cols.accum = accum;
                cols.accum_mul_point = accum * cols.point;
                prev_accum_mul_point = cols.accum_mul_point;
            });
            overall_rows.extend(rows);
        });

        let real_nb_rows = overall_rows.len();
        let mut total_height = next_power_of_two(real_nb_rows, input.fixed_log2_rows(self));
        total_height = padded_rows_threshold(total_height);

        let main = RowMajorMatrix::new(
            overall_rows.into_iter().flatten().collect::<Vec<F>>(),
            NUM_POLY_EVAL_COLS,
        );

        CompressedMatrix::new(main, PaddingRow::Zero { width: NUM_POLY_EVAL_COLS }, total_height)
    }

    fn generate_dependencies(&self, _: &Self::Record, _: &mut Self::Record) {}

    fn included(&self, _record: &Self::Record) -> bool {
        true
    }

    fn preprocessed_width(&self) -> usize {
        NUM_POLY_EVAL_PREPROCESS_COLS
    }

    fn generate_preprocessed_trace(&self, program: &Self::Program) -> Option<CompressedMatrix<F>> {
        let mut rows: Vec<[F; NUM_POLY_EVAL_PREPROCESS_COLS]> = Vec::new();
        program
            .inner
            .iter()
            .filter_map(|instruction| match instruction {
                Instruction::PolyEval(x) => Some(x),
                _ => None,
            })
            .for_each(|instruction: &PolyEvalInstr<F>| {
                let PolyEvalInstr { addrs, mult, chain_accum_addrs } = instruction;
                let num_coeffs = addrs.coeff.len();
                let mut row_add = vec![[F::zero(); NUM_POLY_EVAL_PREPROCESS_COLS]; num_coeffs];
                row_add.iter_mut().enumerate().for_each(|(i, row)| {
                    let row: &mut PolyEvalPreprocessedCols<F> = row.as_mut_slice().borrow_mut();
                    row.iteration_num = F::from_canonical_u32(i as u32);
                    row.is_first = F::from_bool(i == 0);
                    row.is_last = F::from_bool(i == num_coeffs - 1);
                    row.is_real = F::one();
                    row.point_mem =
                        MemoryAccessCols { addr: addrs.point, mult: -F::from_bool(i == 0) };
                    row.coeff_mem = MemoryAccessCols { addr: addrs.coeff[i], mult: F::neg_one() };
                    row.out_mem = MemoryAccessCols {
                        addr: addrs.out,
                        mult: *mult * F::from_bool(i == num_coeffs - 1),
                    };
                    if i < num_coeffs - 1 {
                        row.chain_accum_out =
                            MemoryAccessCols { addr: chain_accum_addrs[i], mult: F::one() };
                    }
                    if i > 0 {
                        row.chain_accum_in =
                            MemoryAccessCols { addr: chain_accum_addrs[i - 1], mult: F::neg_one() };
                    }
                });
                rows.extend(row_add);
            });

        let real_nb_rows = rows.len();
        let mut total_height = next_power_of_two(real_nb_rows, program.fixed_log2_rows(self));
        total_height = padded_rows_threshold(total_height);

        let main = RowMajorMatrix::new(
            rows.into_iter().flatten().collect::<Vec<F>>(),
            NUM_POLY_EVAL_PREPROCESS_COLS,
        );
        Some(CompressedMatrix::new(
            main,
            PaddingRow::Zero { width: NUM_POLY_EVAL_PREPROCESS_COLS },
            total_height,
        ))
    }
}

impl<const DEGREE: usize> PolyEvalChip<DEGREE> {
    pub fn poly_eval<
        AB: BaseAirBuilder + ExtensionAirBuilder + DTRecursionAirBuilder + DTAirBuilder,
    >(
        &self,
        builder: &mut AB,
        local: &PolyEvalCols<AB::Var>,
        local_prepr: &PolyEvalPreprocessedCols<AB::Var>,
    ) {
        // Dummy constraints to normalize to DEGREE when DEGREE > 3.
        if DEGREE > 3 {
            let lhs = (0..DEGREE).map(|_| local_prepr.is_real.into()).product::<AB::Expr>();
            let rhs = (0..DEGREE).map(|_| local_prepr.is_real.into()).product::<AB::Expr>();
            builder.assert_eq(lhs, rhs);
        }

        // Read point from memory (only on first row).
        builder.send_single(local_prepr.point_mem.addr, local.point, local_prepr.point_mem.mult);

        // Read coefficient from memory.
        builder.send_single(
            local_prepr.coeff_mem.addr,
            local.current_coeff,
            local_prepr.coeff_mem.mult,
        );

        // On first row: accum = coeff.
        builder.when(local_prepr.is_first).assert_eq(local.accum, local.current_coeff);

        // On non-first rows: accum = prev_accum_mul_point + coeff.
        builder
            .when(local_prepr.is_real)
            .when_not(local_prepr.is_first)
            .assert_eq(local.accum, local.prev_accum_mul_point + local.current_coeff);

        // accum_mul_point = accum * point (on non-last rows).
        builder
            .when(local_prepr.is_real)
            .when_not(local_prepr.is_last)
            .assert_eq(local.accum_mul_point, local.accum * local.point);

        // Write result.
        builder.send_single(local_prepr.out_mem.addr, local.accum, local_prepr.out_mem.mult);

        // Chain interactions for accum_mul_point.
        builder.send_single(
            local_prepr.chain_accum_out.addr,
            local.accum_mul_point,
            local_prepr.chain_accum_out.mult,
        );
        builder.send_single(
            local_prepr.chain_accum_in.addr,
            local.prev_accum_mul_point,
            local_prepr.chain_accum_in.mult,
        );
    }

    pub const fn do_poly_eval_memory_access<T: Copy>(local: &PolyEvalPreprocessedCols<T>) -> T {
        local.is_real
    }
}

impl<AB, const DEGREE: usize> Air<AB> for PolyEvalChip<DEGREE>
where
    AB: DTRecursionAirBuilder + PairBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &PolyEvalCols<AB::Var> = (*local).borrow();
        let prep = builder.preprocessed();
        let prep_local = prep.row_slice(0);
        let prep_local: &PolyEvalPreprocessedCols<_> = (*prep_local).borrow();
        self.poly_eval::<AB>(builder, local, prep_local);
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::print_stdout)]

    use super::*;
    use crate::{
        chips::test_fixtures,
        linear_program,
        machine::tests::test_recursion_linear_program,
        runtime::{instruction as instr, ExecutionRecord},
        stark::BabyBearPoseidon2Outer,
        Address, Instruction,
        Instruction::PolyEval,
        MemAccessKind, PolyEvalIo, RecursionProgram,
    };
    use dt_core_machine::utils::setup_logger;
    use dt_stark::{air::MachineAir, StarkGenericConfig};
    use itertools::Itertools;
    use p3_baby_bear::BabyBear;
    use p3_field::{AbstractExtensionField, AbstractField};
    use p3_matrix::dense::RowMajorMatrix;
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use std::iter::once;

    const DEGREE: usize = 3;

    #[test]
    fn prove_babybear_circuit_erbl() {
        setup_logger();
        type SC = BabyBearPoseidon2Outer;
        type F = <SC as StarkGenericConfig>::Val;

        let mut rng = StdRng::seed_from_u64(0xDEADBEEF);
        let mut random_felt = move || -> F { F::from_canonical_u32(rng.gen_range(0..1 << 4)) };
        let mut rng = StdRng::seed_from_u64(0xDEADBEEF);
        let mut random_coeff = move || -> F { F::from_canonical_u32(rng.gen_range(0..1 << 4)) };
        let mut addr = 0;

        let instructions = (1..15)
            .flat_map(|i| {
                let point = random_felt();
                let coeff = vec![random_coeff(); i];
                let out = coeff[1..].iter().fold(coeff[0], |acc, &x| acc * point + x);

                let alloc_size = i + 2;
                let coeff_a = (0..i).map(|x| x + addr + 1).collect::<Vec<_>>();
                let coeff_a_clone = coeff_a.clone();
                let point_a = addr;
                let out_a = addr + alloc_size - 1;
                addr += alloc_size;
                let poly_eval_instructions = (0..i).map(move |j| {
                    instr::mem_single(MemAccessKind::Write, 1, coeff_a_clone[j] as u32, coeff[j])
                });
                once(instr::mem_single(MemAccessKind::Write, 1, point_a as u32, point))
                    .chain(poly_eval_instructions)
                    .chain(once(instr::poly_eval(
                        1,
                        F::from_canonical_u32(point_a as u32),
                        coeff_a
                            .into_iter()
                            .map(|co| F::from_canonical_u32(co as u32))
                            .collect_vec(),
                        F::from_canonical_u32(out_a as u32),
                    )))
                    .chain(once(instr::mem_single(MemAccessKind::Read, 1, out_a as u32, out)))
            })
            .collect::<Vec<Instruction<F>>>();

        test_recursion_linear_program(instructions);
    }
}
