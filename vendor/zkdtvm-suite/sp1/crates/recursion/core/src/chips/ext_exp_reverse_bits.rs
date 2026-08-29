#![allow(clippy::needless_range_loop)]

use super::mem::{MemoryAccessCols, MemoryAccessColsChips};
use crate::{
    air::Block, builder::DTRecursionAirBuilder, runtime::ExecutionRecord, Address,
    ExtExpReverseBitsInstr, Instruction,
};
use core::borrow::Borrow;
use dt_core_machine::utils::{next_power_of_two, padded_rows_threshold};
use dt_derive::AlignedBorrow;
use dt_stark::{
    air::{BaseAirBuilder, DTAirBuilder, ExtensionAirBuilder, MachineAir},
    sumcheck::trace::{CompressedMatrix, PaddingRow},
};
use p3_air::{Air, AirBuilder, BaseAir, PairBuilder};
use p3_field::{AbstractField, Field};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use std::borrow::BorrowMut;
use tracing::instrument;

pub const NUM_EXT_EXP_REVERSE_BITS_COLS: usize = size_of::<ExtExpReverseBitsCols<u8>>();
pub const NUM_EXT_EXP_REVERSE_BITS_PREPROCESSED_COLS: usize =
    size_of::<ExtExpReverseBitsPreprocessedCols<u8>>();

#[derive(Clone, Debug, Copy, Default)]

pub struct ExtExpReverseBitsChip<const DEGREE: usize>;

#[derive(AlignedBorrow, Clone, Copy, Debug)]
#[repr(C)]
pub struct ExtExpReverseBitsPreprocessedCols<T: Copy> {
    pub x_addr: Address<T>,
    pub exponent_addr: Address<T>,
    pub prev_acc_addr: Address<T>,
    pub acc_mem: MemoryAccessColsChips<T>,
    pub is_real: T,
}

#[derive(AlignedBorrow, Debug, Clone, Copy)]
#[repr(C)]
pub struct ExtExpReverseBitsCols<T: Copy> {
    /// The base of the exponentiation.
    pub x: Block<T>,

    /// The current bit of the exponent. This is read from memory.
    pub current_bit: T,

    /// The previous accumulator value.
    pub prev_acc: Block<T>,

    /// The current accumulator value.
    pub acc: Block<T>,

    /// A column which equals x if `current_bit` is on, and 1 otherwise.
    pub multiplier: Block<T>,
}

impl<F, const DEGREE: usize> BaseAir<F> for ExtExpReverseBitsChip<DEGREE> {
    fn width(&self) -> usize {
        NUM_EXT_EXP_REVERSE_BITS_COLS
    }
}

impl<F: Field, const DEGREE: usize> MachineAir<F> for ExtExpReverseBitsChip<DEGREE> {
    type Record = ExecutionRecord<F>;

    type Program = crate::RecursionProgram<F>;

    fn name(&self) -> String {
        format!("ExtExpReverseBitsDeg{DEGREE}")
    }

    #[instrument(name = "generate ext exp reverse bits trace", level = "debug", skip_all, fields(rows = input.ext_exp_reverse_bits_events.len()))]
    fn generate_trace(
        &self,
        input: &ExecutionRecord<F>,
        _: &mut ExecutionRecord<F>,
    ) -> CompressedMatrix<F> {
        let mut overall_rows = Vec::new();

        input.ext_exp_reverse_bits_events.iter().for_each(|event| {
            let mut rows = vec![vec![F::zero(); NUM_EXT_EXP_REVERSE_BITS_COLS]; event.exp.len()];

            rows.iter_mut().enumerate().for_each(|(i, row)| {
                let cols: &mut ExtExpReverseBitsCols<F> = row.as_mut_slice().borrow_mut();

                cols.x = event.base;
                cols.current_bit = event.exp[i];
                cols.prev_acc = event.prev_acc_vec[i];
                cols.acc = event.acc_vec[i];
                let one_block = Block::<F>::from(F::one());
                if cols.current_bit == F::one() {
                    cols.multiplier = cols.x;
                } else {
                    cols.multiplier = one_block;
                }
            });
            overall_rows.extend(rows);
        });

        let real_nb_rows = overall_rows.len();
        let total_height =
            padded_rows_threshold(next_power_of_two(real_nb_rows, input.fixed_log2_rows(self)));

        let main = RowMajorMatrix::new(
            overall_rows.into_iter().flatten().collect::<Vec<F>>(),
            NUM_EXT_EXP_REVERSE_BITS_COLS,
        );

        CompressedMatrix::new(
            main,
            PaddingRow::Zero { width: NUM_EXT_EXP_REVERSE_BITS_COLS },
            total_height,
        )
    }

    fn generate_dependencies(&self, _: &Self::Record, _: &mut Self::Record) {}

    fn included(&self, _record: &Self::Record) -> bool {
        true
    }

    fn preprocessed_width(&self) -> usize {
        NUM_EXT_EXP_REVERSE_BITS_PREPROCESSED_COLS
    }

    fn generate_preprocessed_trace(&self, program: &Self::Program) -> Option<CompressedMatrix<F>> {
        let mut rows: Vec<[F; NUM_EXT_EXP_REVERSE_BITS_PREPROCESSED_COLS]> = Vec::new();
        program
            .inner
            .iter()
            .filter_map(|instruction| match instruction {
                Instruction::ExtExpReverseBits(x) => Some(x),
                _ => None,
            })
            .for_each(|instruction: &ExtExpReverseBitsInstr<F>| {
                let ExtExpReverseBitsInstr { addrs, mult } = instruction;
                let num_bits = addrs.exp.len();
                let mut row_add =
                    vec![[F::zero(); NUM_EXT_EXP_REVERSE_BITS_PREPROCESSED_COLS]; num_bits];
                row_add.iter_mut().enumerate().for_each(|(i, row)| {
                    let row: &mut ExtExpReverseBitsPreprocessedCols<F> =
                        row.as_mut_slice().borrow_mut();
                    row.is_real = F::one();
                    row.x_addr = addrs.base;
                    row.exponent_addr = addrs.exp[i];
                    row.prev_acc_addr = addrs.prev_acc_vec[i];
                    row.acc_mem = MemoryAccessCols {
                        addr: addrs.acc_vec[i],
                        mult: if i == num_bits - 1 { *mult } else { F::one() },
                    };
                });
                rows.extend(row_add);
            });

        let real_nb_rows = rows.len();
        let total_height =
            padded_rows_threshold(next_power_of_two(real_nb_rows, program.fixed_log2_rows(self)));

        let main = RowMajorMatrix::new(
            rows.into_iter().flatten().collect::<Vec<F>>(),
            NUM_EXT_EXP_REVERSE_BITS_PREPROCESSED_COLS,
        );
        Some(CompressedMatrix::new(
            main,
            PaddingRow::Zero { width: NUM_EXT_EXP_REVERSE_BITS_PREPROCESSED_COLS },
            total_height,
        ))
    }
}

impl<const DEGREE: usize> ExtExpReverseBitsChip<DEGREE> {
    pub fn eval_ext_exp_reverse_bits<
        AB: BaseAirBuilder + ExtensionAirBuilder + DTRecursionAirBuilder + DTAirBuilder,
    >(
        &self,
        builder: &mut AB,
        local: &ExtExpReverseBitsCols<AB::Var>,
        local_prepr: &ExtExpReverseBitsPreprocessedCols<AB::Var>,
    ) {
        // Dummy constraints to normalize to DEGREE when DEGREE > 3.
        if DEGREE > 3 {
            let lhs = (0..DEGREE).map(|_| local_prepr.is_real.into()).product::<AB::Expr>();
            let rhs = (0..DEGREE).map(|_| local_prepr.is_real.into()).product::<AB::Expr>();
            builder.assert_eq(lhs, rhs);
        }

        let local_acc = local.acc.as_extension::<AB>();
        let local_prev_acc = local.prev_acc.as_extension::<AB>();
        let local_multiplier = local.multiplier.as_extension::<AB>();

        // Receive x from memory.
        builder.receive_block(local_prepr.x_addr, local.x, local_prepr.is_real);

        // Receive exponent bit from memory.
        builder.receive_single(local_prepr.exponent_addr, local.current_bit, local_prepr.is_real);

        // Receive prev_acc from memory.
        builder.receive_block(local_prepr.prev_acc_addr, local.prev_acc, local_prepr.is_real);

        // multiplier = 1 + bit*(x − 1).
        let bit: AB::Expr = local.current_bit.into();
        for k in 0..4 {
            let mul_k: AB::Expr = local.multiplier.0[k].into();
            let x_k: AB::Expr = local.x.0[k].into();
            let one_k = if k == 0 { AB::Expr::one() } else { AB::Expr::zero() };
            builder
                .when(local_prepr.is_real)
                .assert_eq(mul_k - one_k.clone(), bit.clone() * (x_k - one_k));
        }

        // acc = prev_acc^2 * multiplier.
        builder.assert_ext_eq(
            local_acc.clone(),
            local_prev_acc.clone() * local_prev_acc.clone() * local_multiplier.clone(),
        );

        // Send acc to memory.
        builder.send_block(local_prepr.acc_mem.addr, local.acc, local_prepr.acc_mem.mult);
    }
}

impl<AB, const DEGREE: usize> Air<AB> for ExtExpReverseBitsChip<DEGREE>
where
    AB: DTRecursionAirBuilder + PairBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &ExtExpReverseBitsCols<AB::Var> = (*local).borrow();
        let prep = builder.preprocessed();
        let prep_local = prep.row_slice(0);
        let prep_local: &ExtExpReverseBitsPreprocessedCols<_> = (*prep_local).borrow();
        self.eval_ext_exp_reverse_bits::<AB>(builder, local, prep_local);
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::print_stdout)]

    use super::*;
    use crate::{
        machine::tests::test_recursion_linear_program, runtime::instruction as instr,
        stark::BabyBearPoseidon2Outer, Instruction, MemAccessKind,
    };
    use dt_core_machine::utils::setup_logger;
    use dt_stark::StarkGenericConfig;
    use itertools::Itertools;
    use p3_field::{extension::BinomialExtensionField, AbstractExtensionField, AbstractField};
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use std::iter::once;
    use tracing::debug;

    const DEGREE: usize = 3;

    #[test]
    fn prove_babybear_circuit_erbl() {
        setup_logger();
        type SC = BabyBearPoseidon2Outer;
        type F = <SC as StarkGenericConfig>::Val;

        let mut rng = StdRng::seed_from_u64(0xDEADBEEF);
        let mut random_ext = move || {
            let inner: [F; 4] = core::array::from_fn(|_| rng.sample(rand::distributions::Standard));
            BinomialExtensionField::<F, 4>::from_base_slice(&inner)
        };
        let mut rng = StdRng::seed_from_u64(0xDEADBEEF);
        let mut random_bit = move || rng.gen_range(0..2);
        let mut addr = 0;

        let instructions = (1..15)
            .flat_map(|i| {
                let base = random_ext();
                let exponent_bits = vec![random_bit(); i];
                let exponent = F::from_canonical_u32(
                    exponent_bits
                        .clone()
                        .iter()
                        .rev()
                        .enumerate()
                        .fold(0, |acc, (i, x)| acc + x * (1 << i)),
                );
                let mut out = BinomialExtensionField::from_base(F::one());
                exponent_bits.clone().into_iter().for_each(|val| {
                    out = out * out;
                    if val == 1 {
                        out = out * base;
                    }
                });
                if i < 5 {
                    debug!("base: {:?}, exponent: {:?}, out: {:?}", base, exponent, out);
                }

                let alloc_size = i + 2;
                let exp_a = (0..i).map(|x| x + addr + 1).collect::<Vec<_>>();
                let exp_a_clone = exp_a.clone();
                let x_a = addr;
                addr += alloc_size;

                // Allocate prev_acc_vec and acc_vec
                // prev_acc_vec[0] = constant 1 (will be written by compiler)
                // prev_acc_vec[1..] = acc_vec[0..n-1]
                // acc_vec[n-1] = result
                let temp_acc_addrs: Vec<_> = (0..i.saturating_sub(1)).map(|j| addr + j).collect();
                addr += temp_acc_addrs.len();
                let result_addr = addr;
                addr += 1;

                let prev_acc_a: Vec<_> = std::iter::once(addr) // constant 1 address
                    .chain(temp_acc_addrs.iter().copied())
                    .collect();
                let acc_a: Vec<_> =
                    temp_acc_addrs.iter().copied().chain(std::iter::once(result_addr)).collect();
                addr += 1; // for constant 1

                let exp_bit_instructions = (0..i).map(move |j| {
                    instr::mem_single(
                        MemAccessKind::Write,
                        1,
                        exp_a_clone[j] as u32,
                        F::from_canonical_u32(exponent_bits.clone()[j]),
                    )
                });

                // Write initial prev_acc = 1 for first row
                let init_prev_acc = instr::mem_ext(
                    MemAccessKind::Write,
                    1,
                    prev_acc_a[0] as u32,
                    BinomialExtensionField::<F, 4>::from_base(F::one()),
                );

                once(instr::mem_ext(MemAccessKind::Write, 1, x_a as u32, base))
                    .chain(exp_bit_instructions)
                    .chain(once(init_prev_acc))
                    .chain(once(instr::ext_exp_reverse_bits(
                        1,
                        F::from_canonical_u32(x_a as u32),
                        exp_a
                            .into_iter()
                            .map(|bit| F::from_canonical_u32(bit as u32))
                            .collect_vec(),
                        prev_acc_a.iter().map(|a| F::from_canonical_u32(*a as u32)).collect(),
                        acc_a.iter().map(|a| F::from_canonical_u32(*a as u32)).collect(),
                    )))
                    .chain(once(instr::mem_ext(MemAccessKind::Read, 1, result_addr as u32, out)))
            })
            .collect::<Vec<Instruction<F>>>();

        test_recursion_linear_program(instructions);
    }
}
