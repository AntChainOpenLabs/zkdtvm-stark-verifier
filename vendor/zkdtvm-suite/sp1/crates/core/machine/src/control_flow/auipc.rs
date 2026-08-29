use dt_core_executor::{
    events::{ByteLookupEvent, ByteRecord},
    ExecutionRecord, Opcode, Program, DEFAULT_PC_INC,
};
use dt_derive::AlignedBorrow;
use dt_stark::{
    air::MachineAir,
    sumcheck::trace::{CompressedMatrix, PaddingRow},
    Word,
};
use hashbrown::HashMap;
use itertools::Itertools;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use rayon::iter::{ParallelBridge, ParallelIterator};
use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use crate::{
    adapter::{CPUState, JTypeRegisterOp},
    air::{DTCoreAirBuilder, WordAirBuilder},
    operations::{AddOperation, BabyBearWordRangeChecker},
    utils::{next_power_of_two, padded_rows_threshold, zeroed_f_vec},
};

#[derive(Default)]
pub struct AuipcChip;

pub const NUM_AUIPC_COLS: usize = size_of::<AuipcColumns<u8>>();

impl<F> BaseAir<F> for AuipcChip {
    fn width(&self) -> usize {
        NUM_AUIPC_COLS
    }
}

/// The column layout for AUIPC/UNIMP/EBREAK instructions.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct AuipcColumns<T> {
    ///cpu state
    pub cpu_state: CPUState<T>,
    /// j type(1 register, 2 imm)
    pub mem_ops: JTypeRegisterOp<T>,
    /// add op: a_word = pc + op_b
    pub add_op: AddOperation<T>,
    ///pc word
    pub pc: Word<T>,
    /// BabyBear range checker for the program counter.
    pub pc_range_checker: BabyBearWordRangeChecker<T>,

    /// Whether the instruction is an AUIPC instruction.
    pub is_auipc: T,

    /// Whether the instruction is an unimplemented instruction.
    pub is_unimp: T,

    /// Whether the instruction is an ebreak instruction.
    pub is_ebreak: T,
    /// is real row
    pub is_real: T,
}

impl<AB> Air<AB> for AuipcChip
where
    AB: DTCoreAirBuilder,
    AB::Var: Sized,
{
    #[inline(never)]
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &AuipcColumns<AB::Var> = (*local).borrow();
        let execution_shard: AB::Expr = builder.current_shard().into();
        let shard: AB::Expr = local.cpu_state.shard.into();
        let clk: AB::Expr = local.cpu_state.clk::<AB>();
        let a_word = local.mem_ops.op_a_value();
        let b_word = local.mem_ops.op_b_value();
        let _c_word = local.mem_ops.op_c_value();

        // SAFETY: All selectors `is_auipc`, `is_unimp`, `is_ebreak` are checked to be boolean.
        // Each "real" row has exactly one selector turned on, as `is_real`, the sum of the three
        // selectors, is boolean. Therefore, the `opcode` matches the corresponding opcode.
        builder.assert_bool(local.is_auipc);
        builder.assert_bool(local.is_unimp);
        builder.assert_bool(local.is_ebreak);
        let is_real_effect = local.is_auipc + local.is_unimp + local.is_ebreak;
        builder.assert_bool(local.is_real);
        builder.assert_eq(local.is_real, is_real_effect);
        //cpu state
        CPUState::<AB::F>::eval(
            builder,
            local.cpu_state,
            local.cpu_state.pc + AB::F::from_canonical_u32(DEFAULT_PC_INC),
            AB::Expr::from_canonical_u32(DEFAULT_PC_INC),
            local.is_real.into(),
            execution_shard,
        );

        let opcode = AB::Expr::from_canonical_u32(Opcode::AUIPC as u32) * local.is_auipc +
            AB::Expr::from_canonical_u32(Opcode::UNIMP as u32) * local.is_unimp +
            AB::Expr::from_canonical_u32(Opcode::EBREAK as u32) * local.is_ebreak;
        //mem ops
        JTypeRegisterOp::<AB::F>::eval(
            builder,
            shard,
            clk,
            local.cpu_state.pc.into(),
            opcode,
            local.mem_ops,
            local.is_real.into(),
        );
        // Verify that the opcode is never UNIMP or EBREAK.
        builder.assert_zero(local.is_unimp);
        builder.assert_zero(local.is_ebreak);

        // Range check the pc.
        // SAFETY: `is_auipc` is already checked to be boolean above.
        // `BabyBearWordRangeChecker` assumes that the value is already checked to be a valid word.
        // This is checked implicitly, as the ADD ALU table checks that all inputs are valid words.
        // This check is done inside the `AddOperation`. Therefore, `pc` is a valid word.
        BabyBearWordRangeChecker::<AB::F>::range_check(
            builder,
            local.pc,
            local.pc_range_checker,
            local.is_auipc.into(),
        );
        builder.assert_eq(local.pc.reduce::<AB>(), local.cpu_state.pc);

        // Verify that op_a == pc + op_b, when `op_a_not_0 == 1`.
        builder
            .when(local.is_real - local.mem_ops.op_a_zero)
            .assert_word_eq(*a_word, local.add_op.value);
        builder.when(local.mem_ops.op_a_zero).assert_one(local.is_real);
        // AUIPC: rd = pc + imm. The add_op computes pc + imm (b_word).
        AddOperation::<AB::F>::eval(
            builder,
            *b_word,
            local.pc,
            local.add_op,
            local.is_auipc.into(),
        );
    }
}

impl<F: Field> MachineAir<F> for AuipcChip {
    type Record = ExecutionRecord;

    type Program = Program;

    fn name(&self) -> String {
        "Auipc".to_string()
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        output: &mut ExecutionRecord,
    ) -> CompressedMatrix<F> {
        let nb_rows = input.auipc_events.len();
        let size_log2 = input.fixed_log2_rows::<F, _>(self);
        let padded_nb_rows = padded_rows_threshold(next_power_of_two(nb_rows, size_log2));
        let chunk_size = std::cmp::max(nb_rows / num_cpus::get(), 1);
        let mut values = zeroed_f_vec(nb_rows * NUM_AUIPC_COLS);
        let shard = input.execution_shard();
        let blu_events = values
            .chunks_mut(chunk_size * NUM_AUIPC_COLS)
            .enumerate()
            .par_bridge()
            .map(|(i, rows)| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                rows.chunks_mut(NUM_AUIPC_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut AuipcColumns<F> = row.borrow_mut();
                    let (record, event) = &input.auipc_events[idx];
                    cols.is_auipc = F::from_bool(event.opcode == Opcode::AUIPC);
                    cols.is_unimp = F::from_bool(event.opcode == Opcode::UNIMP);
                    cols.is_ebreak = F::from_bool(event.opcode == Opcode::EBREAK);
                    cols.pc = event.pc.into();
                    if event.opcode == Opcode::AUIPC {
                        cols.pc_range_checker.populate(cols.pc, &mut blu);
                    }
                    cols.cpu_state.populate(&mut blu, record.clk, event.pc, shard);
                    cols.mem_ops.populate(&mut blu, *record);
                    cols.is_real = F::one();
                    cols.add_op.populate(&mut blu, event.b, event.pc);
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_events.iter().collect_vec());

        let main = RowMajorMatrix::new(values, NUM_AUIPC_COLS);
        CompressedMatrix::new(main, PaddingRow::Zero { width: NUM_AUIPC_COLS }, padded_nb_rows)
    }

    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            !shard.auipc_events.is_empty()
        }
    }

    fn local_only(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::BorrowMut;

    use dt_core_executor::{
        ExecutionError, ExecutionRecord, Executor, Instruction, Opcode, Program,
    };
    use dt_stark::{
        air::MachineAir, baby_bear_poseidon2::BabyBearPoseidon2, chip_name, CpuProver, DTCoreOpts,
        MachineProver, Val,
    };
    use p3_baby_bear::BabyBear;
    use p3_field::AbstractField;
    use p3_matrix::dense::RowMajorMatrix;

    use crate::{
        control_flow::{AuipcChip, AuipcColumns},
        io::DTStdin,
        riscv::RiscvAir,
        utils::run_malicious_test,
    };

    #[test]
    fn test_malicious_auipc() {
        let instructions = vec![
            Instruction::new(Opcode::AUIPC, 29, 12, 12, true, true),
            Instruction::new(Opcode::ADD, 10, 0, 0, false, false),
        ];
        let program = Program::new(instructions, 0, 0);
        let stdin = DTStdin::new();

        type P = CpuProver<BabyBearPoseidon2, RiscvAir<BabyBear>>;

        let malicious_trace_pv_generator = |prover: &P,
                                            record: &mut ExecutionRecord|
         -> Vec<(
            String,
            dt_stark::sumcheck::trace::CompressedMatrix<Val<BabyBearPoseidon2>>,
        )> {
            // Create a malicious record where the AUIPC instruction result is incorrect.
            let mut malicious_record = record.clone();
            malicious_record.auipc_events[0].1.a = 8;
            prover
                .generate_traces(&malicious_record)
                .into_iter()
                .map(|(n, m)| {
                    (n, dt_stark::sumcheck::trace::CompressedMatrix::from_full_matrix_no_padding(m))
                })
                .collect()
        };

        let result =
            run_malicious_test::<P>(program, stdin, Box::new(malicious_trace_pv_generator));
        assert!(result.is_err() && result.unwrap_err().is_local_cumulative_sum_failing());
    }

    #[test]
    fn test_malicious_multiple_opcode_flags() {
        let instructions = vec![
            Instruction::new(Opcode::AUIPC, 29, 12, 12, true, true),
            Instruction::new(Opcode::ADD, 10, 0, 0, false, false),
        ];
        let program = Program::new(instructions, 0, 0);
        let stdin = DTStdin::new();

        type P = CpuProver<BabyBearPoseidon2, RiscvAir<BabyBear>>;

        let malicious_trace_pv_generator = |prover: &P,
                                            record: &mut ExecutionRecord|
         -> Vec<(
            String,
            dt_stark::sumcheck::trace::CompressedMatrix<Val<BabyBearPoseidon2>>,
        )> {
            // Modify the branch chip to have a row that has multiple opcode flags set.
            let mut traces = prover
                .generate_traces(record)
                .into_iter()
                .map(|(n, m)| {
                    (n, dt_stark::sumcheck::trace::CompressedMatrix::from_full_matrix_no_padding(m))
                })
                .collect::<Vec<_>>();
            let auipc_chip_name = chip_name!(AuipcChip, BabyBear);
            for (chip_name, trace) in traces.iter_mut() {
                if *chip_name == auipc_chip_name {
                    let first_row: &mut [BabyBear] = trace.main.row_mut(0);
                    let first_row: &mut AuipcColumns<BabyBear> = first_row.borrow_mut();
                    assert!(first_row.is_auipc == BabyBear::one());
                    first_row.is_unimp = BabyBear::one();
                }
            }
            traces
        };

        let result =
            run_malicious_test::<P>(program, stdin, Box::new(malicious_trace_pv_generator));
        let auipc_chip_name = chip_name!(AuipcChip, BabyBear);
        assert!(result.is_err() && result.unwrap_err().is_constraints_failing(&auipc_chip_name));
    }

    #[test]
    fn test_unimpl() {
        let instructions = vec![Instruction::new(Opcode::UNIMP, 29, 12, 0, true, true)];
        let program = Program::new(instructions, 0, 0);
        let stdin = DTStdin::new();

        let mut runtime = Executor::new(program, DTCoreOpts::default());
        runtime.maximal_shapes = None;
        runtime.write_vecs(&stdin.buffer);
        let result = runtime.execute();

        assert!(result.is_err() && result.unwrap_err() == ExecutionError::Unimplemented());
    }

    #[test]
    fn test_ebreak() {
        let instructions = vec![Instruction::new(Opcode::EBREAK, 29, 12, 0, true, true)];
        let program = Program::new(instructions, 0, 0);
        let stdin = DTStdin::new();

        let mut runtime = Executor::new(program, DTCoreOpts::default());
        runtime.maximal_shapes = None;
        runtime.write_vecs(&stdin.buffer);
        let result = runtime.execute();

        assert!(result.is_err() && result.unwrap_err() == ExecutionError::Breakpoint());
    }
}
