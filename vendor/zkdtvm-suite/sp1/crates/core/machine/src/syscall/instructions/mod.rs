use columns::NUM_SYSCALL_INSTR_COLS;
use p3_air::BaseAir;

pub mod air;
pub mod columns;
pub mod syscall_instrs_polyair;
pub mod trace;

#[derive(Default)]
pub struct SyscallInstrsChip;

impl<F> BaseAir<F> for SyscallInstrsChip {
    fn width(&self) -> usize {
        NUM_SYSCALL_INSTR_COLS
    }
}

#[cfg(test)]
mod tests {
    use dt_core_executor::{ExecutionRecord, Instruction, Opcode, Program};
    use dt_stark::{
        air::MachineAir, baby_bear_poseidon2::BabyBearPoseidon2, chip_name, CpuProver,
        MachineProver, Val,
    };
    use dt_zkvm::syscalls::{COMMIT, COMMIT_DEFERRED_PROOFS, HALT, SHA_EXTEND};
    use p3_baby_bear::BabyBear;
    use p3_matrix::dense::RowMajorMatrix;

    use crate::{
        io::DTStdin, riscv::RiscvAir, syscall::instructions::SyscallInstrsChip,
        utils::run_malicious_test,
    };

    #[test]
    fn test_malicious_next_pc() {
        struct TestCase {
            program: Vec<Instruction>,
            incorrect_next_pc: u32,
        }

        let test_cases = vec![
            TestCase {
                program: vec![
                    Instruction::new(Opcode::ADD, 5, 0, HALT, false, true), /* Set the syscall
                                                                             * code in register
                                                                             * x5. */
                    Instruction::new(Opcode::ECALL, 5, 10, 11, false, false), // Call the syscall.
                    Instruction::new(Opcode::ADD, 30, 0, 100, false, true),
                ],
                incorrect_next_pc: 8, // The correct next_pc is 0.
            },
            TestCase {
                program: vec![
                    Instruction::new(Opcode::ADD, 5, 0, SHA_EXTEND, false, true), /* Set the syscall code in register x5. */
                    Instruction::new(Opcode::ADD, 10, 0, 40, false, true),        /* Set the syscall
                                                                                   * arg1 to 40. */
                    Instruction::new(Opcode::ECALL, 5, 10, 11, false, false), // Call the syscall.
                    Instruction::new(Opcode::ADD, 30, 0, 100, false, true),
                ],
                incorrect_next_pc: 0, // The correct next_pc is 12.
            },
        ];

        for test_case in test_cases {
            let program = Program::new(test_case.program, 0, 0);
            let stdin = DTStdin::new();

            type P = CpuProver<BabyBearPoseidon2, RiscvAir<BabyBear>>;

            let malicious_trace_pv_generator = move |prover: &P,
                                                     record: &mut ExecutionRecord|
                  -> Vec<(
                String,
                dt_stark::sumcheck::trace::CompressedMatrix<Val<BabyBearPoseidon2>>,
            )> {
                // Create a malicious record where the next pc is set to the incorrect value.
                let mut malicious_record = record.clone();

                // There can be multiple shards for programs with syscalls, so need to figure
                // out which record is for a CPU shard.
                if !malicious_record.cpu_events == 0 {
                    malicious_record.syscall_events[0].1.next_pc = test_case.incorrect_next_pc;
                }

                prover
                        .generate_traces(&malicious_record)
                        .into_iter()
                        .map(|(n, m)| (n, dt_stark::sumcheck::trace::CompressedMatrix::from_full_matrix_no_padding(m)))
                        .collect()
            };

            let result =
                run_malicious_test::<P>(program, stdin, Box::new(malicious_trace_pv_generator));
            let syscall_chip_name = chip_name!(SyscallInstrsChip, BabyBear);
            assert!(
                result.is_err() && result.unwrap_err().is_constraints_failing(&syscall_chip_name)
            );
        }
    }

    #[test]
    fn test_malicious_commit() {
        let instructions = vec![
            Instruction::new(Opcode::ADD, 5, 0, COMMIT, false, true), /* Set the syscall code in
                                                                       * register x5. */
            Instruction::new(Opcode::ADD, 10, 0, 0, false, false), /* Set the syscall code in
                                                                    * register x5. */
            Instruction::new(Opcode::ADD, 11, 0, 40, false, true), // Set the syscall arg1 to 40.
            Instruction::new(Opcode::ECALL, 5, 10, 11, false, false), // Call the syscall.
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
            record.public_values.committed_value_digest[0] = 10; // The correct value is 40.
            prover
                .generate_traces(record)
                .into_iter()
                .map(|(n, m)| {
                    (n, dt_stark::sumcheck::trace::CompressedMatrix::from_full_matrix_no_padding(m))
                })
                .collect()
        };

        let result =
            run_malicious_test::<P>(program, stdin, Box::new(malicious_trace_pv_generator));
        let syscall_chip_name = chip_name!(SyscallInstrsChip, BabyBear);
        assert!(result.is_err() && result.unwrap_err().is_constraints_failing(&syscall_chip_name));
    }

    #[test]
    fn test_malicious_commit_deferred() {
        let instructions = vec![
            Instruction::new(Opcode::ADD, 5, 0, COMMIT_DEFERRED_PROOFS, false, true), /* Set the
                                                                                       * syscall
                                                                                       * code in
                                                                                       * register
                                                                                       * x5. */
            Instruction::new(Opcode::ADD, 10, 0, 0, false, false), /* Set the syscall code in
                                                                    * register x5. */
            Instruction::new(Opcode::ADD, 11, 0, 40, false, true), // Set the syscall arg1 to 40.
            Instruction::new(Opcode::ECALL, 5, 10, 11, false, false), // Call the syscall.
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
            record.public_values.deferred_proofs_digest[0] = 10; // The correct value is 40.
            prover
                .generate_traces(record)
                .into_iter()
                .map(|(n, m)| {
                    (n, dt_stark::sumcheck::trace::CompressedMatrix::from_full_matrix_no_padding(m))
                })
                .collect()
        };

        let result =
            run_malicious_test::<P>(program, stdin, Box::new(malicious_trace_pv_generator));
        let syscall_chip_name = chip_name!(SyscallInstrsChip, BabyBear);
        assert!(result.is_err() && result.unwrap_err().is_constraints_failing(&syscall_chip_name));
    }
}
