pub mod air;
pub mod columns;
pub mod poseidon_permute_polyair;
pub mod trace;

#[cfg(feature = "babybear")]
pub mod poseidon2_bb;
#[cfg(feature = "babybear")]
pub use poseidon2_bb as poseidon2_inner;

#[cfg(feature = "koalabear")]
pub mod poseidon2_kb;
#[cfg(feature = "koalabear")]
pub use poseidon2_kb as poseidon2_inner;

pub const WIDTH: usize = 24;
pub const STATE_NUM_WORDS: usize = WIDTH;

use p3_field::Field;
use poseidon2_inner::{Poseidon2Air, RoundConstants};

pub struct Poseidon2PermuteChip<F: Field> {
    p3_poseidon2_permute: Poseidon2Air<F>,
}

impl<F: Field> Poseidon2PermuteChip<F> {
    pub const fn new(rc: RoundConstants<F>) -> Self {
        Self { p3_poseidon2_permute: Poseidon2Air::new(rc) }
    }
}

impl<F: Field> Default for Poseidon2PermuteChip<F> {
    fn default() -> Self {
        let round_constants = RoundConstants::<F>::default();
        Self { p3_poseidon2_permute: Poseidon2Air::new(round_constants) }
    }
}

#[cfg(test)]
pub mod permute_tests {
    use dt_core_executor::{
        syscalls::SyscallCode, DTContext, Executor, Instruction, Opcode, Program,
    };
    use dt_stark::{CpuProver, DTCoreOpts};
    use test_artifacts::{FIBONACCI_ELF, KECCAK_PERMUTE_ELF};

    use crate::{
        io::DTStdin,
        utils::{self},
    };

    pub fn poseidon2_permute_program() -> Program {
        let state_ptr = 100;
        let mut instructions = vec![Instruction::new(Opcode::ADD, 29, 0, 1, false, true)];
        for i in 0..24 {
            instructions.extend(vec![
                Instruction::new(Opcode::ADD, 30, 0, state_ptr + i * 4, false, true),
                Instruction::new(Opcode::SW, 29, 30, 0, false, true),
            ]);
        }
        instructions.extend(vec![
            Instruction::new(Opcode::ADD, 5, 0, SyscallCode::POSEIDON2_PERMUTE as u32, false, true),
            Instruction::new(Opcode::ADD, 10, 0, state_ptr, false, true),
            Instruction::new(Opcode::ECALL, 5, 10, 11, false, false),
        ]);

        Program::new(instructions, 0, 0)
    }

    pub fn keccak_program_elf() -> Program {
        Program::from(KECCAK_PERMUTE_ELF).unwrap()
    }
    pub fn fibonacci_program_elf() -> Program {
        Program::from(FIBONACCI_ELF).unwrap()
    }

    #[test]
    pub fn test_poseidon2_permute_program_execute() {
        utils::setup_logger();
        let program = poseidon2_permute_program();
        let mut runtime = Executor::new(program, DTCoreOpts::default());
        runtime.run().unwrap();
    }

    #[test]
    fn test_keccak_debug() {
        utils::setup_logger();
        let program = keccak_program_elf();
        let stdin = DTStdin::new();
        utils::run_test::<CpuProver<_, _>>(program, stdin).unwrap();
    }
    #[test]
    fn test_fibo_debug() {
        utils::setup_logger();
        let program = fibonacci_program_elf();
        let stdin = DTStdin::new();
        utils::run_test::<CpuProver<_, _>>(program, stdin).unwrap();
    }
}
