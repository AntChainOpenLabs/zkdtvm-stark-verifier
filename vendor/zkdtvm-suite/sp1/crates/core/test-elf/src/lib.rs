//! Provides pre-built ELF binaries for core instruction coverage testing.
//!
//! The `ALL_INSTRUCTIONS_ELF` binary exercises all RV32IM instructions
//! (no syscalls or precompiles) and fits within a single shard.
//!
//! To rebuild the ELF after modifying the program source:
//! ```sh
//! cd crates/core/test-elf/program && cargo prove build
//! cp program/target/elf-compilation/riscv32im-succinct-zkvm-elf/release/all-instructions-test elf/
//! ```

/// ELF binary that covers all RV32IM instructions:
/// ADD, SUB, AND, OR, XOR, SLL, SRL, SRA, SLT, SLTU,
/// ADDI, SLTI, SLTIU, XORI, ORI, ANDI, SLLI, SRLI, SRAI,
/// MUL, MULH, MULHSU, MULHU, DIV, DIVU, REM, REMU,
/// LW, LH, LHU, LB, LBU, SW, SH, SB,
/// BEQ, BNE, BLT, BGE, BLTU, BGEU,
/// JAL, JALR, LUI, AUIPC.
///
/// No syscalls or precompiles. Fits in one shard (< 2^15 instructions).
pub const ALL_INSTRUCTIONS_ELF: &[u8] = include_bytes!("../elf/all-instructions-test");
