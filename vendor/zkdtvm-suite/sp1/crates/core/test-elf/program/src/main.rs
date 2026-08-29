//! A test program that exercises all RV32IM instructions without any syscalls or precompiles.
//!
//! Instruction coverage:
//!   R-type ALU:  ADD, SUB, AND, OR, XOR, SLL, SRL, SRA, SLT, SLTU
//!   I-type ALU:  ADDI, SLTI, SLTIU, XORI, ORI, ANDI, SLLI, SRLI, SRAI
//!   M extension: MUL, MULH, MULHSU, MULHU, DIV, DIVU, REM, REMU
//!   Load:        LW, LH, LHU, LB, LBU
//!   Store:       SW, SH, SB
//!   Branch:      BEQ, BNE, BLT, BGE, BLTU, BGEU  (both taken and not-taken)
//!   Jump:        JAL, JALR
//!   Upper:       LUI, AUIPC
//!
//! The program is intentionally small (< 2^15 total instructions, fits in one shard).
//! It avoids all syscalls and precompiles — only the implicit HALT at program exit.

#![no_main]
dt_zkvm::entrypoint!(main);

pub fn main() {
    // ================================================================
    // Part 1: R-type ALU, I-type ALU, M extension, LUI, AUIPC
    // ================================================================
    unsafe {
        core::arch::asm!(
            // Initialize working registers with non-trivial values.
            // `li` expands to LUI+ADDI for large immediates, covering LUI implicitly too.
            "li   t1, 123",          // t1 = 123
            "li   t2, 17",           // t2 = 17

            // ---- R-type ALU (10 instructions) ----
            "add  t0, t1, t2",       // ADD
            "sub  t0, t1, t2",       // SUB
            "and  t0, t1, t2",       // AND
            "or   t0, t1, t2",       // OR
            "xor  t0, t1, t2",       // XOR
            "sll  t0, t1, t2",       // SLL
            "srl  t0, t1, t2",       // SRL
            "sra  t0, t1, t2",       // SRA
            "slt  t0, t1, t2",       // SLT
            "sltu t0, t1, t2",       // SLTU

            // ---- I-type ALU (9 instructions) ----
            "addi  t0, t1, 100",     // ADDI
            "slti  t0, t1, 100",     // SLTI
            "sltiu t0, t1, 100",     // SLTIU
            "xori  t0, t1, 0xFF",    // XORI
            "ori   t0, t1, 0xFF",    // ORI
            "andi  t0, t1, 0xFF",    // ANDI
            "slli  t0, t1, 7",       // SLLI
            "srli  t0, t1, 7",       // SRLI
            "srai  t0, t1, 7",       // SRAI

            // ---- M extension (8 instructions) ----
            "mul    t0, t1, t2",     // MUL
            "mulh   t0, t1, t2",     // MULH
            "mulhsu t0, t1, t2",     // MULHSU
            "mulhu  t0, t1, t2",     // MULHU
            "div    t0, t1, t2",     // DIV   (123 / 17 = 7)
            "divu   t0, t1, t2",     // DIVU
            "rem    t0, t1, t2",     // REM   (123 % 17 = 4)
            "remu   t0, t1, t2",     // REMU

            // ---- Upper immediate (2 instructions) ----
            "lui   t0, 0x12345",     // LUI
            "auipc t0, 0",           // AUIPC

            out("t0") _,
            out("t1") _,
            out("t2") _,
        );
    }

    // Also test with negative values to exercise signed arithmetic paths.
    unsafe {
        core::arch::asm!(
            "li   t1, -100",         // negative value (LUI + ADDI)
            "li   t2, 7",

            "add  t0, t1, t2",       // ADD with negative
            "sub  t0, t1, t2",       // SUB with negative
            "mul  t0, t1, t2",       // MUL with negative
            "mulh t0, t1, t2",       // MULH with negative
            "div  t0, t1, t2",       // DIV  signed: -100 / 7 = -14
            "rem  t0, t1, t2",       // REM  signed: -100 % 7 = -2
            "sra  t0, t1, t2",       // SRA of negative (sign extension)
            "slt  t0, t1, t2",       // SLT: -100 < 7 → 1
            "sltu t0, t1, t2",       // SLTU: large unsigned > 7 → 0

            out("t0") _,
            out("t1") _,
            out("t2") _,
        );
    }

    // ================================================================
    // Part 2: Load and Store (all widths)
    // ================================================================
    let mut buf = [0u32; 4]; // 16 bytes, 4-byte aligned
    let ptr = buf.as_mut_ptr() as *mut u8;
    unsafe {
        core::arch::asm!(
            "li   t0, 0x12345678",

            // ---- Store ----
            "sw   t0, 0({ptr})",     // SW  (store word)
            "sh   t0, 4({ptr})",     // SH  (store halfword)
            "sb   t0, 6({ptr})",     // SB  (store byte)

            // ---- Load ----
            "lw   t0, 0({ptr})",     // LW  (load word)
            "lh   t0, 4({ptr})",     // LH  (load halfword, sign-extended)
            "lhu  t0, 4({ptr})",     // LHU (load halfword, zero-extended)
            "lb   t0, 6({ptr})",     // LB  (load byte, sign-extended)
            "lbu  t0, 6({ptr})",     // LBU (load byte, zero-extended)

            ptr = in(reg) ptr,
            out("t0") _,
        );
    }

    // Additional load/store with different values to exercise more patterns.
    unsafe {
        core::arch::asm!(
            "li   t0, -1",           // 0xFFFFFFFF
            "sw   t0, 8({ptr})",     // SW  (all 1s)
            "lh   t1, 8({ptr})",     // LH  → should sign-extend to 0xFFFFFFFF
            "lhu  t1, 8({ptr})",     // LHU → should zero-extend to 0x0000FFFF
            "lb   t1, 8({ptr})",     // LB  → should sign-extend to 0xFFFFFFFF
            "lbu  t1, 8({ptr})",     // LBU → should zero-extend to 0x000000FF

            ptr = in(reg) ptr,
            out("t0") _,
            out("t1") _,
        );
    }

    // ================================================================
    // Part 3: Branch instructions (both taken and not-taken paths)
    // ================================================================
    unsafe {
        core::arch::asm!(
            "li t0, 10",
            "li t1, 20",
            "li t2, 10",

            // BEQ taken (t0 == t2 → 10 == 10)
            "beq  t0, t2, 1f",
            "nop",
            "1:",
            // BEQ not taken (t0 != t1 → 10 != 20)
            "beq  t0, t1, 2f",
            "2:",

            // BNE taken (t0 != t1)
            "bne  t0, t1, 3f",
            "nop",
            "3:",
            // BNE not taken (t0 == t2)
            "bne  t0, t2, 4f",
            "4:",

            // BLT taken (10 < 20, signed)
            "blt  t0, t1, 5f",
            "nop",
            "5:",
            // BLT not taken (20 < 10 is false)
            "blt  t1, t0, 6f",
            "6:",

            // BGE taken (20 >= 10, signed)
            "bge  t1, t0, 7f",
            "nop",
            "7:",
            // BGE not taken (10 >= 20 is false)
            "bge  t0, t1, 8f",
            "8:",

            // BLTU taken (10 <u 20)
            "bltu t0, t1, 9f",
            "nop",
            "9:",
            // BLTU not taken (20 <u 10 is false)
            "bltu t1, t0, 10f",
            "10:",

            // BGEU taken (20 >=u 10)
            "bgeu t1, t0, 11f",
            "nop",
            "11:",
            // BGEU not taken (10 >=u 20 is false)
            "bgeu t0, t1, 12f",
            "12:",

            out("t0") _,
            out("t1") _,
            out("t2") _,
        );
    }

    // Also test branches with negative values (signed vs unsigned difference).
    unsafe {
        core::arch::asm!(
            "li t0, -1",             // 0xFFFFFFFF (large unsigned, -1 signed)
            "li t1, 1",

            // BLT taken: -1 < 1 (signed)
            "blt  t0, t1, 13f",
            "nop",
            "13:",
            // BLTU not taken: 0xFFFFFFFF <u 1 is false
            "bltu t0, t1, 14f",
            "14:",
            // BGEU taken: 0xFFFFFFFF >=u 1 (unsigned)
            "bgeu t0, t1, 15f",
            "nop",
            "15:",

            out("t0") _,
            out("t1") _,
        );
    }

    // ================================================================
    // Part 4: JAL and JALR
    // ================================================================
    unsafe {
        core::arch::asm!(
            // JAL: jump forward, save return address in t0
            "jal   t0, 16f",
            "16:",

            // JALR: compute target address, then jump
            "auipc t0, 0",          // t0 = PC of this instruction
            "addi  t0, t0, 12",     // t0 = address of label 17 (3 instructions * 4 bytes)
            "jalr  t1, t0, 0",      // jump to t0, save return in t1
            "17:",

            out("t0") _,
            out("t1") _,
        );
    }

    // ================================================================
    // Part 5: A small loop to generate more trace rows
    // ================================================================
    // This loop naturally generates ADDI, BNE, ADD, etc. from Rust code,
    // providing additional real rows beyond the inline assembly.
    let mut acc: u32 = 0;
    for i in 0u32..64 {
        acc = acc.wrapping_add(i.wrapping_mul(i));
        acc ^= i << (i & 7);
    }

    // Use black_box to prevent the compiler from optimizing away the computation.
    core::hint::black_box(acc);
    core::hint::black_box(&buf);
}
