//! PolyAir failure-oriented checker (A3).
//!
//! Proves that the PolyAir verifier *rejects* deliberately-corrupted traces.
//! For each malicious case we install a `malicious_trace_pv_generator` hook into
//! `prove_polyair::sc_prove_core` that tampers one cell of one chip's main trace,
//! then assert that `machine().verify(...)` returns `Err`.
//!
//! A passing run means every corruption was caught. **If any case unexpectedly
//! verifies `Ok`, that is a real PolyAir constraint bug (a dropped gate/global
//! constraint): the run STOPS with a non-zero exit and a loud `UNEXPECTED PASS`
//! marker. Do not weaken the test to make it pass — fix the adapter.**
//!
//! This is an `examples/` target (needs `test-artifacts`, a dev-dependency; see
//! `check_polyair_prove.rs` for why bins can't reach it).
//!
//! Usage:
//!   cargo run -r --example check_polyair_failure -p dt-core-machine --features eth
//!
//! ## Malicious cases (each documents the four required elements):
//!
//! 1. lookup-failure  — chip `AddPolyAir`,   col `add_operation.value[0]`
//! 2. gate-failure    — chip `LtPolyAir`,     col `lt_operation.result.not_eq_inv`
//! 3. gate-failure    — chip `Uint256MulModPolyAir`, col `y_memory[0].compare_clk`
//! 4. global-failure  — chip `Global`, canonical `cumulative.x[0]`

#![allow(clippy::print_stdout, clippy::print_stderr)]

#[cfg(not(any(feature = "eth", feature = "legacy")))]
fn main() {
    eprintln!("check_polyair_failure requires --features eth or --features legacy");
}

#[cfg(any(feature = "eth", feature = "legacy"))]
fn main() {
    inner::run();
}

#[cfg(any(feature = "eth", feature = "legacy"))]
mod inner {
    use dt_core_executor::{DTContext, ExecutionRecord, Instruction, Opcode, Program};
    use dt_core_machine::{alu::NUM_ADD_COLS, io::DTStdin, riscv::riscv_polyair::RiscvPolyAir};
    use dt_stark::{
        sumcheck::{config::SCStarkGenericConfig, trace::CompressedMatrix},
        DTCoreOpts, MachineVerificationError, Val,
    };
    use p3_field::AbstractField;
    use p3_matrix::Matrix;
    use polyair::prover::SCMachineProver as _;

    #[cfg(feature = "eth")]
    type CoreSC = dt_stark::koalabear_poseidon2::koala_bear_poseidon2::SCKoalaBearPoseidon2;
    #[cfg(all(feature = "legacy", not(feature = "eth")))]
    type CoreSC = dt_stark::baby_bear_poseidon2::SCBabyBearPoseidon2;

    #[cfg(feature = "eth")]
    const D: usize = 5;
    #[cfg(all(feature = "legacy", not(feature = "eth")))]
    const D: usize = 4;

    type V = Val<CoreSC>;
    type PolyairProver = polyair::prover::SumcheckProver<CoreSC, RiscvPolyAir<V>, D>;

    /// How a malicious case classifies its corruption.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum FailKind {
        Lookup,
        Gate,
        Global,
    }

    impl FailKind {
        fn label(self) -> &'static str {
            match self {
                FailKind::Lookup => "lookup",
                FailKind::Gate => "gate",
                FailKind::Global => "global",
            }
        }
    }

    /// A documented malicious case (the four required elements).
    struct MaliciousCase {
        /// (1) target chip `name()` String (the trace-Vec key).
        chip: &'static str,
        /// (2) target row & column, by semantic name.
        target: &'static str,
        /// (3) mutation applied.
        mutation: &'static str,
        /// (4) why it is classified lookup / gate / global.
        why: &'static str,
        kind: FailKind,
    }

    /// Locate a real (non-padding) row of `mat` whose `is_real_col` holds a
    /// non-zero value. Falls back to the first non-padding row, then row 0.
    fn first_real_row(mat: &p3_matrix::dense::RowMajorMatrix<V>, is_real_col: usize) -> usize {
        let w = mat.width();
        let h = mat.values.len() / w.max(1);
        for r in 0..h {
            if mat.values[r * w + is_real_col] != V::zero() {
                return r;
            }
        }
        0
    }

    /// Build a `malicious_trace_pv_generator` closure that finds `chip` by name
    /// in the freshly generated traces, decompresses it, mutates one cell, and
    /// repacks. Shards not containing the chip pass through unchanged.
    ///
    /// `pick` receives the decompressed matrix and returns `(row, col)`; the cell
    /// at that position is overwritten with `new_value(old)`.
    ///
    /// `hit` is set to true (across all shards / all parallel invocations) the
    /// first time the target chip is found and a cell is mutated. The caller
    /// checks it after proving: if a corruption "verifies OK" but `hit` is false,
    /// the chip name was simply wrong (a test bug), NOT a dropped constraint.
    fn make_generator<Pick, NewVal>(
        chip: &'static str,
        hit: std::sync::Arc<std::sync::atomic::AtomicBool>,
        pick: Pick,
        new_value: NewVal,
    ) -> impl Fn(&PolyairProver, &mut ExecutionRecord) -> Vec<(String, CompressedMatrix<V>)> + Send + Sync
    where
        Pick: Fn(&p3_matrix::dense::RowMajorMatrix<V>) -> (usize, usize) + Send + Sync,
        NewVal: Fn(V) -> V + Send + Sync,
    {
        move |prover: &PolyairProver, record: &mut ExecutionRecord| {
            let mut traces = prover.generate_traces(record);
            for (name, m) in traces.iter_mut() {
                if name == chip {
                    let mut full = m.decompress();
                    let w = full.width();
                    if w == 0 || full.values.is_empty() {
                        continue;
                    }
                    let h = full.values.len() / w;
                    let (row, col) = pick(&full);
                    let off = row * w + col;
                    if off < full.values.len() {
                        let old = full.values[off];
                        let newv = new_value(old);
                        full.values[off] = newv;
                        hit.store(true, std::sync::atomic::Ordering::SeqCst);
                        if std::env::var("POLYAIR_FAIL_DEBUG").is_ok() {
                            eprintln!(
                                "[gen] chip={name} width={w} height={h} row={row} col={col} \
                                 old={old:?} new={newv:?}"
                            );
                        }
                    }
                    *m = CompressedMatrix::from_full_matrix_no_padding(full);
                }
            }
            traces
        }
    }

    /// Prove `program` with `generator` installed, then verify. Returns the
    /// verify `Result` (we expect `Err`).
    fn run_one_malicious<G>(
        program: Program,
        stdin: &DTStdin,
        generator: G,
    ) -> Result<(), MachineVerificationError<CoreSC>>
    where
        G: Fn(&PolyairProver, &mut ExecutionRecord) -> Vec<(String, CompressedMatrix<V>)>
            + Send
            + Sync
            + 'static,
    {
        let core_machine = RiscvPolyAir::sc_machine(CoreSC::default());
        let core_prover = PolyairProver::new(core_machine);
        let (host_pk, vk) = core_prover.setup(&program);
        let pk = core_prover.pk_to_device(&host_pk);

        let (proof, _pv, _cycles) =
            dt_core_machine::utils::prove_polyair::sc_prove_core::<CoreSC, PolyairProver, D>(
                &core_prover,
                &pk,
                &vk,
                program,
                stdin,
                DTCoreOpts::default(),
                DTContext::default(),
                Some(Box::new(generator)),
            )
            .expect("prove failed (prover should still produce a proof for a corrupted trace)");

        let challenger = core_prover.config().mlchallenger();
        core_prover.machine().verify(&vk, &proof, &mut challenger.clone(), 1, 0)
    }

    /// An ADD program: forces the Add chip to have real rows.
    /// `x7 = 7 + 11`, padded with extra ADDs (mirrors the proven branch test shape).
    fn add_program() -> Program {
        let instructions = vec![
            Instruction::new(Opcode::ADD, 5, 0, 7, false, true),
            Instruction::new(Opcode::ADD, 6, 0, 11, false, true),
            Instruction::new(Opcode::ADD, 7, 5, 6, false, false),
            Instruction::new(Opcode::ADD, 28, 0, 5, false, true),
            Instruction::new(Opcode::ADD, 28, 0, 5, false, true),
        ];
        Program::new(instructions, 0, 0)
    }

    /// An SLT program with `b != c`, so the Lt chip has a row with
    /// `is_comp_eq == 0` (the precondition for the `not_eq_inv` gate to bind).
    fn slt_program() -> Program {
        let instructions = vec![
            Instruction::new(Opcode::ADD, 5, 0, 3, false, true),
            Instruction::new(Opcode::ADD, 6, 0, 9, false, true),
            // SLT x7, x5, x6  → 3 < 9, b != c so comparison bytes differ.
            Instruction::new(Opcode::SLT, 7, 5, 6, false, false),
            Instruction::new(Opcode::ADD, 28, 0, 5, false, true),
            Instruction::new(Opcode::ADD, 28, 0, 5, false, true),
        ];
        Program::new(instructions, 0, 0)
    }

    pub fn run() {
        use std::sync::{
            atomic::{AtomicBool, Ordering},
            Arc,
        };

        dt_core_machine::utils::setup_logger();

        println!("PolyAir failure checker (D={D}) — expecting every corruption to be REJECTED.\n");

        // ----- Add chip column offsets (from add_polyair.rs) -----
        let add_cols = NUM_ADD_COLS;
        // ----- Lt chip column offsets (from lt_polyair.rs) -----
        const COL_LT_NOT_EQ_INV: usize = 51;
        const COL_LT_IS_SLT: usize = 43;
        const COL_LT_IS_SLTU: usize = 44;
        // ----- Uint256 column offsets (from uint256_polyair.rs) -----
        // COL_Y_MEM_BASE = 4 + 8*13 = 108; first y_memory read compare_clk = 108 + 6.
        const COL_U256_Y0_COMPARE_CLK: usize = 108 + 6;
        const COL_U256_IS_REAL: usize = 417 - 1;

        // is_real column for Add is the last column.
        let add_is_real = add_cols - 1;

        // Each runner returns (verify result, chip-was-hit). `hit == false` means
        // the chip name never matched, i.e. the corruption never happened — a
        // TEST error, not a dropped constraint.
        type Runner = Box<dyn Fn() -> (Result<(), MachineVerificationError<CoreSC>>, bool)>;
        let mut cases: Vec<(MaliciousCase, Runner)> = Vec::new();

        // (1) lookup-failure: Add result byte feeds the U8Range lookup.
        cases.push((
            MaliciousCase {
                chip: "AddPolyAir",
                target: "add_operation.value[0] (result byte 0, col 0), first real row",
                mutation: "value[0] += 1",
                why: "value[0] is sent as a U8Range lookup value (add_op_lookup); changing it \
                       unbalances the Byte chip U8Range send/recv → lookup failure.",
                kind: FailKind::Lookup,
            },
            Box::new(move || {
                let hit = Arc::new(AtomicBool::new(false));
                let r = run_one_malicious(
                    add_program(),
                    &DTStdin::new(),
                    make_generator(
                        "AddPolyAir",
                        hit.clone(),
                        move |m| (first_real_row(m, add_is_real), 0usize),
                        |old| old + V::one(),
                    ),
                );
                (r, hit.load(Ordering::SeqCst))
            }),
        ));

        // (2) gate-failure (ALU): Lt not_eq_inv is gate-constrained, not in any lookup.
        cases.push((
            MaliciousCase {
                chip: "LtPolyAir",
                target: "lt_operation.result.not_eq_inv (col 51), first real row",
                mutation: "not_eq_inv += 1",
                why: "eval() binds not_eq_inv*(b_comp-c_comp)=is_real (lt_polyair.rs:428). \
                       lookup() sends msb/u8range/cpu_state/alu_adapter/bitvec only — not_eq_inv \
                       is neither a lookup value nor range-checked → pure gate failure.",
                kind: FailKind::Gate,
            },
            Box::new(move || {
                let hit = Arc::new(AtomicBool::new(false));
                let r = run_one_malicious(
                    slt_program(),
                    &DTStdin::new(),
                    make_generator(
                        "LtPolyAir",
                        hit.clone(),
                        move |m| {
                            // is_real = is_slt + is_sltu; find a row where either is set.
                            let w = m.width();
                            let h = m.values.len() / w.max(1);
                            let mut row = 0;
                            for r in 0..h {
                                let slt = m.values[r * w + COL_LT_IS_SLT];
                                let sltu = m.values[r * w + COL_LT_IS_SLTU];
                                if slt != V::zero() || sltu != V::zero() {
                                    row = r;
                                    break;
                                }
                            }
                            (row, COL_LT_NOT_EQ_INV)
                        },
                        |old| old + V::one(),
                    ),
                );
                (r, hit.load(Ordering::SeqCst))
            }),
        ));

        // (3) gate-failure (precompile): Uint256 memory compare_clk boolean.
        cases.push((
            MaliciousCase {
                chip: "Uint256MulModPolyAir",
                target: "y_memory[0].compare_clk (col 114), first real row",
                mutation: "compare_clk set to 2 (breaks the boolean)",
                why: "eval()→memory_timestamp_gate_constraints binds compare_clk*(1-compare_clk)=0. \
                       Memory lookups send prev_shard/prev_clk/addr/value and diff_* range checks \
                       only; compare_clk is neither a lookup value nor range-checked → gate failure.",
                kind: FailKind::Gate,
            },
            Box::new(move || {
                let hit = Arc::new(AtomicBool::new(false));
                let r = run_one_malicious(
                    Program::from(test_artifacts::UINT256_MUL_ELF).expect("load UINT256_MUL_ELF"),
                    &DTStdin::new(),
                    make_generator(
                        "Uint256MulModPolyAir",
                        hit.clone(),
                        move |m| (first_real_row(m, COL_U256_IS_REAL), COL_U256_Y0_COMPARE_CLK),
                        |_old| V::from_canonical_u32(2),
                    ),
                );
                (r, hit.load(Ordering::SeqCst))
            }),
        ));

        // (4) global-failure: canonical Global projective cumulative X limb.
        cases.push((
            MaliciousCase {
                chip: "Global",
                target: "cumulative.x[0] (canonical col 113), last real row",
                mutation: "cumulative.x[0] += 1",
                why: "The shared 102-residual relation binds every selected cumulative limb to \
                       the complete mixed-add output, and the projective-chain send also binds the \
                       same terminal row. Corrupting this limb must fail the Global AIR.",
                kind: FailKind::Gate,
            },
            Box::new(move || {
                let hit = Arc::new(AtomicBool::new(false));
                let r = run_one_malicious(
                    add_program(),
                    &DTStdin::new(),
                    make_generator(
                        "Global",
                        hit.clone(),
                        move |m| {
                            let w = m.width();
                            let h = m.values.len() / w.max(1);
                            // Canonical Global is_real is column 23; pick the last real row.
                            const COL_GLOBAL_IS_REAL: usize = 23;
                            let mut row = 0;
                            for r in 0..h {
                                if m.values[r * w + COL_GLOBAL_IS_REAL] != V::zero() {
                                    row = r;
                                }
                            }
                            (row, 113)
                        },
                        |old| old + V::one(),
                    ),
                );
                (r, hit.load(Ordering::SeqCst))
            }),
        ));

        let mut unexpected_pass: Vec<String> = Vec::new();
        let mut test_errors: Vec<String> = Vec::new();
        for (case, runner) in &cases {
            println!("---- [{}] {} ----", case.kind.label(), case.chip);
            println!("  target  : {}", case.target);
            println!("  mutation: {}", case.mutation);
            println!("  class   : {}", case.why);
            let (result, hit) = runner();
            if !hit {
                println!(
                    "  [TEST ERROR] target chip '{}' was never found in any shard — corruption \
                     never applied. Fix the chip name / activating program; this case is \
                     inconclusive (NOT a constraint result).\n",
                    case.chip
                );
                test_errors.push(format!("{} (chip not found)", case.chip));
                continue;
            }
            match result {
                Err(e) => {
                    println!("  [OK] corruption REJECTED: {e:?}\n");
                }
                Ok(()) => {
                    println!(
                        "  [UNEXPECTED PASS] corruption was ACCEPTED — suspected dropped \
                         {} constraint in {} (real PolyAir bug).\n",
                        case.kind.label(),
                        case.chip
                    );
                    unexpected_pass.push(format!("{} / {}", case.chip, case.kind.label()));
                }
            }
        }

        println!("========== Summary ==========");
        println!("cases run:      {}", cases.len());
        println!("rejected (OK):  {}", cases.len() - unexpected_pass.len() - test_errors.len());
        if !test_errors.is_empty() {
            println!("test errors:    {test_errors:?}");
        }
        if unexpected_pass.is_empty() && test_errors.is_empty() {
            println!("[PASS] all corruptions were rejected.");
        } else if !unexpected_pass.is_empty() {
            println!(
                "[FAIL] UNEXPECTED PASS — suspected real PolyAir constraint bug(s): {unexpected_pass:?}"
            );
            println!("       STOP: report to planner; do NOT weaken these tests. Phase A is NOT complete.");
            std::process::exit(1);
        } else {
            println!(
                "[FAIL] TEST ERROR(S) — some cases did not apply their corruption; fix and rerun."
            );
            std::process::exit(1);
        }
    }
}
