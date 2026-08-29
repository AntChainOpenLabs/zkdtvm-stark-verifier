//! PolyAir multi-program core prove + verify checker (A1).
//!
//! Runs a *suite* of ELFs through the PolyAir prove path
//! (`RiscvPolyAir::sc_machine` + `prove_polyair::sc_prove_core`) and verifies
//! each proof, so that every chip class (arithmetic / memory / control flow /
//! syscall / precompile) is exercised by at least one real proof rather than
//! only the smoke programs.
//!
//! This is an `examples/` target (not `src/bin/`) on purpose: it needs the
//! `test-artifacts` ELFs, which live in `[dev-dependencies]`. `test-artifacts`
//! cannot be promoted to a normal dependency because that closes a package
//! cycle (`dt-build -> dt-prover -> dt-core-machine -> test-artifacts ->
//! dt-build`); the cycle is tolerated only for dev-dependencies. examples are
//! dev targets, so they may use dev-dependencies.
//!
//! Usage:
//!   cargo run -r --example check_polyair_prove -p dt-core-machine --features eth -- [suite]
//!
//! `suite` is one of `smoke` (default), `eth-full`, `legacy-smoke`.

#![allow(clippy::print_stdout, clippy::print_stderr)]

#[cfg(not(any(feature = "eth", feature = "legacy")))]
fn main() {
    eprintln!("check_polyair_prove requires --features eth or --features legacy");
}

#[cfg(any(feature = "eth", feature = "legacy"))]
fn main() {
    inner::run();
}

#[cfg(any(feature = "eth", feature = "legacy"))]
mod inner {
    use dt_core_machine::{io::DTStdin, riscv::riscv_polyair::RiscvPolyAir};
    use dt_stark::{sumcheck::config::SCStarkGenericConfig, DTCoreOpts, Val};
    use polyair::prover::SCMachineProver as _;

    use dt_core_executor::{DTContext, Program};

    #[cfg(feature = "eth")]
    type CoreSC = dt_stark::koalabear_poseidon2::koala_bear_poseidon2::SCKoalaBearPoseidon2;
    #[cfg(all(feature = "legacy", not(feature = "eth")))]
    type CoreSC = dt_stark::baby_bear_poseidon2::SCBabyBearPoseidon2;

    #[cfg(feature = "eth")]
    const D: usize = 5;
    #[cfg(all(feature = "legacy", not(feature = "eth")))]
    const D: usize = 4;

    type PolyairProver = polyair::prover::SumcheckProver<CoreSC, RiscvPolyAir<Val<CoreSC>>, D>;

    /// A named ELF in a suite.
    type SuiteEntry = (&'static str, &'static [u8]);

    /// `smoke` (default): minimal bounded coverage.
    fn smoke_suite() -> Vec<SuiteEntry> {
        vec![
            ("ALL_INSTRUCTIONS_ELF", dt_core_test_elf::ALL_INSTRUCTIONS_ELF),
            ("FIBONACCI_ELF", test_artifacts::FIBONACCI_ELF),
        ]
    }

    /// `eth-full`: full curated list covering all five chip classes.
    fn eth_full_suite() -> Vec<SuiteEntry> {
        vec![
            // base / control flow / memory
            ("ALL_INSTRUCTIONS_ELF", dt_core_test_elf::ALL_INSTRUCTIONS_ELF),
            ("FIBONACCI_ELF", test_artifacts::FIBONACCI_ELF),
            // hash / syscall
            ("SHA2_ELF", test_artifacts::SHA2_ELF),
            ("KECCAK_PERMUTE_ELF", test_artifacts::KECCAK_PERMUTE_ELF),
            ("SHA_COMPRESS_ELF", test_artifacts::SHA_COMPRESS_ELF),
            ("SHA_EXTEND_ELF", test_artifacts::SHA_EXTEND_ELF),
            // elliptic curve precompiles
            ("ED25519_ELF", test_artifacts::ED25519_ELF),
            ("SECP256K1_MUL_ELF", test_artifacts::SECP256K1_MUL_ELF),
            ("BN254_MUL_ELF", test_artifacts::BN254_MUL_ELF),
            ("BLS12381_MUL_ELF", test_artifacts::BLS12381_MUL_ELF),
            // big-integer precompiles
            ("UINT256_MUL_ELF", test_artifacts::UINT256_MUL_ELF),
            ("U256XU2048_MUL_ELF", test_artifacts::U256XU2048_MUL_ELF),
        ]
    }

    /// `legacy-smoke`: ALL_INSTRUCTIONS + one hash + one ecc.
    fn legacy_smoke_suite() -> Vec<SuiteEntry> {
        vec![
            ("ALL_INSTRUCTIONS_ELF", dt_core_test_elf::ALL_INSTRUCTIONS_ELF),
            ("SHA2_ELF", test_artifacts::SHA2_ELF),
            ("SECP256K1_MUL_ELF", test_artifacts::SECP256K1_MUL_ELF),
        ]
    }

    fn select_suite(name: &str) -> Vec<SuiteEntry> {
        match name {
            "smoke" => smoke_suite(),
            "eth-full" => eth_full_suite(),
            "legacy-smoke" => legacy_smoke_suite(),
            other => {
                eprintln!("unknown suite '{other}', falling back to 'smoke'");
                smoke_suite()
            }
        }
    }

    /// Prove + verify one ELF. Returns true on success; prints and returns false
    /// on any failure (does not panic, so the suite can continue).
    fn prove_one(name: &str, elf: &[u8]) -> bool {
        println!("\n=== [{name}] PolyAir core prove (D={D}) ===");

        let program = match Program::from(elf) {
            Ok(p) => p,
            Err(e) => {
                println!("[FAIL] {name}: failed to load ELF: {e:?}");
                return false;
            }
        };
        let core_machine = RiscvPolyAir::sc_machine(CoreSC::default());
        let core_prover = PolyairProver::new(core_machine);
        let (host_pk, vk) = core_prover.setup(&program);
        let pk = core_prover.pk_to_device(&host_pk);

        let stdin = DTStdin::new();

        let prove_result =
            dt_core_machine::utils::prove_polyair::sc_prove_core::<CoreSC, PolyairProver, D>(
                &core_prover,
                &pk,
                &vk,
                program,
                &stdin,
                DTCoreOpts::default(),
                DTContext::default(),
                None,
            );

        let (proof, _public_values, cycles) = match prove_result {
            Ok(t) => t,
            Err(e) => {
                println!("[FAIL] {name}: prove failed: {e:?}");
                return false;
            }
        };

        println!("  proved: {} shard(s), {} cycles", proof.shard_proofs.len(), cycles);

        let challenger = core_prover.config().mlchallenger();
        let verify_result =
            core_prover.machine().verify(&vk, &proof, &mut challenger.clone(), 1, 0);
        match verify_result {
            Ok(()) => {
                println!("[PASS] {name}: verification passed.");
                true
            }
            Err(e) => {
                println!("[FAIL] {name}: verification failed: {e:?}");
                false
            }
        }
    }

    pub fn run() {
        dt_core_machine::utils::setup_logger();

        let suite_name = std::env::args().nth(1).unwrap_or_else(|| "smoke".to_string());
        let suite = select_suite(&suite_name);

        println!("Running PolyAir prove suite '{suite_name}' ({} ELF(s))...", suite.len());

        let mut failed: Vec<&str> = Vec::new();
        for (name, elf) in &suite {
            if !prove_one(name, elf) {
                failed.push(name);
            }
        }

        println!("\n========== Summary (suite '{suite_name}') ==========");
        println!("ELFs run:    {}", suite.len());
        println!("ELFs passed: {}", suite.len() - failed.len());
        if failed.is_empty() {
            println!("[PASS] all ELFs proved and verified.");
        } else {
            println!("[FAIL] failed ELFs: {failed:?}");
            std::process::exit(1);
        }
    }
}
