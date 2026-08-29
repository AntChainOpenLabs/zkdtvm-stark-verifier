//! # zkDTVM SDK
//!
//! A library for interacting with the zkDTVM RISC-V zkVM.
//!
//! Visit the [Getting Started](https://docs.succinct.xyz/docs/sp1/getting-started/install) section
//! in the official zkDTVM documentation for a quick start guide.

#![warn(clippy::pedantic)]
#![allow(clippy::similar_names)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::bool_to_int_with_if)]
#![allow(clippy::should_panic_without_expect)]
#![allow(clippy::field_reassign_with_default)]
#![allow(clippy::manual_assert)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::match_wildcard_for_single_variants)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::explicit_iter_loop)]

pub mod artifacts;
pub mod client;
pub mod cpu;
// pub mod cuda;
pub mod env;
pub mod install;
#[cfg(feature = "network")]
pub mod network;
pub mod utils;

// Re-export the client.
pub use crate::client::ProverClient;

// Re-export the provers.
pub use crate::{cpu::CpuProver, env::EnvProver};

#[cfg(feature = "network")]
pub use crate::network::prover::NetworkProver;

// Re-export the proof and prover traits.
pub mod proof;
pub use proof::*;
pub mod prover;

pub use prover::{DTVerificationError, Prover};

// Re-export the build utilities and executor primitives.
pub use dt_build::include_elf;
pub use dt_core_executor::{DTContext, DTContextBuilder, ExecutionReport, Executor, HookEnv};

// Re-export the machine/prover primitives.
pub use dt_core_machine::io::DTStdin;
pub use dt_primitives::io::DTPublicValues;
pub use dt_prover::{
    DTProver, DTProvingKey, DTVerifyingKey, HashableKey, ProverMode, DT_CIRCUIT_VERSION,
};
pub use dt_stark::RecursionBackend;

// Re-export the utilities.
pub use utils::setup_logger;

#[cfg(test)]
mod tests {
    use dt_primitives::io::DTPublicValues;

    use crate::{utils, DTStdin, Prover, ProverClient};

    #[test]
    fn test_execute() {
        utils::setup_logger();
        let client = ProverClient::builder().cpu().build();
        let elf = test_artifacts::FIBONACCI_ELF;
        let mut stdin = DTStdin::new();
        stdin.write(&10usize);
        let (_, _) = client.execute(elf, &stdin).run().unwrap();
    }

    #[test]
    #[should_panic]
    fn test_execute_panic() {
        utils::setup_logger();
        let client = ProverClient::builder().cpu().build();
        let elf = test_artifacts::PANIC_ELF;
        let mut stdin = DTStdin::new();
        stdin.write(&10usize);
        client.execute(elf, &stdin).run().unwrap();
    }

    #[should_panic]
    #[test]
    fn test_cycle_limit_fail() {
        utils::setup_logger();
        let client = ProverClient::builder().cpu().build();
        let elf = test_artifacts::PANIC_ELF;
        let mut stdin = DTStdin::new();
        stdin.write(&10usize);
        client.execute(elf, &stdin).cycle_limit(1).run().unwrap();
    }

    #[test]
    fn test_e2e_core() {
        utils::setup_logger();
        let client = ProverClient::builder().cpu().build();
        let elf = test_artifacts::FIBONACCI_ELF;
        let (pk, vk) = client.setup(elf);
        let mut stdin = DTStdin::new();
        stdin.write(&10usize);

        // Generate proof & verify.
        let mut proof = client.prove(&pk, &stdin).run().unwrap();
        client.verify(&proof, &vk).unwrap();

        // Test invalid public values.
        proof.public_values = DTPublicValues::from(&[255, 4, 84]);
        if client.verify(&proof, &vk).is_ok() {
            panic!("verified proof with invalid public values")
        }
    }

    #[test]
    fn test_e2e_io_override() {
        utils::setup_logger();
        let client = ProverClient::builder().cpu().build();
        let elf = test_artifacts::HELLO_WORLD_ELF;

        let mut stdout = Vec::new();

        // Generate proof & verify.
        let stdin = DTStdin::new();
        let _ = client.execute(elf, &stdin).stdout(&mut stdout).run().unwrap();

        assert_eq!(stdout, b"Hello, world!\n");
    }

    #[test]
    fn test_e2e_compressed() {
        utils::setup_logger();
        let client = ProverClient::builder().cpu().build();
        let elf = test_artifacts::FIBONACCI_ELF;
        let (pk, vk) = client.setup(elf);
        let mut stdin = DTStdin::new();
        // The production-shaped input (22 full shards at shard_size 2^16): the same
        // fixture shape both recursion backends were proven on. The old input (n=10,
        // one near-empty shard at the default shard size) is a degenerate shape the
        // DSL lift-program compiler cannot build a circuit for (division by a
        // vanishing denominator at compile time).
        stdin.write(&100_000usize);

        // Generate proof & verify.
        let mut proof = client.prove(&pk, &stdin).compressed().shard_size(1 << 16).run().unwrap();
        client.verify(&proof, &vk).unwrap();

        // Test invalid public values.
        proof.public_values = DTPublicValues::from(&[255, 4, 84]);
        if client.verify(&proof, &vk).is_ok() {
            panic!("verified proof with invalid public values")
        }
    }

    #[test]
    fn test_e2e_prove_plonk() {
        utils::setup_logger();
        let client = ProverClient::builder().cpu().build();
        let elf = test_artifacts::FIBONACCI_ELF;
        let (pk, vk) = client.setup(elf);
        let mut stdin = DTStdin::new();
        stdin.write(&10usize);

        // Generate proof & verify.
        let mut proof = client.prove(&pk, &stdin).plonk().run().unwrap();
        client.verify(&proof, &vk).unwrap();

        // Test invalid public values.
        proof.public_values = DTPublicValues::from(&[255, 4, 84]);
        if client.verify(&proof, &vk).is_ok() {
            panic!("verified proof with invalid public values")
        }
    }

    #[test]
    fn test_e2e_prove_plonk_mock() {
        utils::setup_logger();
        let client = ProverClient::builder().mock().build();
        let elf = test_artifacts::FIBONACCI_ELF;
        let (pk, vk) = client.setup(elf);
        let mut stdin = DTStdin::new();
        stdin.write(&10usize);
        let proof = client.prove(&pk, &stdin).plonk().run().unwrap();
        client.verify(&proof, &vk).unwrap();
    }
}

#[cfg(all(feature = "cuda", not(dt_ci_in_progress)))]
mod deprecated_check {
    #[deprecated(
        since = "4.0.0",
        note = "The `cuda` feature is deprecated, as the CudaProver is now supported by default."
    )]
    #[allow(unused)]
    fn cuda_is_deprecated() {}

    /// Show a warning if the `cuda` feature is enabled.
    #[allow(unused, deprecated)]
    fn show_cuda_warning() {
        cuda_is_deprecated();
    }
}
