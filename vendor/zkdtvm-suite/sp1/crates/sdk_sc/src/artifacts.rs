//! # zkDTVM Artifacts
//!
//! A library for exporting the zkDTVM artifacts to the specified output directory.

#[cfg(not(feature = "ext5"))]
use std::path::PathBuf;

#[cfg(not(feature = "ext5"))]
use anyhow::{Context, Result};

#[cfg(not(feature = "ext5"))]
use crate::install::try_install_circuit_artifacts;
#[cfg(not(feature = "ext5"))]
pub use dt_prover::build::build_plonk_bn254_artifacts_with_dummy;

/// Exports the solidity verifier for PLONK proofs to the specified output directory.
///
/// WARNING: If you are on development mode, this function assumes that the PLONK artifacts have
/// already been built.
#[cfg(not(feature = "ext5"))]
pub fn export_solidity_plonk_bn254_verifier(output_dir: impl Into<PathBuf>) -> Result<()> {
    let output_dir: PathBuf = output_dir.into();
    let artifacts_dir = if dt_prover::build::dt_dev_mode() {
        dt_prover::build::plonk_bn254_artifacts_dev_dir()
    } else {
        try_install_circuit_artifacts("plonk")
    };
    let verifier_path = artifacts_dir.join("DTVerifierPlonk.sol");

    if !verifier_path.exists() {
        return Err(anyhow::anyhow!("verifier file not found at {:?}", verifier_path));
    }

    std::fs::create_dir_all(&output_dir).context("Failed to create output directory.")?;
    let output_path = output_dir.join("DTVerifierPlonk.sol");
    std::fs::copy(&verifier_path, &output_path).context("Failed to copy verifier file.")?;
    tracing::info!(
        "exported verifier from {} to {}",
        verifier_path.display(),
        output_path.display()
    );

    Ok(())
}

/// Exports the solidity verifier for Groth16 proofs to the specified output directory.
///
/// WARNING: If you are on development mode, this function assumes that the Groth16 artifacts have
/// already been built.
#[cfg(not(feature = "ext5"))]
pub fn export_solidity_groth16_bn254_verifier(output_dir: impl Into<PathBuf>) -> Result<()> {
    let output_dir: PathBuf = output_dir.into();
    let artifacts_dir = if dt_prover::build::dt_dev_mode() {
        dt_prover::build::groth16_bn254_artifacts_dev_dir()
    } else {
        try_install_circuit_artifacts("groth16")
    };
    let verifier_path = artifacts_dir.join("DTVerifierGroth16.sol");

    if !verifier_path.exists() {
        return Err(anyhow::anyhow!("verifier file not found at {:?}", verifier_path));
    }

    std::fs::create_dir_all(&output_dir).context("Failed to create output directory.")?;
    let output_path = output_dir.join("DTVerifierGroth16.sol");
    std::fs::copy(&verifier_path, &output_path).context("Failed to copy verifier file.")?;
    tracing::info!(
        "exported verifier from {} to {}",
        verifier_path.display(),
        output_path.display()
    );

    Ok(())
}
