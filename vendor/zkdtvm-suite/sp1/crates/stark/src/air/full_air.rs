//! FullAir trait and related types for sumcheck-based AIR constraints.
//!
//! This module provides an alternative AIR abstraction optimized for sumcheck-based
//! proving, where constraint evaluation is separated into distinct phases:
//! - Precompute phase: Compute linear combinations for later use
//! - Permutation phase: Generate permutation traces for lookup arguments
//! - Evaluation phase: Final constraint verification

use p3_matrix::dense::RowMajorMatrix;
use serde::{Deserialize, Serialize};

use super::full_air_builder::FullAirBuilder;
use crate::PROOF_MAX_NUM_PVS;

/// Enum to reference columns from either preprocessed or main trace.
///
/// Used in `FullAir::reserved_poly()` to specify which columns should be
/// kept for later sumcheck rounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PairCol {
    /// Reference to a column in the preprocessed trace.
    Prep(usize),
    /// Reference to a column in the main trace.
    Main(usize),
}

/// A trait for AIRs that support sumcheck-based constraint evaluation.
///
/// Unlike the traditional `Air` trait which uses a single evaluation method,
/// `FullAir` separates constraint logic into three phases:
///
/// 1. **Precompute phase** (`precompute_lc`): Compute and retain linear combinations that will be
///    used in later phases. This allows efficient reuse of computed values.
///
/// 2. **Evaluation phase** (`eval`): Define custom gate constraints using the precomputed values.
///
/// 3. **Lookup phase** (`lookup`): Define lookup argument constraints via `send` and `recv`
///    operations.
///
/// # Type Parameters
///
/// - `AB`: The builder type implementing `FullAirBuilder` for the current phase.
pub trait FullAir<AB: FullAirBuilder>: Send + Sync {
    /// Returns the number of main trace columns for this AIR.
    fn width(&self) -> usize;

    /// Deprecated: use `MachineAir::generate_preprocessed_trace` instead.
    ///
    /// This hook remains only for backward compatibility in `FullAir`.
    #[deprecated(note = "use MachineAir::generate_preprocessed_trace instead")]
    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<AB::F>> {
        panic!("use MachineAir::generate_preprocessed_trace instead")
    }

    /// Returns the number of expected public values.
    ///
    /// Default implementation returns 0.
    fn num_public_values(&self) -> usize {
        PROOF_MAX_NUM_PVS
    }

    /// Returns the maximum power of beta required for lookup denominator computations.
    ///
    /// This is used to determine how many beta powers need to be generated during
    /// the proving phase.
    fn required_max_beta_power(&self) -> usize {
        13
    }

    /// Returns the list of columns to reserve for later sumcheck rounds.
    ///
    /// These columns will be available via `reserved_poly()` in the evaluation phase.
    fn reserved_poly(&self) -> Vec<PairCol>;

    /// Compute and retain linear combinations for later use.
    ///
    /// This method is called during the precompute phase. Implementations should
    /// use `builder.retain_precomputed()` to store values that will be needed
    /// in later phases.
    fn precompute_lc(&self, builder: &mut AB);

    /// Evaluate custom gate constraints.
    ///
    /// This method is called during the evaluation phase. Implementations should
    /// use `builder.assert_zero()` and `builder.assert_zero_ext()` to record
    /// constraints.
    fn eval(&self, builder: &mut AB);

    /// Define lookup argument constraints.
    ///
    /// This method is called during both permutation trace generation and
    /// evaluation phases. Implementations should use `builder.send()` and
    /// `builder.recv()` to record lookup multiplicities.
    fn lookup(&self, builder: &mut AB);

    /// Returns whether this AIR uses global constraints.
    ///
    /// Default implementation returns `false`.
    fn global(&self) -> bool {
        false
    }
}
