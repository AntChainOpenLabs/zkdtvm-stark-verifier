use dt_primitives::SCField;
use dt_stark::{InnerChallenge, InnerVal};
use p3_baby_bear::BabyBear;
use p3_bn254_fr::Bn254Fr;
use p3_field::extension::BinomialExtensionField;
#[cfg(feature = "ext5")]
use p3_field::extension::QuinticTrinomialExtensionField;

use crate::{circuit::AsmConfig, prelude::Config};

pub type InnerConfig = AsmConfig<InnerVal, InnerChallenge>;

/// The active SC (sumcheck-prover) challenge extension over `SCField`.
/// Quintic trinomial under feature `ext5`, binomial quartic otherwise.
#[cfg(not(feature = "ext5"))]
pub type SCChallenge = BinomialExtensionField<SCField, 4>;
#[cfg(feature = "ext5")]
pub type SCChallenge = QuinticTrinomialExtensionField<SCField>;

/// KoalaBear version of InnerConfig for SC Prover.
/// Uses KoalaBear field (SCField) instead of BabyBear for the recursion compiler.
pub type SCInnerConfig = AsmConfig<SCField, SCChallenge>;

/// Shrink-stage `Config`.
///
/// A distinct nominal newtype whose math types mirror `SCInnerConfig`
/// (KoalaBear path) exactly. The point is to allow `CircuitConfig` /
/// `exp_reverse_bits_ext` to be specialised for the shrink stage without
/// affecting compress / wrap, so we can inline `x^e` purely with ExtAlu ops
/// and drop `ExtExpReverseBitsChip` from `sc_shrink_machine`.
///
/// Wrapped as a newtype to satisfy Rust's trait coherence rules.
#[derive(Clone, Default, Debug)]
pub struct ShrinkConfig;

impl Config for ShrinkConfig {
    type N = SCField;
    type F = SCField;
    type EF = SCChallenge;
}

#[derive(Clone, Default, Debug)]
pub struct OuterConfig;

impl Config for OuterConfig {
    type N = Bn254Fr;
    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;
}

/// KoalaBear version of OuterConfig for SC Prover.
/// Uses KoalaBear field (SCField) with Bn254Fr as the native field for Gnark constraint generation.
#[derive(Clone, Default, Debug)]
pub struct SCOuterConfig;

impl Config for SCOuterConfig {
    type N = Bn254Fr;
    type F = SCField;
    type EF = BinomialExtensionField<SCField, 4>;
}
