#![allow(missing_docs)]
use p3_challenger::{CanObserve, CanSample, FieldChallenger};
use p3_commit::{Pcs, PolynomialSpace};
use p3_field::{
    extension::BinomiallyExtendable, packed_ext_generic::PackedExtensionField, AbstractField,
    ExtensionField, Field, PrimeField32, TwoAdicField,
};
use serde::{de::DeserializeOwned, Serialize};

pub type Domain<SC> = <<SC as StarkGenericConfig>::Pcs as Pcs<
    Challenge<SC>,
    <SC as StarkGenericConfig>::Challenger,
>>::Domain;

pub type Val<SC> = <SC as StarkGenericConfig>::Val;

pub type Dom<SC> = <<SC as StarkGenericConfig>::Pcs as Pcs<
    Challenge<SC>,
    <SC as StarkGenericConfig>::Challenger,
>>::Domain;

pub type PackedVal<SC> = <Val<SC> as Field>::Packing;

/// The active Fiat-Shamir challenge extension field for this configuration.
pub type Challenge<SC> = <SC as StarkGenericConfig>::Challenge;

/// The packed (SIMD) form of the active challenge extension field.
pub type PackedChallenge<SC> = <Challenge<SC> as ExtensionField<Val<SC>>>::ExtensionPacking;

/// Packed extension wrapper used by the sumcheck extension folder, which treats
/// the (scalar) challenge extension as its `AirBuilder::F` and therefore needs a
/// packed type supporting cross-scalar-extension arithmetic. The raw
/// `PackedChallenge<SC>` cannot carry those ops generically, so we wrap it.
pub type PackedExt<SC> = PackedExtensionField<Val<SC>, Challenge<SC>>;

pub type Com<SC> = <<SC as StarkGenericConfig>::Pcs as Pcs<
    Challenge<SC>,
    <SC as StarkGenericConfig>::Challenger,
>>::Commitment;

pub type OpeningProof<SC> = <<SC as StarkGenericConfig>::Pcs as Pcs<
    Challenge<SC>,
    <SC as StarkGenericConfig>::Challenger,
>>::Proof;

pub type OpeningError<SC> = <<SC as StarkGenericConfig>::Pcs as Pcs<
    Challenge<SC>,
    <SC as StarkGenericConfig>::Challenger,
>>::Error;

pub type PcsProverData<SC> = <<SC as StarkGenericConfig>::Pcs as Pcs<
    Challenge<SC>,
    <SC as StarkGenericConfig>::Challenger,
>>::ProverData;

pub type Challenger<SC> = <SC as StarkGenericConfig>::Challenger;

pub trait StarkGenericConfig: 'static + Send + Sync + Serialize + DeserializeOwned + Clone {
    // NOTE: `BinomiallyExtendable<4>` is retained on `Val` (not the challenge): both
    // BabyBear and KoalaBear implement it, and downstream recursion/core code still
    // constructs `BinomialExtensionField<Val, 4>` (e.g. recursion AIR field, quartic
    // wrap-facing configs) independently of the active challenge. The quintic
    // challenge is modeled separately via the `Challenge` associated type, not via a
    // binomial-5 bound on `Val`.
    type Val: PrimeField32 + TwoAdicField + AbstractField + BinomiallyExtendable<4>;

    /// The active Fiat-Shamir challenge extension field.
    type Challenge: ExtensionField<Self::Val> + TwoAdicField;

    type Domain: PolynomialSpace<Val = Self::Val> + Sync;

    /// The PCS used to commit to trace polynomials.
    type Pcs: Pcs<Self::Challenge, Self::Challenger, Domain = Self::Domain>
        + Sync
        + ZeroCommitment<Self>;

    /// The challenger (Fiat-Shamir) implementation used.
    type Challenger: FieldChallenger<Val<Self>>
        + CanObserve<<Self::Pcs as Pcs<Self::Challenge, Self::Challenger>>::Commitment>
        + CanSample<Self::Challenge>
        + Serialize
        + DeserializeOwned;

    /// Get the PCS used by this configuration.
    fn pcs(&self) -> &Self::Pcs;

    /// Initialize a new challenger.
    fn challenger(&self) -> Self::Challenger;
}

pub trait ZeroCommitment<SC: StarkGenericConfig> {
    fn zero_commitment(&self) -> Com<SC>;
}

pub struct UniConfig<SC>(pub SC);

impl<SC: StarkGenericConfig> p3_uni_stark::StarkGenericConfig for UniConfig<SC> {
    type Pcs = SC::Pcs;

    type Challenge = Challenge<SC>;

    type Challenger = SC::Challenger;

    fn pcs(&self) -> &Self::Pcs {
        self.0.pcs()
    }
}
