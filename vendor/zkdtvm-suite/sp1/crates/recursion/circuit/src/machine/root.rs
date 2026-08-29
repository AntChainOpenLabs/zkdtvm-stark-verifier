use std::marker::PhantomData;

use p3_air::Air;
use p3_baby_bear::BabyBear;
use p3_commit::Mmcs;
use p3_field::AbstractField;
use p3_matrix::dense::RowMajorMatrix;

use super::{
    DTCompressVerifier, DTCompressWithVKeyVerifier, DTCompressWithVKeyWitnessVariable,
    DTCompressWitnessVariable, PublicValuesOutputDigest,
};
use crate::{
    challenger::DuplexChallengerVariable, constraints::RecursiveVerifierConstraintFolder,
    BabyBearFriConfigVariable, CircuitConfig,
};
use dt_recursion_compiler::ir::{Builder, Felt};
use dt_recursion_core::DIGEST_SIZE;
use dt_stark::{air::MachineAir, Challenge, StarkMachine};

/// A program to verify a single recursive proof representing a complete proof of program execution.
///
/// The root verifier is simply a `DTCompressVerifier` with an assertion that the `is_complete`
/// flag is set to true.
#[derive(Debug, Clone, Copy)]
pub struct DTCompressRootVerifier<C, SC, A> {
    _phantom: PhantomData<(C, SC, A)>,
}

/// A program to verify a single recursive proof representing a complete proof of program execution.
///
/// The root verifier is simply a `DTCompressVerifier` with an assertion that the `is_complete`
/// flag is set to true.
#[derive(Debug, Clone, Copy)]
pub struct DTCompressRootVerifierWithVKey<C, SC, A> {
    _phantom: PhantomData<(C, SC, A)>,
}

impl<C, SC, A> DTCompressRootVerifier<C, SC, A>
where
    SC: BabyBearFriConfigVariable<C>,
    C: CircuitConfig<F = SC::Val, EF = Challenge<SC>>,
    <SC::ValMmcs as Mmcs<BabyBear>>::ProverData<RowMajorMatrix<BabyBear>>: Clone,
    A: MachineAir<SC::Val> + for<'a> Air<RecursiveVerifierConstraintFolder<'a, C>>,
{
    pub fn verify(
        builder: &mut Builder<C>,
        machine: &StarkMachine<SC, A>,
        input: DTCompressWitnessVariable<C, SC>,
        vk_root: [Felt<C::F>; DIGEST_SIZE],
    ) {
        // Assert that the program is complete.
        builder.assert_felt_eq(input.is_complete, C::F::one());
        // Verify the proof, as a compress proof.
        DTCompressVerifier::verify(
            builder,
            machine,
            input,
            vk_root,
            PublicValuesOutputDigest::Root,
        );
    }
}

impl<C, SC, A> DTCompressRootVerifierWithVKey<C, SC, A>
where
    SC: BabyBearFriConfigVariable<
        C,
        FriChallengerVariable = DuplexChallengerVariable<C>,
        DigestVariable = [Felt<BabyBear>; DIGEST_SIZE],
    >,
    C: CircuitConfig<F = SC::Val, EF = Challenge<SC>, Bit = Felt<BabyBear>>,
    <SC::ValMmcs as Mmcs<BabyBear>>::ProverData<RowMajorMatrix<BabyBear>>: Clone,
    A: MachineAir<SC::Val> + for<'a> Air<RecursiveVerifierConstraintFolder<'a, C>>,
{
    pub fn verify(
        builder: &mut Builder<C>,
        machine: &StarkMachine<SC, A>,
        input: DTCompressWithVKeyWitnessVariable<C, SC>,
        value_assertions: bool,
        kind: PublicValuesOutputDigest,
    ) {
        // Assert that the program is complete.
        builder.assert_felt_eq(input.compress_var.is_complete, C::F::one());
        // Verify the proof, as a compress proof.
        DTCompressWithVKeyVerifier::verify(builder, machine, input, value_assertions, kind);
    }
}
