use std::marker::PhantomData;

use p3_commit::Mmcs;
use p3_field::AbstractField;
use p3_matrix::dense::RowMajorMatrix;

use super::{
    SCDTCompressVerifier, SCDTCompressWithVKeyVerifier, SCDTCompressWithVKeyWitnessVariable,
    SCDTCompressWitnessVariable,
};
use crate::{
    challenger::DuplexChallengerVariable,
    hash::FieldHasherVariable,
    machine::PublicValuesOutputDigest,
    sumcheck::{
        polyair_folder::RecursivePolyAirConstraintFolder,
        polyair_precompute::RecursivePolyAirPrecomputeRowBuilder, SCBabyBearFriConfigVariable,
    },
    CircuitConfig,
};
use dt_recursion_compiler::{
    circuit::CircuitV2Builder,
    ir::{Builder, Felt},
};
use dt_recursion_core::DIGEST_SIZE;
use dt_stark::{
    air::{FullAir, MachineAir, PolyAirExtendable},
    Challenge, Val,
};
use polyair::SCStarkMachine as PolyAirStarkMachine;

/// A program to verify a single recursive proof representing a complete proof of program execution.
///
/// The root verifier is simply a `DTCompressVerifier` with an assertion that the `is_complete`
/// flag is set to true.
#[derive(Debug, Clone, Copy)]
pub struct SCDTCompressRootVerifier<C, SC, A, const D: usize> {
    _phantom: PhantomData<(C, SC, A)>,
}

/// A program to verify a single recursive proof representing a complete proof of program execution.
///
/// The root verifier is simply a `DTCompressVerifier` with an assertion that the `is_complete`
/// flag is set to true.
#[derive(Debug, Clone, Copy)]
pub struct SCDTCompressRootVerifierWithVKey<C, SC, A, const D: usize> {
    _phantom: PhantomData<(C, SC, A)>,
}

impl<C, SC, A, const D: usize> SCDTCompressRootVerifier<C, SC, A, D>
where
    SC: SCBabyBearFriConfigVariable<C>,
    C: CircuitConfig<F = SC::Val, EF = Challenge<SC>>,
    SC::ValMmcs: Mmcs<SC::Val, ProverData<RowMajorMatrix<SC::Val>>: Clone>,
    A: MachineAir<SC::Val>,
    Val<SC>: PolyAirExtendable<D>,
{
    pub fn verify(
        builder: &mut Builder<C>,
        machine: &PolyAirStarkMachine<SC, A, D>,
        input: SCDTCompressWitnessVariable<C, SC>,
        vk_root: [Felt<C::F>; DIGEST_SIZE],
    ) where
        A: for<'a> FullAir<RecursivePolyAirConstraintFolder<'a, C>>,
        A: for<'a> FullAir<RecursivePolyAirPrecomputeRowBuilder<'a, C>>,
        Builder<C>: CircuitV2Builder<C>,
    {
        // Assert that the program is complete.
        builder.assert_felt_eq(input.is_complete, C::F::one());
        // Verify the proof, as a compress proof.
        SCDTCompressVerifier::verify(
            builder,
            machine,
            input,
            vk_root,
            PublicValuesOutputDigest::Root,
        );
    }
}

impl<C, SC, A, const D: usize> SCDTCompressRootVerifierWithVKey<C, SC, A, D>
where
    SC: SCBabyBearFriConfigVariable<C, FriChallengerVariable = DuplexChallengerVariable<C>>
        + FieldHasherVariable<C, DigestVariable = [Felt<C::F>; DIGEST_SIZE]>,
    C: CircuitConfig<F = SC::Val, EF = Challenge<SC>>,
    SC::ValMmcs: Mmcs<SC::Val, ProverData<RowMajorMatrix<SC::Val>>: Clone>,
    A: MachineAir<SC::Val>,
    Val<SC>: PolyAirExtendable<D>,
    Builder<C>: CircuitV2Builder<C>,
{
    pub fn verify(
        builder: &mut Builder<C>,
        machine: &PolyAirStarkMachine<SC, A, D>,
        input: SCDTCompressWithVKeyWitnessVariable<C, SC>,
        value_assertions: bool,
        kind: PublicValuesOutputDigest,
    ) where
        A: for<'a> FullAir<RecursivePolyAirConstraintFolder<'a, C>>,
        A: for<'a> FullAir<RecursivePolyAirPrecomputeRowBuilder<'a, C>>,
    {
        // Assert that the program is complete.
        builder.assert_felt_eq(input.compress_var.is_complete, C::F::one());
        // Verify the proof, as a compress proof.
        SCDTCompressWithVKeyVerifier::verify(builder, machine, input, value_assertions, kind);
    }
}
