use dt_core_machine::shape::{chip_log_height_threshold, num_skip_rounds};
use std::{borrow::Borrow, marker::PhantomData};

use dt_recursion_compiler::{
    circuit::CircuitV2Builder,
    ir::{Builder, Felt},
};
use dt_stark::{air::MachineAir, Challenge, SCStarkMachine, Val};
use p3_air::Air;
use p3_commit::Mmcs;
use p3_field::AbstractField;
use p3_matrix::dense::RowMajorMatrix;

use super::{
    compress::SCDTCompressWitnessVariable, public_values::sc_assert_root_public_values_valid,
};
use crate::{
    challenger::CanObserveVariable,
    constraints::RecursiveSumcheckConstraintFolder,
    machine::{assert_complete, RootPublicValues},
    sumcheck::{verifier::SumcheckVerifier, SCBabyBearFriConfigVariable},
    CircuitConfig,
};

/// A program that recursively verifies a proof made by [super::DTRootVerifier].
#[derive(Debug, Clone, Copy)]
pub struct SCDTWrapVerifier<C, SC, A, AE> {
    _phantom: PhantomData<(C, SC, A, AE)>,
}

impl<C, SC, A, AE> SCDTWrapVerifier<C, SC, A, AE>
where
    SC: SCBabyBearFriConfigVariable<C>,
    C: CircuitConfig<F = SC::Val, EF = Challenge<SC>>,
    SC::ValMmcs: Mmcs<Val<SC>, ProverData<RowMajorMatrix<Val<SC>>>: Clone>,
    A: MachineAir<SC::Val> + for<'a> Air<RecursiveSumcheckConstraintFolder<'a, C>>,
    AE: MachineAir<Challenge<SC>>,
    Builder<C>: CircuitV2Builder<C>,
{
    /// Verify a batch of recursive proofs and aggregate their public values.
    ///
    /// The compression verifier can aggregate proofs of different kinds:
    /// - Core proofs: proofs which are recursive proof of a batch of zkDTVM shard proofs. The
    ///   implementation in this function assumes a fixed recursive verifier specified by
    ///   `recursive_vk`.
    /// - Deferred proofs: proofs which are recursive proof of a batch of deferred proofs. The
    ///   implementation in this function assumes a fixed deferred verification program specified by
    ///   `deferred_vk`.
    /// - Compress proofs: these are proofs which refer to a prove of this program. The key for it
    ///   is part of public values will be propagated across all levels of recursion and will be
    ///   checked against itself as in the prover or as in [super::DTRootVerifier].
    pub fn verify(
        builder: &mut Builder<C>,
        machine: &SCStarkMachine<SC, A, AE>,
        input: SCDTCompressWitnessVariable<C, SC>,
    ) {
        // Read input.
        let SCDTCompressWitnessVariable { vks_and_proofs, is_complete } = input;

        // Assert that there is only one proof, and get the verification key and proof.
        let [(vk, proof)] = vks_and_proofs.try_into().ok().unwrap();

        // Verify the stark proof.

        // Prepare a challenger.
        let mut challenger = machine.config().challenger_variable(builder);

        vk.observe_into(builder, &mut challenger);

        // Observe the public values.
        challenger
            .observe_slice(builder, proof.public_values[0..machine.num_pv_elts()].iter().copied());

        SumcheckVerifier::verify_shard(
            builder,
            &vk,
            machine,
            &mut challenger,
            &proof,
            num_skip_rounds(),
            chip_log_height_threshold(),
        );

        // Get the public values, and assert that they are valid.
        let public_values: &RootPublicValues<Felt<C::F>> = proof.public_values.as_slice().borrow();
        sc_assert_root_public_values_valid::<C, SC>(builder, public_values);

        // Assert the public values are of a complete proof.
        assert_complete(builder, &public_values.inner, is_complete);
        builder.assert_felt_eq(is_complete, C::F::one());
        // Reflect the public values to the next level.
        SC::commit_recursion_public_values(builder, public_values.inner);
    }
}
