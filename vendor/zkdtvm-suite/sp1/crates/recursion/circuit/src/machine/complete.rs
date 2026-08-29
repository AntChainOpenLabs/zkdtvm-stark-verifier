use dt_recursion_compiler::{
    circuit::CircuitV2Builder,
    ir::{Builder, Config, Felt},
};
use dt_recursion_core::air::RecursionPublicValues;
use itertools::Itertools;
use p3_field::AbstractField;

/// Assertions on recursion public values which represent a complete proof.
///
/// The assertions consist of checking all the expected boundary conditions from a compress proof
/// that represents the end of the recursion tower.
pub(crate) fn assert_complete<C: Config>(
    builder: &mut Builder<C>,
    public_values: &RecursionPublicValues<Felt<C::F>>,
    is_complete: Felt<C::F>,
) where
    Builder<C>: CircuitV2Builder<C>,
{
    assert_complete_without_global(builder, public_values, is_complete);

    for limb in public_values.global_interval_end[0]
        .iter()
        .chain(public_values.global_interval_end[2].iter())
    {
        builder.assert_felt_eq(is_complete * *limb, C::F::zero());
    }
    for (limb, value) in public_values.global_interval_end[1].iter().enumerate() {
        let expected = if limb == 0 { C::F::one() } else { C::F::zero() };
        builder.assert_felt_eq(is_complete * (*value - expected), C::F::zero());
    }
}

pub(crate) fn assert_complete_without_global<C: Config>(
    builder: &mut Builder<C>,
    public_values: &RecursionPublicValues<Felt<C::F>>,
    is_complete: Felt<C::F>,
) where
    Builder<C>: CircuitV2Builder<C>,
{
    let RecursionPublicValues {
        deferred_proofs_digest,
        next_pc,
        start_shard,
        next_shard,
        start_execution_shard,
        start_reconstruct_deferred_digest,
        end_reconstruct_deferred_digest,
        contains_execution_shard,
        ..
    } = public_values;

    // Assert that the `is_complete` flag is boolean.
    builder.assert_felt_eq(is_complete * (is_complete - C::F::one()), C::F::zero());

    // Assert that `next_pc` is equal to zero (so program execution has completed)
    builder.assert_felt_eq(is_complete * *next_pc, C::F::zero());

    // Assert that start shard is equal to 1.
    builder.assert_felt_eq(is_complete * (*start_shard - C::F::one()), C::F::zero());

    // Assert that the next shard is not equal to one. This guarantees that there is at least one
    // shard that contains CPU.
    builder.assert_felt_ne(is_complete * *next_shard, C::F::one());

    // Assert that that an execution shard is present.
    builder.assert_felt_eq(is_complete * (*contains_execution_shard - C::F::one()), C::F::zero());
    // Assert that the start execution shard is equal to 1.
    builder.assert_felt_eq(is_complete * (*start_execution_shard - C::F::one()), C::F::zero());

    // The start reconstruct deferred digest should be zero.
    for start_digest_word in start_reconstruct_deferred_digest {
        builder.assert_felt_eq(is_complete * *start_digest_word, C::F::zero());
    }

    // The end reconstruct deferred digest should be equal to the deferred proofs digest.
    debug_assert_eq!(end_reconstruct_deferred_digest.len(), deferred_proofs_digest.len());
    for (end_digest_word, deferred_digest_word) in
        end_reconstruct_deferred_digest.iter().zip_eq(deferred_proofs_digest.iter())
    {
        builder
            .assert_felt_eq(is_complete * (*end_digest_word - *deferred_digest_word), C::F::zero());
    }
}
