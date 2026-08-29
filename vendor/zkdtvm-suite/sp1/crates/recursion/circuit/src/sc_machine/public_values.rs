use dt_recursion_compiler::ir::{Builder, Felt};
use dt_recursion_core::{
    air::{RecursionPublicValues, NUM_PV_ELMS_TO_HASH},
    DIGEST_SIZE,
};
use itertools::Itertools;

use crate::{hash::Poseidon2HasherVariable, machine::RootPublicValues, CircuitConfig};

pub(crate) fn sc_assert_recursion_public_values_valid<C, H>(
    builder: &mut Builder<C>,
    public_values: &RecursionPublicValues<Felt<C::F>>,
) where
    C: CircuitConfig,
    H: Poseidon2HasherVariable<C>,
{
    let digest = sc_recursion_public_values_digest::<C, H>(builder, public_values);
    for (value, expected) in public_values.digest.iter().copied().zip_eq(digest) {
        builder.assert_felt_eq(value, expected);
    }
}

pub(crate) fn sc_recursion_public_values_digest<C, H>(
    builder: &mut Builder<C>,
    public_values: &RecursionPublicValues<Felt<C::F>>,
) -> [Felt<C::F>; DIGEST_SIZE]
where
    C: CircuitConfig,
    H: Poseidon2HasherVariable<C>,
{
    let pv_slice = public_values.as_array();
    H::poseidon2_hash(builder, &pv_slice[..NUM_PV_ELMS_TO_HASH])
}

pub(crate) fn sc_assert_root_public_values_valid<C, H>(
    builder: &mut Builder<C>,
    public_values: &RootPublicValues<Felt<C::F>>,
) where
    C: CircuitConfig,
    H: Poseidon2HasherVariable<C>,
{
    let expected_digest = sc_root_public_values_digest::<C, H>(builder, &public_values.inner);
    for (value, expected) in public_values.inner.digest.iter().copied().zip_eq(expected_digest) {
        builder.assert_felt_eq(value, expected);
    }
}

pub(crate) fn sc_root_public_values_digest<C, H>(
    builder: &mut Builder<C>,
    public_values: &RecursionPublicValues<Felt<C::F>>,
) -> [Felt<C::F>; DIGEST_SIZE]
where
    C: CircuitConfig,
    H: Poseidon2HasherVariable<C>,
{
    let input = public_values
        .dt_vk_digest
        .into_iter()
        .chain(public_values.committed_value_digest.into_iter().flat_map(|word| word.0.into_iter()))
        .collect::<Vec<_>>();
    H::poseidon2_hash(builder, &input)
}
