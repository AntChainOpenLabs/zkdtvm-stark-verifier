use std::{
    fmt::Debug,
    iter::{repeat, zip},
};

use itertools::Itertools;
use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, Field};

#[cfg(feature = "koalabear")]
use dt_primitives::SCField;
use dt_recursion_compiler::{
    circuit::CircuitV2Builder,
    ir::{Builder, Config, DslIr, Felt, Var},
};
use dt_recursion_core::{
    stark::{
        outer_perm, BabyBearPoseidon2Outer, SCBabyBearPoseidon2Outer,
        OUTER_MULTI_FIELD_CHALLENGER_WIDTH,
    },
    DIGEST_SIZE, HASH_RATE, PERMUTATION_WIDTH,
};
use dt_stark::{baby_bear_poseidon2::BabyBearPoseidon2, inner_perm};
use p3_bn254_fr::Bn254Fr;
use p3_symmetric::Permutation;

use crate::{
    challenger::{reduce_32, POSEIDON_2_BB_RATE},
    CircuitConfig,
};

pub trait FieldHasher<F: Field> {
    type Digest: Copy + Default + Eq + Ord + Copy + Debug + Send + Sync;

    fn constant_compress(input: [Self::Digest; 2]) -> Self::Digest;
}

pub trait Poseidon2KoalaBearHasherVariable<C: CircuitConfig> {
    fn poseidon2_permute(
        builder: &mut Builder<C>,
        state: [Felt<C::F>; PERMUTATION_WIDTH],
    ) -> [Felt<C::F>; PERMUTATION_WIDTH];

    /// Applies the Poseidon2 hash function to the given array.
    ///
    /// Reference: [p3_symmetric::PaddingFreeSponge]
    fn poseidon2_hash(builder: &mut Builder<C>, input: &[Felt<C::F>]) -> [Felt<C::F>; DIGEST_SIZE] {
        // static_assert(RATE < WIDTH)
        let mut state = core::array::from_fn(|_| builder.eval(C::F::zero()));
        for input_chunk in input.chunks(HASH_RATE) {
            state[..input_chunk.len()].copy_from_slice(input_chunk);
            state = Self::poseidon2_permute(builder, state);
        }
        let digest: [Felt<C::F>; DIGEST_SIZE] = state[..DIGEST_SIZE].try_into().unwrap();
        digest
    }
}

pub trait Posedion2BabyBearHasherVariable<C: CircuitConfig> {
    fn poseidon2_permute(
        builder: &mut Builder<C>,
        state: [Felt<C::F>; PERMUTATION_WIDTH],
    ) -> [Felt<C::F>; PERMUTATION_WIDTH];

    /// Applies the Poseidon2 hash function to the given array.
    ///
    /// Reference: [p3_symmetric::PaddingFreeSponge]
    fn poseidon2_hash(builder: &mut Builder<C>, input: &[Felt<C::F>]) -> [Felt<C::F>; DIGEST_SIZE] {
        // static_assert(RATE < WIDTH)
        let mut state = core::array::from_fn(|_| builder.eval(C::F::zero()));
        for input_chunk in input.chunks(HASH_RATE) {
            state[..input_chunk.len()].copy_from_slice(input_chunk);
            state = Self::poseidon2_permute(builder, state);
        }
        let digest: [Felt<C::F>; DIGEST_SIZE] = state[..DIGEST_SIZE].try_into().unwrap();
        digest
    }
}

#[cfg(feature = "babybear")]
pub trait Poseidon2HasherVariable<C: CircuitConfig>: Posedion2BabyBearHasherVariable<C> {}
#[cfg(feature = "babybear")]
impl<C: CircuitConfig, T: Posedion2BabyBearHasherVariable<C>> Poseidon2HasherVariable<C> for T {}

#[cfg(feature = "koalabear")]
pub trait Poseidon2HasherVariable<C: CircuitConfig>: Poseidon2KoalaBearHasherVariable<C> {}
#[cfg(feature = "koalabear")]
impl<C: CircuitConfig, T: Poseidon2KoalaBearHasherVariable<C>> Poseidon2HasherVariable<C> for T {}

pub trait FieldHasherVariable<C: CircuitConfig>: FieldHasher<C::F> {
    type DigestVariable: Clone + Copy + Debug;

    fn hash(builder: &mut Builder<C>, input: &[Felt<C::F>]) -> Self::DigestVariable;

    fn compress(builder: &mut Builder<C>, input: [Self::DigestVariable; 2])
        -> Self::DigestVariable;

    fn assert_digest_eq(builder: &mut Builder<C>, a: Self::DigestVariable, b: Self::DigestVariable);

    // Encountered many issues trying to make the following two parametrically polymorphic.
    fn select_chain_digest(
        builder: &mut Builder<C>,
        should_swap: C::Bit,
        input: [Self::DigestVariable; 2],
    ) -> [Self::DigestVariable; 2];

    fn print_digest(builder: &mut Builder<C>, digest: Self::DigestVariable);
}

impl FieldHasher<BabyBear> for BabyBearPoseidon2 {
    type Digest = [BabyBear; DIGEST_SIZE];

    fn constant_compress(input: [Self::Digest; 2]) -> Self::Digest {
        let mut pre_iter = input.into_iter().flatten().chain(repeat(BabyBear::zero()));
        let mut pre = core::array::from_fn(move |_| pre_iter.next().unwrap());
        (inner_perm()).permute_mut(&mut pre);
        pre[..DIGEST_SIZE].try_into().unwrap()
    }
}

impl<C: CircuitConfig<F = BabyBear>> Posedion2BabyBearHasherVariable<C> for BabyBearPoseidon2 {
    fn poseidon2_permute(
        builder: &mut Builder<C>,
        input: [Felt<<C>::F>; PERMUTATION_WIDTH],
    ) -> [Felt<<C>::F>; PERMUTATION_WIDTH] {
        builder.poseidon2_permute_v2(input)
    }
}

impl<C: CircuitConfig> Posedion2BabyBearHasherVariable<C> for BabyBearPoseidon2Outer {
    fn poseidon2_permute(
        builder: &mut Builder<C>,
        state: [Felt<<C>::F>; PERMUTATION_WIDTH],
    ) -> [Felt<<C>::F>; PERMUTATION_WIDTH] {
        let state: [Felt<_>; PERMUTATION_WIDTH] = state.map(|x| builder.eval(x));
        builder.push_op(DslIr::CircuitPoseidon2PermuteBabyBear(Box::new(state)));
        state
    }
}
#[cfg(feature = "babybear")]
impl<C: CircuitConfig> Posedion2BabyBearHasherVariable<C> for SCBabyBearPoseidon2Outer {
    fn poseidon2_permute(
        builder: &mut Builder<C>,
        state: [Felt<<C>::F>; PERMUTATION_WIDTH],
    ) -> [Felt<<C>::F>; PERMUTATION_WIDTH] {
        let state: [Felt<_>; PERMUTATION_WIDTH] = state.map(|x| builder.eval(x));
        builder.push_op(DslIr::CircuitPoseidon2PermuteBabyBear(Box::new(state)));
        state
    }
}
#[cfg(feature = "koalabear")]
impl<C: CircuitConfig> Poseidon2KoalaBearHasherVariable<C>
    for dt_recursion_core::stark::SCKoalaBearPoseidon2Outer
{
    fn poseidon2_permute(
        builder: &mut Builder<C>,
        state: [Felt<<C>::F>; PERMUTATION_WIDTH],
    ) -> [Felt<<C>::F>; PERMUTATION_WIDTH] {
        let state: [Felt<_>; PERMUTATION_WIDTH] = state.map(|x| builder.eval(x));
        builder.push_op(DslIr::CircuitPoseidon2PermuteKoalaBear(Box::new(state)));
        state
    }
}

impl<C: CircuitConfig<F = BabyBear, Bit = Felt<BabyBear>>> FieldHasherVariable<C>
    for BabyBearPoseidon2
{
    type DigestVariable = [Felt<BabyBear>; DIGEST_SIZE];

    fn hash(builder: &mut Builder<C>, input: &[Felt<<C as Config>::F>]) -> Self::DigestVariable {
        <Self as Posedion2BabyBearHasherVariable<C>>::poseidon2_hash(builder, input)
    }

    fn compress(
        builder: &mut Builder<C>,
        input: [Self::DigestVariable; 2],
    ) -> Self::DigestVariable {
        builder.poseidon2_compress_v2(input.into_iter().flatten())
    }

    fn assert_digest_eq(
        builder: &mut Builder<C>,
        a: Self::DigestVariable,
        b: Self::DigestVariable,
    ) {
        // Push the instruction directly instead of passing through `assert_felt_eq` in order to
        //avoid symbolic expression overhead.
        zip(a, b).for_each(|(e1, e2)| builder.push_op(DslIr::AssertEqF(e1, e2)));
    }

    fn select_chain_digest(
        builder: &mut Builder<C>,
        should_swap: <C as CircuitConfig>::Bit,
        input: [Self::DigestVariable; 2],
    ) -> [Self::DigestVariable; 2] {
        let result0: [Felt<BabyBear>; DIGEST_SIZE] = core::array::from_fn(|_| builder.uninit());
        let result1: [Felt<BabyBear>; DIGEST_SIZE] = core::array::from_fn(|_| builder.uninit());

        (0..DIGEST_SIZE).for_each(|i| {
            builder.push_op(DslIr::Select(
                should_swap,
                result0[i],
                result1[i],
                input[0][i],
                input[1][i],
            ));
        });

        [result0, result1]
    }

    fn print_digest(builder: &mut Builder<C>, digest: Self::DigestVariable) {
        for d in digest.iter() {
            builder.print_f(*d);
        }
    }
}

pub const BN254_DIGEST_SIZE: usize = 1;

impl FieldHasher<BabyBear> for BabyBearPoseidon2Outer {
    type Digest = [Bn254Fr; BN254_DIGEST_SIZE];

    fn constant_compress(input: [Self::Digest; 2]) -> Self::Digest {
        let mut state = [input[0][0], input[1][0], Bn254Fr::zero()];
        outer_perm().permute_mut(&mut state);
        [state[0]; BN254_DIGEST_SIZE]
    }
}
#[cfg(feature = "babybear")]
impl FieldHasher<BabyBear> for SCBabyBearPoseidon2Outer {
    type Digest = [Bn254Fr; BN254_DIGEST_SIZE];

    fn constant_compress(input: [Self::Digest; 2]) -> Self::Digest {
        let mut state = [input[0][0], input[1][0], Bn254Fr::zero()];
        outer_perm().permute_mut(&mut state);
        [state[0]; BN254_DIGEST_SIZE]
    }
}

impl<C: CircuitConfig<F = BabyBear, N = Bn254Fr, Bit = Var<Bn254Fr>>> FieldHasherVariable<C>
    for BabyBearPoseidon2Outer
{
    type DigestVariable = [Var<Bn254Fr>; BN254_DIGEST_SIZE];

    fn hash(builder: &mut Builder<C>, input: &[Felt<<C as Config>::F>]) -> Self::DigestVariable {
        assert!(C::N::bits() == p3_bn254_fr::Bn254Fr::bits());
        assert!(C::F::bits() == p3_baby_bear::BabyBear::bits());
        let num_f_elms = C::N::bits() / C::F::bits();
        let mut state: [Var<C::N>; OUTER_MULTI_FIELD_CHALLENGER_WIDTH] =
            [builder.eval(C::N::zero()), builder.eval(C::N::zero()), builder.eval(C::N::zero())];
        for block_chunk in &input.iter().chunks(POSEIDON_2_BB_RATE) {
            for (chunk_id, chunk) in (&block_chunk.chunks(num_f_elms)).into_iter().enumerate() {
                let chunk = chunk.copied().collect::<Vec<_>>();
                state[chunk_id] = reduce_32(builder, chunk.as_slice());
            }
            builder.push_op(DslIr::CircuitPoseidon2Permute(state))
        }

        [state[0]; BN254_DIGEST_SIZE]
    }

    fn compress(
        builder: &mut Builder<C>,
        input: [Self::DigestVariable; 2],
    ) -> Self::DigestVariable {
        let state: [Var<C::N>; OUTER_MULTI_FIELD_CHALLENGER_WIDTH] =
            [builder.eval(input[0][0]), builder.eval(input[1][0]), builder.eval(C::N::zero())];
        builder.push_op(DslIr::CircuitPoseidon2Permute(state));
        [state[0]; BN254_DIGEST_SIZE]
    }

    fn assert_digest_eq(
        builder: &mut Builder<C>,
        a: Self::DigestVariable,
        b: Self::DigestVariable,
    ) {
        zip(a, b).for_each(|(e1, e2)| builder.assert_var_eq(e1, e2));
    }

    fn select_chain_digest(
        builder: &mut Builder<C>,
        should_swap: <C as CircuitConfig>::Bit,
        input: [Self::DigestVariable; 2],
    ) -> [Self::DigestVariable; 2] {
        let result0: [Var<_>; BN254_DIGEST_SIZE] = core::array::from_fn(|j| {
            let result = builder.uninit();
            builder.push_op(DslIr::CircuitSelectV(should_swap, input[1][j], input[0][j], result));
            result
        });
        let result1: [Var<_>; BN254_DIGEST_SIZE] = core::array::from_fn(|j| {
            let result = builder.uninit();
            builder.push_op(DslIr::CircuitSelectV(should_swap, input[0][j], input[1][j], result));
            result
        });

        [result0, result1]
    }

    fn print_digest(builder: &mut Builder<C>, digest: Self::DigestVariable) {
        for d in digest.iter() {
            builder.print_v(*d);
        }
    }
}
#[cfg(feature = "babybear")]
impl<C: CircuitConfig<F = BabyBear, N = Bn254Fr, Bit = Var<Bn254Fr>>> FieldHasherVariable<C>
    for SCBabyBearPoseidon2Outer
{
    type DigestVariable = [Var<Bn254Fr>; BN254_DIGEST_SIZE];

    fn hash(builder: &mut Builder<C>, input: &[Felt<<C as Config>::F>]) -> Self::DigestVariable {
        assert!(C::N::bits() == p3_bn254_fr::Bn254Fr::bits());
        assert!(C::F::bits() == p3_baby_bear::BabyBear::bits());
        let num_f_elms = C::N::bits() / C::F::bits();
        let mut state: [Var<C::N>; OUTER_MULTI_FIELD_CHALLENGER_WIDTH] =
            [builder.eval(C::N::zero()), builder.eval(C::N::zero()), builder.eval(C::N::zero())];
        for block_chunk in &input.iter().chunks(POSEIDON_2_BB_RATE) {
            for (chunk_id, chunk) in (&block_chunk.chunks(num_f_elms)).into_iter().enumerate() {
                let chunk = chunk.copied().collect::<Vec<_>>();
                state[chunk_id] = reduce_32(builder, chunk.as_slice());
            }
            builder.push_op(DslIr::CircuitPoseidon2Permute(state))
        }

        [state[0]; BN254_DIGEST_SIZE]
    }

    fn compress(
        builder: &mut Builder<C>,
        input: [Self::DigestVariable; 2],
    ) -> Self::DigestVariable {
        let state: [Var<C::N>; OUTER_MULTI_FIELD_CHALLENGER_WIDTH] =
            [builder.eval(input[0][0]), builder.eval(input[1][0]), builder.eval(C::N::zero())];
        builder.push_op(DslIr::CircuitPoseidon2Permute(state));
        [state[0]; BN254_DIGEST_SIZE]
    }

    fn assert_digest_eq(
        builder: &mut Builder<C>,
        a: Self::DigestVariable,
        b: Self::DigestVariable,
    ) {
        zip(a, b).for_each(|(e1, e2)| builder.assert_var_eq(e1, e2));
    }

    fn select_chain_digest(
        builder: &mut Builder<C>,
        should_swap: <C as CircuitConfig>::Bit,
        input: [Self::DigestVariable; 2],
    ) -> [Self::DigestVariable; 2] {
        let result0: [Var<_>; BN254_DIGEST_SIZE] = core::array::from_fn(|j| {
            let result = builder.uninit();
            builder.push_op(DslIr::CircuitSelectV(should_swap, input[1][j], input[0][j], result));
            result
        });
        let result1: [Var<_>; BN254_DIGEST_SIZE] = core::array::from_fn(|j| {
            let result = builder.uninit();
            builder.push_op(DslIr::CircuitSelectV(should_swap, input[0][j], input[1][j], result));
            result
        });

        [result0, result1]
    }

    fn print_digest(builder: &mut Builder<C>, digest: Self::DigestVariable) {
        for d in digest.iter() {
            builder.print_v(*d);
        }
    }
}

#[cfg(feature = "koalabear")]
impl FieldHasher<p3_koala_bear::KoalaBear> for dt_recursion_core::stark::SCKoalaBearPoseidon2Outer {
    type Digest = [Bn254Fr; BN254_DIGEST_SIZE];

    fn constant_compress(input: [Self::Digest; 2]) -> Self::Digest {
        let mut state = [input[0][0], input[1][0], Bn254Fr::zero()];
        outer_perm().permute_mut(&mut state);
        [state[0]; BN254_DIGEST_SIZE]
    }
}

#[cfg(feature = "koalabear")]
impl<C: CircuitConfig<F = SCField, N = Bn254Fr, Bit = Var<Bn254Fr>>> FieldHasherVariable<C>
    for dt_recursion_core::stark::SCKoalaBearPoseidon2Outer
{
    type DigestVariable = [Var<Bn254Fr>; BN254_DIGEST_SIZE];

    fn hash(builder: &mut Builder<C>, input: &[Felt<<C as Config>::F>]) -> Self::DigestVariable {
        assert!(C::N::bits() == p3_bn254_fr::Bn254Fr::bits());
        assert!(C::F::bits() == SCField::bits());
        let num_f_elms = C::N::bits() / C::F::bits();
        let mut state: [Var<C::N>; OUTER_MULTI_FIELD_CHALLENGER_WIDTH] =
            [builder.eval(C::N::zero()), builder.eval(C::N::zero()), builder.eval(C::N::zero())];
        for block_chunk in &input.iter().chunks(POSEIDON_2_BB_RATE) {
            for (chunk_id, chunk) in (&block_chunk.chunks(num_f_elms)).into_iter().enumerate() {
                let chunk = chunk.copied().collect::<Vec<_>>();
                state[chunk_id] = reduce_32(builder, chunk.as_slice());
            }
            builder.push_op(DslIr::CircuitPoseidon2Permute(state))
        }

        [state[0]; BN254_DIGEST_SIZE]
    }

    fn compress(
        builder: &mut Builder<C>,
        input: [Self::DigestVariable; 2],
    ) -> Self::DigestVariable {
        let state: [Var<C::N>; OUTER_MULTI_FIELD_CHALLENGER_WIDTH] =
            [builder.eval(input[0][0]), builder.eval(input[1][0]), builder.eval(C::N::zero())];
        builder.push_op(DslIr::CircuitPoseidon2Permute(state));
        [state[0]; BN254_DIGEST_SIZE]
    }

    fn assert_digest_eq(
        builder: &mut Builder<C>,
        a: Self::DigestVariable,
        b: Self::DigestVariable,
    ) {
        zip(a, b).for_each(|(e1, e2)| builder.assert_var_eq(e1, e2));
    }

    fn select_chain_digest(
        builder: &mut Builder<C>,
        should_swap: <C as CircuitConfig>::Bit,
        input: [Self::DigestVariable; 2],
    ) -> [Self::DigestVariable; 2] {
        let result0: [Var<_>; BN254_DIGEST_SIZE] = core::array::from_fn(|j| {
            let result = builder.uninit();
            builder.push_op(DslIr::CircuitSelectV(should_swap, input[1][j], input[0][j], result));
            result
        });
        let result1: [Var<_>; BN254_DIGEST_SIZE] = core::array::from_fn(|j| {
            let result = builder.uninit();
            builder.push_op(DslIr::CircuitSelectV(should_swap, input[0][j], input[1][j], result));
            result
        });

        [result0, result1]
    }

    fn print_digest(builder: &mut Builder<C>, digest: Self::DigestVariable) {
        for d in digest.iter() {
            builder.print_v(*d);
        }
    }
}
