//! Copied from [`dt_recursion_program`].
#![allow(unused_mut)]
#![allow(unused_variables)]

use challenger::{
    CanCopyChallenger, CanObserveVariable, DuplexChallengerVariable, FieldChallengerVariable,
    MultiField32ChallengerVariable, SpongeChallengerShape,
};
use dt_recursion_compiler::{
    circuit::CircuitV2Builder,
    config::{InnerConfig, OuterConfig, ShrinkConfig},
    ir::{Builder, Config, DslIr, Ext, Felt, SymbolicFelt, Var, Variable},
};
use hash::{FieldHasherVariable, Posedion2BabyBearHasherVariable};
use itertools::izip;
use p3_bn254_fr::Bn254Fr;
use p3_field::{extension::BinomialExtensionField, AbstractField};
use p3_matrix::dense::RowMajorMatrix;
use std::iter::{repeat, zip};

mod types;

pub mod challenger;
pub mod constraints;
pub mod domain;
pub mod fri;
pub(crate) mod global_claim;
pub mod hash;
pub mod machine;
pub mod merkle_tree;
pub mod sc_machine;
pub mod stark;
pub mod sumcheck;
pub(crate) mod utils;
pub mod witness;

use dt_stark::{
    baby_bear_poseidon2::{BabyBearPoseidon2, ValMmcs},
    StarkGenericConfig,
};
pub use types::*;

use dt_recursion_core::{
    air::RecursionPublicValues,
    stark::{BabyBearPoseidon2Outer, OuterValMmcs},
    D,
};
use p3_challenger::{CanObserve, CanSample, FieldChallenger, GrindingChallenger};
use p3_commit::{ExtensionMmcs, Mmcs};
use p3_dft::Radix2DitParallel;
use p3_fri::{FriConfig, TwoAdicFriPcs};

use p3_baby_bear::BabyBear;
use utils::{felt_bytes_to_bn254_var, felts_to_bn254_var, words_to_bytes};

// type EF = <BabyBearPoseidon2 as StarkGenericConfig>::Challenge;
type EF = BinomialExtensionField<<BabyBearPoseidon2 as StarkGenericConfig>::Val, 4>;
pub type PcsConfig<C> = FriConfig<
    ExtensionMmcs<
        <C as StarkGenericConfig>::Val,
        // <C as StarkGenericConfig>::Challenge,
        BinomialExtensionField<<C as StarkGenericConfig>::Val, 4>,
        <C as BabyBearFriConfig>::ValMmcs,
    >,
>;

pub type Digest<C, SC> = <SC as FieldHasherVariable<C>>::DigestVariable;

pub type FriMmcs<C> = ExtensionMmcs<BabyBear, EF, <C as BabyBearFriConfig>::ValMmcs>;

pub trait BabyBearFriConfig:
    StarkGenericConfig<
    Val = BabyBear,
    Challenge = EF,
    Challenger = Self::FriChallenger,
    Pcs = TwoAdicFriPcs<
        BabyBear,
        Radix2DitParallel,
        Self::ValMmcs,
        ExtensionMmcs<BabyBear, EF, Self::ValMmcs>,
    >,
>
{
    type ValMmcs: Mmcs<BabyBear, ProverData<RowMajorMatrix<BabyBear>> = Self::RowMajorProverData>
        + Send
        + Sync;
    type RowMajorProverData: Clone + Send + Sync;
    type FriChallenger: CanObserve<<Self::ValMmcs as Mmcs<BabyBear>>::Commitment>
        + CanSample<EF>
        + GrindingChallenger<Witness = BabyBear>
        + FieldChallenger<BabyBear>;

    fn fri_config(&self) -> &FriConfig<FriMmcs<Self>>;

    fn challenger_shape(challenger: &Self::FriChallenger) -> SpongeChallengerShape;
}

pub trait BabyBearFriConfigVariable<C: CircuitConfig<F = BabyBear>>:
    BabyBearFriConfig + FieldHasherVariable<C> + Posedion2BabyBearHasherVariable<C>
{
    type FriChallengerVariable: FieldChallengerVariable<C, <C as CircuitConfig>::Bit>
        + CanObserveVariable<C, <Self as FieldHasherVariable<C>>::DigestVariable>
        + CanCopyChallenger<C>;

    /// Get a new challenger corresponding to the given config.
    fn challenger_variable(&self, builder: &mut Builder<C>) -> Self::FriChallengerVariable;

    fn commit_recursion_public_values(
        builder: &mut Builder<C>,
        public_values: RecursionPublicValues<Felt<C::F>>,
    );
}

pub trait CircuitConfig: Config {
    type Bit: Copy + Variable<Self>;

    fn read_bit(builder: &mut Builder<Self>) -> Self::Bit;

    fn read_bit_val(builder: &mut Builder<Self>, val: Self::N) -> Self::Bit;

    fn read_felt(builder: &mut Builder<Self>) -> Felt<Self::F>;
    /*

    fn read_felt_val(builder:&mut Builder<Self>, val:Self::F>)->Felt<Self::F>;
     */

    fn read_felt_val(builder: &mut Builder<Self>, val: Self::F) -> Felt<Self::F>;

    fn read_ext(builder: &mut Builder<Self>) -> Ext<Self::F, Self::EF>;

    fn read_ext_val(builder: &mut Builder<Self>, val: Self::EF) -> Ext<Self::F, Self::EF>;

    fn assert_bit_zero(builder: &mut Builder<Self>, bit: Self::Bit);

    fn assert_bit_one(builder: &mut Builder<Self>, bit: Self::Bit);

    fn ext2felt(
        builder: &mut Builder<Self>,
        ext: Ext<<Self as Config>::F, <Self as Config>::EF>,
    ) -> [Felt<<Self as Config>::F>; D];

    fn exp_reverse_bits(
        builder: &mut Builder<Self>,
        input: Felt<<Self as Config>::F>,
        power_bits: Vec<Self::Bit>,
    ) -> Felt<<Self as Config>::F>;
    fn exp_reverse_bits_ext(
        builder: &mut Builder<Self>,
        input: Ext<<Self as Config>::F, <Self as Config>::EF>,
        power_bits: Vec<Self::Bit>,
    ) -> Ext<<Self as Config>::F, <Self as Config>::EF>;

    /// Evaluate the multilinear eq polynomial: ∏ᵢ eq(xᵢ, yᵢ) where
    /// eq(x, y) = 1 - x - y + 2xy.
    ///
    /// Default implementation delegates to `builder.eq_poly` which uses the
    /// `PrefixSumChecksChip`. Configs that inline this into ExtAlu ops
    /// (e.g. `ShrinkConfig`) override this method.
    fn eq_poly(
        builder: &mut Builder<Self>,
        x_vec: Vec<Ext<<Self as Config>::F, <Self as Config>::EF>>,
        y_vec: Vec<Ext<<Self as Config>::F, <Self as Config>::EF>>,
    ) -> Ext<<Self as Config>::F, <Self as Config>::EF> {
        builder.eq_poly(x_vec, y_vec)
    }

    /// Exponentiates a felt x to a list of bits in little endian. Uses precomputed powers
    /// of x.
    fn exp_f_bits_precomputed(
        builder: &mut Builder<Self>,
        power_bits: &[Self::Bit],
        two_adic_powers_of_x: &[Felt<Self::F>],
    ) -> Felt<Self::F>;

    fn batch_fri(
        builder: &mut Builder<Self>,
        alpha_pows: Vec<Ext<Self::F, Self::EF>>,
        p_at_zs: Vec<Ext<Self::F, Self::EF>>,
        p_at_xs: Vec<Felt<Self::F>>,
    ) -> Ext<Self::F, Self::EF>;

    fn num2bits(
        builder: &mut Builder<Self>,
        num: Felt<<Self as Config>::F>,
        num_bits: usize,
    ) -> Vec<Self::Bit>;

    fn bits2num(
        builder: &mut Builder<Self>,
        bits: impl IntoIterator<Item = Self::Bit>,
    ) -> Felt<<Self as Config>::F>;

    #[allow(clippy::type_complexity)]
    fn select_chain_f(
        builder: &mut Builder<Self>,
        should_swap: Self::Bit,
        first: impl IntoIterator<Item = Felt<<Self as Config>::F>> + Clone,
        second: impl IntoIterator<Item = Felt<<Self as Config>::F>> + Clone,
    ) -> Vec<Felt<<Self as Config>::F>>;

    #[allow(clippy::type_complexity)]
    fn select_chain_ef(
        builder: &mut Builder<Self>,
        should_swap: Self::Bit,
        first: impl IntoIterator<Item = Ext<<Self as Config>::F, <Self as Config>::EF>> + Clone,
        second: impl IntoIterator<Item = Ext<<Self as Config>::F, <Self as Config>::EF>> + Clone,
    ) -> Vec<Ext<<Self as Config>::F, <Self as Config>::EF>>;

    fn range_check_felt(builder: &mut Builder<Self>, value: Felt<Self::F>, num_bits: usize) {
        let bits = Self::num2bits(builder, value, 31);
        for bit in bits.into_iter().skip(num_bits) {
            Self::assert_bit_zero(builder, bit);
        }
    }
}

impl CircuitConfig for InnerConfig {
    type Bit = Felt<<Self as Config>::F>;

    fn assert_bit_zero(builder: &mut Builder<Self>, bit: Self::Bit) {
        builder.assert_felt_eq(bit, Self::F::zero());
    }

    fn assert_bit_one(builder: &mut Builder<Self>, bit: Self::Bit) {
        builder.assert_felt_eq(bit, Self::F::one());
    }

    fn read_bit(builder: &mut Builder<Self>) -> Self::Bit {
        builder.hint_felt_v2()
    }

    fn read_bit_val(builder: &mut Builder<Self>, val: Self::N) -> Self::Bit {
        let mut bit = builder.hint_felt_v2();
        #[cfg(feature = "verify")]
        {
            bit.val = val;
        }
        bit
    }

    fn read_felt(builder: &mut Builder<Self>) -> Felt<Self::F> {
        builder.hint_felt_v2()
    }

    fn read_felt_val(builder: &mut Builder<Self>, val: Self::F) -> Felt<Self::F> {
        let mut felt = builder.hint_felt_v2();
        #[cfg(feature = "verify")]
        {
            felt.val = val;
        }
        felt
    }

    fn read_ext(builder: &mut Builder<Self>) -> Ext<Self::F, Self::EF> {
        builder.hint_ext_v2()
    }

    fn read_ext_val(builder: &mut Builder<Self>, val: Self::EF) -> Ext<Self::F, Self::EF> {
        let mut ext = builder.hint_ext_v2();
        #[cfg(feature = "verify")]
        {
            ext.val = val;
        }
        ext
    }

    fn ext2felt(
        builder: &mut Builder<Self>,
        ext: Ext<<Self as Config>::F, <Self as Config>::EF>,
    ) -> [Felt<<Self as Config>::F>; D] {
        builder.ext2felt_v2(ext)
    }

    fn exp_reverse_bits(
        builder: &mut Builder<Self>,
        input: Felt<<Self as Config>::F>,
        power_bits: Vec<Felt<<Self as Config>::F>>,
    ) -> Felt<<Self as Config>::F> {
        builder.exp_reverse_bits_v2(input, power_bits)
    }
    fn exp_reverse_bits_ext(
        builder: &mut Builder<Self>,
        input: Ext<<Self as Config>::F, <Self as Config>::EF>,
        power_bits: Vec<Felt<<Self as Config>::F>>,
    ) -> Ext<<Self as Config>::F, <Self as Config>::EF> {
        builder.ext_exp_reverse_bits(input, power_bits)
    }

    fn batch_fri(
        builder: &mut Builder<Self>,
        alpha_pows: Vec<Ext<<Self as Config>::F, <Self as Config>::EF>>,
        p_at_zs: Vec<Ext<<Self as Config>::F, <Self as Config>::EF>>,
        p_at_xs: Vec<Felt<<Self as Config>::F>>,
    ) -> Ext<<Self as Config>::F, <Self as Config>::EF> {
        builder.batch_fri_v2(alpha_pows, p_at_zs, p_at_xs)
    }

    fn num2bits(
        builder: &mut Builder<Self>,
        num: Felt<<Self as Config>::F>,
        num_bits: usize,
    ) -> Vec<Felt<<Self as Config>::F>> {
        builder.num2bits_v2_f(num, num_bits)
    }

    fn bits2num(
        builder: &mut Builder<Self>,
        bits: impl IntoIterator<Item = Felt<<Self as Config>::F>>,
    ) -> Felt<<Self as Config>::F> {
        builder.bits2num_v2_f(bits)
    }

    fn select_chain_f(
        builder: &mut Builder<Self>,
        should_swap: Self::Bit,
        first: impl IntoIterator<Item = Felt<<Self as Config>::F>> + Clone,
        second: impl IntoIterator<Item = Felt<<Self as Config>::F>> + Clone,
    ) -> Vec<Felt<<Self as Config>::F>> {
        let one: Felt<_> = builder.constant(Self::F::one());
        let shouldnt_swap: Felt<_> = builder.eval(one - should_swap);

        let id_branch = first.clone().into_iter().chain(second.clone());
        let swap_branch = second.into_iter().chain(first);
        zip(zip(id_branch, swap_branch), zip(repeat(shouldnt_swap), repeat(should_swap)))
            .map(|((id_v, sw_v), (id_c, sw_c))| builder.eval(id_v * id_c + sw_v * sw_c))
            .collect()
    }

    fn select_chain_ef(
        builder: &mut Builder<Self>,
        should_swap: Self::Bit,
        first: impl IntoIterator<Item = Ext<<Self as Config>::F, <Self as Config>::EF>> + Clone,
        second: impl IntoIterator<Item = Ext<<Self as Config>::F, <Self as Config>::EF>> + Clone,
    ) -> Vec<Ext<<Self as Config>::F, <Self as Config>::EF>> {
        let one: Felt<_> = builder.constant(Self::F::one());
        let shouldnt_swap: Felt<_> = builder.eval(one - should_swap);

        let id_branch = first.clone().into_iter().chain(second.clone());
        let swap_branch = second.into_iter().chain(first);
        zip(zip(id_branch, swap_branch), zip(repeat(shouldnt_swap), repeat(should_swap)))
            .map(|((id_v, sw_v), (id_c, sw_c))| builder.eval(id_v * id_c + sw_v * sw_c))
            .collect()
    }

    fn exp_f_bits_precomputed(
        builder: &mut Builder<Self>,
        power_bits: &[Self::Bit],
        two_adic_powers_of_x: &[Felt<Self::F>],
    ) -> Felt<Self::F> {
        Self::exp_reverse_bits(
            builder,
            two_adic_powers_of_x[0],
            power_bits.iter().rev().copied().collect(),
        )
    }
}

#[cfg(feature = "koalabear")]
use dt_recursion_compiler::config::SCInnerConfig;
#[cfg(feature = "koalabear")]
impl CircuitConfig for SCInnerConfig {
    type Bit = Felt<<Self as Config>::F>;

    fn assert_bit_zero(builder: &mut Builder<Self>, bit: Self::Bit) {
        builder.assert_felt_eq(bit, Self::F::zero());
    }

    fn assert_bit_one(builder: &mut Builder<Self>, bit: Self::Bit) {
        builder.assert_felt_eq(bit, Self::F::one());
    }

    fn read_bit(builder: &mut Builder<Self>) -> Self::Bit {
        builder.hint_felt_v2()
    }

    fn read_bit_val(builder: &mut Builder<Self>, val: Self::N) -> Self::Bit {
        let mut bit = builder.hint_felt_v2();
        #[cfg(feature = "verify")]
        {
            bit.val = val;
        }
        bit
    }

    fn read_felt(builder: &mut Builder<Self>) -> Felt<Self::F> {
        builder.hint_felt_v2()
    }

    fn read_felt_val(builder: &mut Builder<Self>, val: Self::F) -> Felt<Self::F> {
        let mut felt = builder.hint_felt_v2();
        #[cfg(feature = "verify")]
        {
            felt.val = val;
        }
        felt
    }

    fn read_ext(builder: &mut Builder<Self>) -> Ext<Self::F, Self::EF> {
        builder.hint_ext_v2()
    }

    fn read_ext_val(builder: &mut Builder<Self>, val: Self::EF) -> Ext<Self::F, Self::EF> {
        let mut ext = builder.hint_ext_v2();
        #[cfg(feature = "verify")]
        {
            ext.val = val;
        }
        ext
    }

    fn ext2felt(
        builder: &mut Builder<Self>,
        ext: Ext<<Self as Config>::F, <Self as Config>::EF>,
    ) -> [Felt<<Self as Config>::F>; D] {
        builder.ext2felt_v2(ext)
    }

    fn exp_reverse_bits(
        builder: &mut Builder<Self>,
        input: Felt<<Self as Config>::F>,
        power_bits: Vec<Felt<<Self as Config>::F>>,
    ) -> Felt<<Self as Config>::F> {
        builder.exp_reverse_bits_v2(input, power_bits)
    }

    fn exp_reverse_bits_ext(
        builder: &mut Builder<Self>,
        input: Ext<<Self as Config>::F, <Self as Config>::EF>,
        power_bits: Vec<Felt<<Self as Config>::F>>,
    ) -> Ext<<Self as Config>::F, <Self as Config>::EF> {
        builder.ext_exp_reverse_bits(input, power_bits)
    }

    fn batch_fri(
        builder: &mut Builder<Self>,
        alpha_pows: Vec<Ext<<Self as Config>::F, <Self as Config>::EF>>,
        p_at_zs: Vec<Ext<<Self as Config>::F, <Self as Config>::EF>>,
        p_at_xs: Vec<Felt<<Self as Config>::F>>,
    ) -> Ext<<Self as Config>::F, <Self as Config>::EF> {
        builder.batch_fri_v2(alpha_pows, p_at_zs, p_at_xs)
    }

    fn num2bits(
        builder: &mut Builder<Self>,
        num: Felt<<Self as Config>::F>,
        num_bits: usize,
    ) -> Vec<Felt<<Self as Config>::F>> {
        builder.num2bits_v2_f(num, num_bits)
    }

    fn bits2num(
        builder: &mut Builder<Self>,
        bits: impl IntoIterator<Item = Felt<<Self as Config>::F>>,
    ) -> Felt<<Self as Config>::F> {
        builder.bits2num_v2_f(bits)
    }

    fn select_chain_f(
        builder: &mut Builder<Self>,
        should_swap: Self::Bit,
        first: impl IntoIterator<Item = Felt<<Self as Config>::F>> + Clone,
        second: impl IntoIterator<Item = Felt<<Self as Config>::F>> + Clone,
    ) -> Vec<Felt<<Self as Config>::F>> {
        let one: Felt<_> = builder.constant(Self::F::one());
        let shouldnt_swap: Felt<_> = builder.eval(one - should_swap);

        let id_branch = first.clone().into_iter().chain(second.clone());
        let swap_branch = second.into_iter().chain(first);
        zip(zip(id_branch, swap_branch), zip(repeat(shouldnt_swap), repeat(should_swap)))
            .map(|((id_v, sw_v), (id_c, sw_c))| builder.eval(id_v * id_c + sw_v * sw_c))
            .collect()
    }

    fn select_chain_ef(
        builder: &mut Builder<Self>,
        should_swap: Self::Bit,
        first: impl IntoIterator<Item = Ext<<Self as Config>::F, <Self as Config>::EF>> + Clone,
        second: impl IntoIterator<Item = Ext<<Self as Config>::F, <Self as Config>::EF>> + Clone,
    ) -> Vec<Ext<<Self as Config>::F, <Self as Config>::EF>> {
        let one: Felt<_> = builder.constant(Self::F::one());
        let shouldnt_swap: Felt<_> = builder.eval(one - should_swap);

        let id_branch = first.clone().into_iter().chain(second.clone());
        let swap_branch = second.into_iter().chain(first);
        zip(zip(id_branch, swap_branch), zip(repeat(shouldnt_swap), repeat(should_swap)))
            .map(|((id_v, sw_v), (id_c, sw_c))| builder.eval(id_v * id_c + sw_v * sw_c))
            .collect()
    }

    fn exp_f_bits_precomputed(
        builder: &mut Builder<Self>,
        power_bits: &[Self::Bit],
        two_adic_powers_of_x: &[Felt<Self::F>],
    ) -> Felt<Self::F> {
        Self::exp_reverse_bits(
            builder,
            two_adic_powers_of_x[0],
            power_bits.iter().rev().copied().collect(),
        )
    }
}

/// Shrink-stage `CircuitConfig` impl.
///
/// All methods delegate to the same DSL operations as `SCInnerConfig`, except
/// for [`exp_reverse_bits_ext`], which is inlined as a manual square-and-multiply
/// loop instead of emitting `DslIr::CircuitV2ExtExpReverseBits`. This removes
/// the dependency on `ExtExpReverseBitsChip` for all FRI verifier code that is
/// compiled with `ShrinkConfig`, allowing `sc_shrink_machine` to drop the chip
/// and shrink the proof size.
#[cfg(any(feature = "koalabear", feature = "babybear"))]
impl CircuitConfig for ShrinkConfig {
    type Bit = Felt<<Self as Config>::F>;

    fn assert_bit_zero(builder: &mut Builder<Self>, bit: Self::Bit) {
        builder.assert_felt_eq(bit, Self::F::zero());
    }

    fn assert_bit_one(builder: &mut Builder<Self>, bit: Self::Bit) {
        builder.assert_felt_eq(bit, Self::F::one());
    }

    fn read_bit(builder: &mut Builder<Self>) -> Self::Bit {
        builder.hint_felt_v2()
    }

    fn read_bit_val(builder: &mut Builder<Self>, val: Self::N) -> Self::Bit {
        let mut bit = builder.hint_felt_v2();
        #[cfg(feature = "verify")]
        {
            bit.val = val;
        }
        bit
    }

    fn read_felt(builder: &mut Builder<Self>) -> Felt<Self::F> {
        builder.hint_felt_v2()
    }

    fn read_felt_val(builder: &mut Builder<Self>, val: Self::F) -> Felt<Self::F> {
        let mut felt = builder.hint_felt_v2();
        #[cfg(feature = "verify")]
        {
            felt.val = val;
        }
        felt
    }

    fn read_ext(builder: &mut Builder<Self>) -> Ext<Self::F, Self::EF> {
        builder.hint_ext_v2()
    }

    fn read_ext_val(builder: &mut Builder<Self>, val: Self::EF) -> Ext<Self::F, Self::EF> {
        let mut ext = builder.hint_ext_v2();
        #[cfg(feature = "verify")]
        {
            ext.val = val;
        }
        ext
    }

    fn ext2felt(
        builder: &mut Builder<Self>,
        ext: Ext<<Self as Config>::F, <Self as Config>::EF>,
    ) -> [Felt<<Self as Config>::F>; D] {
        builder.ext2felt_v2(ext)
    }

    fn exp_reverse_bits(
        builder: &mut Builder<Self>,
        input: Felt<<Self as Config>::F>,
        power_bits: Vec<Felt<<Self as Config>::F>>,
    ) -> Felt<<Self as Config>::F> {
        builder.exp_reverse_bits_v2(input, power_bits)
    }

    /// Inlined `x^e` over the extension field where `e`'s bits are given in
    /// reverse-bit (high-to-low) order, identical in semantics to
    /// `builder.ext_exp_reverse_bits` but expressed purely with `ExtAlu`
    /// ops so that no `ExtExpReverseBitsChip` event is emitted.
    ///
    ///   result = 1
    ///   power  = input
    ///   for i in (1..=n).rev():
    ///       prod    = result * power               // ext * ext
    ///       diff    = prod - result                // ext - ext
    ///       result  = result + bit_i * diff        // = bit ? prod : result
    ///       power   = power * power                // ext * ext
    fn exp_reverse_bits_ext(
        builder: &mut Builder<Self>,
        input: Ext<<Self as Config>::F, <Self as Config>::EF>,
        power_bits: Vec<Felt<<Self as Config>::F>>,
    ) -> Ext<<Self as Config>::F, <Self as Config>::EF> {
        let mut result: Ext<_, _> = builder.constant(<Self as Config>::EF::one());
        let mut power_e = input;
        let bit_len = power_bits.len();

        for i in 1..=bit_len {
            let index = bit_len - i;
            let bit = power_bits[index];
            let prod: Ext<_, _> = builder.eval(result * power_e);
            // diff = prod - result, both Ext.
            let diff: Ext<_, _> = builder.eval(prod - result);
            // result = result + bit * diff. `Felt * Ext` is not implemented,
            // but `Ext * Felt` is supported via `SymbolicExt`, so place the
            // Ext on the left.
            result = builder.eval(result + diff * bit);
            power_e = builder.eval(power_e * power_e);
        }
        result
    }

    /// Inlined eq polynomial: ∏ᵢ (1 - xᵢ - yᵢ + 2·xᵢ·yᵢ), expressed purely
    /// with `ExtAlu` ops so that no `PrefixSumChecksChip` event is emitted.
    /// This allows `sc_shrink_machine` to drop the chip and shrink the proof size.
    ///
    /// Per step: 2 MulE + 2 SubE + 2 AddE = 6 ExtAlu ops.
    fn eq_poly(
        builder: &mut Builder<Self>,
        x_vec: Vec<Ext<<Self as Config>::F, <Self as Config>::EF>>,
        y_vec: Vec<Ext<<Self as Config>::F, <Self as Config>::EF>>,
    ) -> Ext<<Self as Config>::F, <Self as Config>::EF> {
        type EF<S> = Ext<<S as Config>::F, <S as Config>::EF>;
        let one: EF<Self> = builder.constant(<Self as Config>::EF::one());
        let mut result: EF<Self> = one;
        for (x, y) in x_vec.into_iter().zip(y_vec.into_iter()) {
            let p: EF<Self> = builder.eval(x * y); // MulE: p = x*y
            let two_p: EF<Self> = builder.eval(p + p); // AddE: 2p = p+p
            let diff: EF<Self> = builder.eval(two_p - x - y); // SubE + SubE: 2p - x - y
            let t: EF<Self> = builder.eval(diff + one); // AddE: 2p - x - y + 1
            result = builder.eval(result * t); // MulE: acc *= t
        }
        result
    }

    fn batch_fri(
        builder: &mut Builder<Self>,
        alpha_pows: Vec<Ext<<Self as Config>::F, <Self as Config>::EF>>,
        p_at_zs: Vec<Ext<<Self as Config>::F, <Self as Config>::EF>>,
        p_at_xs: Vec<Felt<<Self as Config>::F>>,
    ) -> Ext<<Self as Config>::F, <Self as Config>::EF> {
        builder.batch_fri_v2(alpha_pows, p_at_zs, p_at_xs)
    }

    fn num2bits(
        builder: &mut Builder<Self>,
        num: Felt<<Self as Config>::F>,
        num_bits: usize,
    ) -> Vec<Felt<<Self as Config>::F>> {
        builder.num2bits_v2_f(num, num_bits)
    }

    fn bits2num(
        builder: &mut Builder<Self>,
        bits: impl IntoIterator<Item = Felt<<Self as Config>::F>>,
    ) -> Felt<<Self as Config>::F> {
        builder.bits2num_v2_f(bits)
    }

    fn select_chain_f(
        builder: &mut Builder<Self>,
        should_swap: Self::Bit,
        first: impl IntoIterator<Item = Felt<<Self as Config>::F>> + Clone,
        second: impl IntoIterator<Item = Felt<<Self as Config>::F>> + Clone,
    ) -> Vec<Felt<<Self as Config>::F>> {
        let one: Felt<_> = builder.constant(Self::F::one());
        let shouldnt_swap: Felt<_> = builder.eval(one - should_swap);

        let id_branch = first.clone().into_iter().chain(second.clone());
        let swap_branch = second.into_iter().chain(first);
        zip(zip(id_branch, swap_branch), zip(repeat(shouldnt_swap), repeat(should_swap)))
            .map(|((id_v, sw_v), (id_c, sw_c))| builder.eval(id_v * id_c + sw_v * sw_c))
            .collect()
    }

    fn select_chain_ef(
        builder: &mut Builder<Self>,
        should_swap: Self::Bit,
        first: impl IntoIterator<Item = Ext<<Self as Config>::F, <Self as Config>::EF>> + Clone,
        second: impl IntoIterator<Item = Ext<<Self as Config>::F, <Self as Config>::EF>> + Clone,
    ) -> Vec<Ext<<Self as Config>::F, <Self as Config>::EF>> {
        let one: Felt<_> = builder.constant(Self::F::one());
        let shouldnt_swap: Felt<_> = builder.eval(one - should_swap);

        let id_branch = first.clone().into_iter().chain(second.clone());
        let swap_branch = second.into_iter().chain(first);
        zip(zip(id_branch, swap_branch), zip(repeat(shouldnt_swap), repeat(should_swap)))
            .map(|((id_v, sw_v), (id_c, sw_c))| builder.eval(id_v * id_c + sw_v * sw_c))
            .collect()
    }

    fn exp_f_bits_precomputed(
        builder: &mut Builder<Self>,
        power_bits: &[Self::Bit],
        two_adic_powers_of_x: &[Felt<Self::F>],
    ) -> Felt<Self::F> {
        Self::exp_reverse_bits(
            builder,
            two_adic_powers_of_x[0],
            power_bits.iter().rev().copied().collect(),
        )
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WrapConfig;

impl Config for WrapConfig {
    type F = <InnerConfig as Config>::F;
    type EF = <InnerConfig as Config>::EF;
    type N = <InnerConfig as Config>::N;
}

impl CircuitConfig for WrapConfig {
    type Bit = <InnerConfig as CircuitConfig>::Bit;

    fn assert_bit_zero(builder: &mut Builder<Self>, bit: Self::Bit) {
        builder.assert_felt_eq(bit, Self::F::zero());
    }

    fn assert_bit_one(builder: &mut Builder<Self>, bit: Self::Bit) {
        builder.assert_felt_eq(bit, Self::F::one());
    }

    fn read_bit(builder: &mut Builder<Self>) -> Self::Bit {
        builder.hint_felt_v2()
    }

    fn read_bit_val(builder: &mut Builder<Self>, val: Self::N) -> Self::Bit {
        let mut bit = builder.hint_felt_v2();
        #[cfg(feature = "verify")]
        {
            bit.val = val;
        }
        bit
    }

    fn read_felt(builder: &mut Builder<Self>) -> Felt<Self::F> {
        builder.hint_felt_v2()
    }

    fn read_felt_val(builder: &mut Builder<Self>, val: Self::F) -> Felt<Self::F> {
        let mut felt = builder.hint_felt_v2();
        #[cfg(feature = "verify")]
        {
            felt.val = val;
        }
        felt
    }

    fn read_ext(builder: &mut Builder<Self>) -> Ext<Self::F, Self::EF> {
        builder.hint_ext_v2()
    }

    fn read_ext_val(builder: &mut Builder<Self>, val: Self::EF) -> Ext<Self::F, Self::EF> {
        let mut ext = builder.hint_ext_v2();
        #[cfg(feature = "verify")]
        {
            ext.val = val;
        }
        ext
    }

    fn ext2felt(
        builder: &mut Builder<Self>,
        ext: Ext<<Self as Config>::F, <Self as Config>::EF>,
    ) -> [Felt<<Self as Config>::F>; D] {
        builder.ext2felt_v2(ext)
    }

    fn exp_reverse_bits(
        builder: &mut Builder<Self>,
        input: Felt<<Self as Config>::F>,
        power_bits: Vec<Felt<<Self as Config>::F>>,
    ) -> Felt<<Self as Config>::F> {
        // builder.exp_reverse_bits_v2(input, power_bits)
        let mut result = builder.constant(Self::F::one());
        let mut power_f = input;
        let bit_len = power_bits.len();

        for i in 1..=bit_len {
            let index = bit_len - i;
            let bit = power_bits[index];
            let prod: Felt<_> = builder.eval(result * power_f);
            result = builder.eval(bit * prod + (SymbolicFelt::one() - bit) * result);
            power_f = builder.eval(power_f * power_f);
        }
        result
    }
    fn exp_reverse_bits_ext(
        builder: &mut Builder<Self>,
        input: Ext<<Self as Config>::F, <Self as Config>::EF>,
        power_bits: Vec<Felt<<Self as Config>::F>>,
    ) -> Ext<<Self as Config>::F, <Self as Config>::EF> {
        builder.ext_exp_reverse_bits(input, power_bits)
    }

    fn batch_fri(
        builder: &mut Builder<Self>,
        alpha_pows: Vec<Ext<<Self as Config>::F, <Self as Config>::EF>>,
        p_at_zs: Vec<Ext<<Self as Config>::F, <Self as Config>::EF>>,
        p_at_xs: Vec<Felt<<Self as Config>::F>>,
    ) -> Ext<<Self as Config>::F, <Self as Config>::EF> {
        // builder.batch_fri_v2(alpha_pows, p_at_zs, p_at_xs)
        // Initialize the `acc` to zero.
        let mut acc: Ext<_, _> = builder.uninit();
        builder.push_op(DslIr::ImmE(acc, <Self as Config>::EF::zero()));
        for (alpha_pow, p_at_z, p_at_x) in izip!(alpha_pows, p_at_zs, p_at_xs) {
            // Set `temp_1 = p_at_z - p_at_x`
            let temp_1: Ext<_, _> = builder.uninit();
            builder.push_op(DslIr::SubEF(temp_1, p_at_z, p_at_x));
            // Set `temp_2 = alpha_pow * temp_1 = alpha_pow * (p_at_z - p_at_x)`
            let temp_2: Ext<_, _> = builder.uninit();
            builder.push_op(DslIr::MulE(temp_2, alpha_pow, temp_1));
            // Set `acc += temp_2`, so that `acc` becomes the sum of `alpha_pow * (p_at_z - p_at_x)`
            let temp_3: Ext<_, _> = builder.uninit();
            builder.push_op(DslIr::AddE(temp_3, acc, temp_2));
            acc = temp_3;
        }
        acc
    }

    fn num2bits(
        builder: &mut Builder<Self>,
        num: Felt<<Self as Config>::F>,
        num_bits: usize,
    ) -> Vec<Felt<<Self as Config>::F>> {
        builder.num2bits_v2_f(num, num_bits)
    }

    fn bits2num(
        builder: &mut Builder<Self>,
        bits: impl IntoIterator<Item = Felt<<Self as Config>::F>>,
    ) -> Felt<<Self as Config>::F> {
        builder.bits2num_v2_f(bits)
    }

    fn select_chain_f(
        builder: &mut Builder<Self>,
        should_swap: Self::Bit,
        first: impl IntoIterator<Item = Felt<<Self as Config>::F>> + Clone,
        second: impl IntoIterator<Item = Felt<<Self as Config>::F>> + Clone,
    ) -> Vec<Felt<<Self as Config>::F>> {
        let one: Felt<_> = builder.constant(Self::F::one());
        let shouldnt_swap: Felt<_> = builder.eval(one - should_swap);

        let id_branch = first.clone().into_iter().chain(second.clone());
        let swap_branch = second.into_iter().chain(first);
        zip(zip(id_branch, swap_branch), zip(repeat(shouldnt_swap), repeat(should_swap)))
            .map(|((id_v, sw_v), (id_c, sw_c))| builder.eval(id_v * id_c + sw_v * sw_c))
            .collect()
    }

    fn select_chain_ef(
        builder: &mut Builder<Self>,
        should_swap: Self::Bit,
        first: impl IntoIterator<Item = Ext<<Self as Config>::F, <Self as Config>::EF>> + Clone,
        second: impl IntoIterator<Item = Ext<<Self as Config>::F, <Self as Config>::EF>> + Clone,
    ) -> Vec<Ext<<Self as Config>::F, <Self as Config>::EF>> {
        let one: Felt<_> = builder.constant(Self::F::one());
        let shouldnt_swap: Felt<_> = builder.eval(one - should_swap);

        let id_branch = first.clone().into_iter().chain(second.clone());
        let swap_branch = second.into_iter().chain(first);
        zip(zip(id_branch, swap_branch), zip(repeat(shouldnt_swap), repeat(should_swap)))
            .map(|((id_v, sw_v), (id_c, sw_c))| builder.eval(id_v * id_c + sw_v * sw_c))
            .collect()
    }

    fn exp_f_bits_precomputed(
        builder: &mut Builder<Self>,
        power_bits: &[Self::Bit],
        two_adic_powers_of_x: &[Felt<Self::F>],
    ) -> Felt<Self::F> {
        Self::exp_reverse_bits(
            builder,
            two_adic_powers_of_x[0],
            power_bits.iter().rev().copied().collect(),
        )
    }
}

/// KoalaBear version of [`WrapConfig`] for SC Prover wrap stage.
#[cfg(feature = "koalabear")]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SCWrapConfig;

#[cfg(feature = "koalabear")]
impl Config for SCWrapConfig {
    type F = <SCInnerConfig as Config>::F;
    type EF = <SCInnerConfig as Config>::EF;
    type N = <SCInnerConfig as Config>::N;
}

#[cfg(feature = "koalabear")]
impl CircuitConfig for SCWrapConfig {
    type Bit = <SCInnerConfig as CircuitConfig>::Bit;

    fn assert_bit_zero(builder: &mut Builder<Self>, bit: Self::Bit) {
        builder.assert_felt_eq(bit, Self::F::zero());
    }

    fn assert_bit_one(builder: &mut Builder<Self>, bit: Self::Bit) {
        builder.assert_felt_eq(bit, Self::F::one());
    }

    fn read_bit(builder: &mut Builder<Self>) -> Self::Bit {
        builder.hint_felt_v2()
    }

    fn read_bit_val(builder: &mut Builder<Self>, val: Self::N) -> Self::Bit {
        let mut bit = builder.hint_felt_v2();
        #[cfg(feature = "verify")]
        {
            bit.val = val;
        }
        bit
    }

    fn read_felt(builder: &mut Builder<Self>) -> Felt<Self::F> {
        builder.hint_felt_v2()
    }

    fn read_felt_val(builder: &mut Builder<Self>, val: Self::F) -> Felt<Self::F> {
        let mut felt = builder.hint_felt_v2();
        #[cfg(feature = "verify")]
        {
            felt.val = val;
        }
        felt
    }

    fn read_ext(builder: &mut Builder<Self>) -> Ext<Self::F, Self::EF> {
        builder.hint_ext_v2()
    }

    fn read_ext_val(builder: &mut Builder<Self>, val: Self::EF) -> Ext<Self::F, Self::EF> {
        let mut ext = builder.hint_ext_v2();
        #[cfg(feature = "verify")]
        {
            ext.val = val;
        }
        ext
    }

    fn ext2felt(
        builder: &mut Builder<Self>,
        ext: Ext<<Self as Config>::F, <Self as Config>::EF>,
    ) -> [Felt<<Self as Config>::F>; D] {
        builder.ext2felt_v2(ext)
    }

    fn exp_reverse_bits(
        builder: &mut Builder<Self>,
        input: Felt<<Self as Config>::F>,
        power_bits: Vec<Felt<<Self as Config>::F>>,
    ) -> Felt<<Self as Config>::F> {
        let mut result = builder.constant(Self::F::one());
        let mut power_f = input;
        let bit_len = power_bits.len();

        for i in 1..=bit_len {
            let index = bit_len - i;
            let bit = power_bits[index];
            let prod: Felt<_> = builder.eval(result * power_f);
            result = builder.eval(bit * prod + (SymbolicFelt::one() - bit) * result);
            power_f = builder.eval(power_f * power_f);
        }
        result
    }

    fn exp_reverse_bits_ext(
        builder: &mut Builder<Self>,
        input: Ext<<Self as Config>::F, <Self as Config>::EF>,
        power_bits: Vec<Felt<<Self as Config>::F>>,
    ) -> Ext<<Self as Config>::F, <Self as Config>::EF> {
        builder.ext_exp_reverse_bits(input, power_bits)
    }

    fn batch_fri(
        builder: &mut Builder<Self>,
        alpha_pows: Vec<Ext<<Self as Config>::F, <Self as Config>::EF>>,
        p_at_zs: Vec<Ext<<Self as Config>::F, <Self as Config>::EF>>,
        p_at_xs: Vec<Felt<<Self as Config>::F>>,
    ) -> Ext<<Self as Config>::F, <Self as Config>::EF> {
        let mut acc: Ext<_, _> = builder.uninit();
        builder.push_op(DslIr::ImmE(acc, <Self as Config>::EF::zero()));
        for (alpha_pow, p_at_z, p_at_x) in izip!(alpha_pows, p_at_zs, p_at_xs) {
            let temp_1: Ext<_, _> = builder.uninit();
            builder.push_op(DslIr::SubEF(temp_1, p_at_z, p_at_x));
            let temp_2: Ext<_, _> = builder.uninit();
            builder.push_op(DslIr::MulE(temp_2, alpha_pow, temp_1));
            builder.push_op(DslIr::AddE(acc, acc, temp_2));
        }
        acc
    }

    fn num2bits(
        builder: &mut Builder<Self>,
        num: Felt<<Self as Config>::F>,
        num_bits: usize,
    ) -> Vec<Felt<<Self as Config>::F>> {
        builder.num2bits_v2_f(num, num_bits)
    }

    fn bits2num(
        builder: &mut Builder<Self>,
        bits: impl IntoIterator<Item = Felt<<Self as Config>::F>>,
    ) -> Felt<<Self as Config>::F> {
        builder.bits2num_v2_f(bits)
    }

    fn select_chain_f(
        builder: &mut Builder<Self>,
        should_swap: Self::Bit,
        first: impl IntoIterator<Item = Felt<<Self as Config>::F>> + Clone,
        second: impl IntoIterator<Item = Felt<<Self as Config>::F>> + Clone,
    ) -> Vec<Felt<<Self as Config>::F>> {
        let one: Felt<_> = builder.constant(Self::F::one());
        let shouldnt_swap: Felt<_> = builder.eval(one - should_swap);

        let id_branch = first.clone().into_iter().chain(second.clone());
        let swap_branch = second.into_iter().chain(first);
        zip(zip(id_branch, swap_branch), zip(repeat(shouldnt_swap), repeat(should_swap)))
            .map(|((id_v, sw_v), (id_c, sw_c))| builder.eval(id_v * id_c + sw_v * sw_c))
            .collect()
    }

    fn select_chain_ef(
        builder: &mut Builder<Self>,
        should_swap: Self::Bit,
        first: impl IntoIterator<Item = Ext<<Self as Config>::F, <Self as Config>::EF>> + Clone,
        second: impl IntoIterator<Item = Ext<<Self as Config>::F, <Self as Config>::EF>> + Clone,
    ) -> Vec<Ext<<Self as Config>::F, <Self as Config>::EF>> {
        let one: Felt<_> = builder.constant(Self::F::one());
        let shouldnt_swap: Felt<_> = builder.eval(one - should_swap);

        let id_branch = first.clone().into_iter().chain(second.clone());
        let swap_branch = second.into_iter().chain(first);
        zip(zip(id_branch, swap_branch), zip(repeat(shouldnt_swap), repeat(should_swap)))
            .map(|((id_v, sw_v), (id_c, sw_c))| builder.eval(id_v * id_c + sw_v * sw_c))
            .collect()
    }

    fn exp_f_bits_precomputed(
        builder: &mut Builder<Self>,
        power_bits: &[Self::Bit],
        two_adic_powers_of_x: &[Felt<Self::F>],
    ) -> Felt<Self::F> {
        Self::exp_reverse_bits(
            builder,
            two_adic_powers_of_x[0],
            power_bits.iter().rev().copied().collect(),
        )
    }
}

impl CircuitConfig for OuterConfig {
    type Bit = Var<<Self as Config>::N>;

    fn assert_bit_zero(builder: &mut Builder<Self>, bit: Self::Bit) {
        builder.assert_var_eq(bit, Self::N::zero());
    }

    fn assert_bit_one(builder: &mut Builder<Self>, bit: Self::Bit) {
        builder.assert_var_eq(bit, Self::N::one());
    }

    fn read_bit(builder: &mut Builder<Self>) -> Self::Bit {
        builder.witness_var()
    }

    fn read_bit_val(builder: &mut Builder<Self>, val: Self::N) -> Self::Bit {
        let mut bit = builder.witness_var();
        #[cfg(feature = "verify")]
        {
            bit.val = val;
        }
        bit
    }

    fn read_felt(builder: &mut Builder<Self>) -> Felt<Self::F> {
        builder.witness_felt()
    }

    fn read_felt_val(builder: &mut Builder<Self>, val: Self::F) -> Felt<Self::F> {
        let mut felt = builder.witness_felt();
        #[cfg(feature = "verify")]
        {
            felt.val = val;
        }
        felt
    }

    fn read_ext(builder: &mut Builder<Self>) -> Ext<Self::F, Self::EF> {
        builder.witness_ext()
    }

    fn read_ext_val(builder: &mut Builder<Self>, val: Self::EF) -> Ext<Self::F, Self::EF> {
        let mut ext = builder.witness_ext();
        #[cfg(feature = "verify")]
        {
            ext.val = val;
        }
        ext
    }

    fn ext2felt(
        builder: &mut Builder<Self>,
        ext: Ext<<Self as Config>::F, <Self as Config>::EF>,
    ) -> [Felt<<Self as Config>::F>; D] {
        let felts = core::array::from_fn(|_| builder.uninit());
        builder.push_op(DslIr::CircuitExt2Felt(felts, ext));
        felts
    }

    fn exp_reverse_bits(
        builder: &mut Builder<Self>,
        input: Felt<<Self as Config>::F>,
        power_bits: Vec<Var<<Self as Config>::N>>,
    ) -> Felt<<Self as Config>::F> {
        let mut result = builder.constant(Self::F::one());
        let power_f = input;
        let bit_len = power_bits.len();

        for i in 1..=bit_len {
            let index = bit_len - i;
            let bit = power_bits[index];
            let prod = builder.eval(result * power_f);
            result = builder.select_f(bit, prod, result);
            builder.assign(power_f, power_f * power_f);
        }
        result
    }
    fn exp_reverse_bits_ext(
        builder: &mut Builder<Self>,
        input: Ext<<Self as Config>::F, <Self as Config>::EF>,
        power_bits: Vec<Var<<Self as Config>::N>>,
    ) -> Ext<<Self as Config>::F, <Self as Config>::EF> {
        let mut result = builder.constant(Self::EF::one());
        let power_f = input;
        let bit_len = power_bits.len();

        for i in 1..=bit_len {
            let index = bit_len - i;
            let bit = power_bits[index];
            let prod = builder.eval(result * power_f);
            result = builder.select_ef(bit, prod, result);
            builder.assign(power_f, power_f * power_f);
        }
        result
    }

    fn batch_fri(
        builder: &mut Builder<Self>,
        alpha_pows: Vec<Ext<<Self as Config>::F, <Self as Config>::EF>>,
        p_at_zs: Vec<Ext<<Self as Config>::F, <Self as Config>::EF>>,
        p_at_xs: Vec<Felt<<Self as Config>::F>>,
    ) -> Ext<<Self as Config>::F, <Self as Config>::EF> {
        // Initialize the `acc` to zero.
        let mut acc: Ext<_, _> = builder.uninit();
        builder.push_op(DslIr::ImmE(acc, <Self as Config>::EF::zero()));
        for (alpha_pow, p_at_z, p_at_x) in izip!(alpha_pows, p_at_zs, p_at_xs) {
            // Set `temp_1 = p_at_z - p_at_x`
            let temp_1: Ext<_, _> = builder.uninit();
            builder.push_op(DslIr::SubEF(temp_1, p_at_z, p_at_x));
            // Set `temp_2 = alpha_pow * temp_1 = alpha_pow * (p_at_z - p_at_x)`
            let temp_2: Ext<_, _> = builder.uninit();
            builder.push_op(DslIr::MulE(temp_2, alpha_pow, temp_1));
            // Set `acc += temp_2`, so that `acc` becomes the sum of `alpha_pow * (p_at_z - p_at_x)`
            let temp_3: Ext<_, _> = builder.uninit();
            builder.push_op(DslIr::AddE(temp_3, acc, temp_2));
            acc = temp_3;
        }
        acc
    }

    fn num2bits(
        builder: &mut Builder<Self>,
        num: Felt<<Self as Config>::F>,
        num_bits: usize,
    ) -> Vec<Var<<Self as Config>::N>> {
        builder.num2bits_f_circuit(num)[..num_bits].to_vec()
    }

    fn bits2num(
        builder: &mut Builder<Self>,
        bits: impl IntoIterator<Item = Var<<Self as Config>::N>>,
    ) -> Felt<<Self as Config>::F> {
        let result = builder.eval(Self::F::zero());
        for (i, bit) in bits.into_iter().enumerate() {
            let to_add: Felt<_> = builder.uninit();
            let pow2 = builder.constant(Self::F::from_canonical_u32(1 << i));
            let zero = builder.constant(Self::F::zero());
            builder.push_op(DslIr::CircuitSelectF(bit, pow2, zero, to_add));
            builder.assign(result, result + to_add);
        }
        result
    }

    fn select_chain_f(
        builder: &mut Builder<Self>,
        should_swap: Self::Bit,
        first: impl IntoIterator<Item = Felt<<Self as Config>::F>> + Clone,
        second: impl IntoIterator<Item = Felt<<Self as Config>::F>> + Clone,
    ) -> Vec<Felt<<Self as Config>::F>> {
        let id_branch = first.clone().into_iter().chain(second.clone());
        let swap_branch = second.into_iter().chain(first);
        zip(id_branch, swap_branch)
            .map(|(id_v, sw_v): (Felt<_>, Felt<_>)| -> Felt<_> {
                let result: Felt<_> = builder.uninit();
                builder.push_op(DslIr::CircuitSelectF(should_swap, sw_v, id_v, result));
                result
            })
            .collect()
    }

    fn select_chain_ef(
        builder: &mut Builder<Self>,
        should_swap: Self::Bit,
        first: impl IntoIterator<Item = Ext<<Self as Config>::F, <Self as Config>::EF>> + Clone,
        second: impl IntoIterator<Item = Ext<<Self as Config>::F, <Self as Config>::EF>> + Clone,
    ) -> Vec<Ext<<Self as Config>::F, <Self as Config>::EF>> {
        let id_branch = first.clone().into_iter().chain(second.clone());
        let swap_branch = second.into_iter().chain(first);
        zip(id_branch, swap_branch)
            .map(|(id_v, sw_v): (Ext<_, _>, Ext<_, _>)| -> Ext<_, _> {
                let result: Ext<_, _> = builder.uninit();
                builder.push_op(DslIr::CircuitSelectE(should_swap, sw_v, id_v, result));
                result
            })
            .collect()
    }

    fn exp_f_bits_precomputed(
        builder: &mut Builder<Self>,
        power_bits: &[Self::Bit],
        two_adic_powers_of_x: &[Felt<Self::F>],
    ) -> Felt<Self::F> {
        let mut result: Felt<_> = builder.eval(Self::F::one());
        let one = builder.constant(Self::F::one());
        for (&bit, &power) in power_bits.iter().zip(two_adic_powers_of_x) {
            let multiplier = builder.select_f(bit, power, one);
            result = builder.eval(multiplier * result);
        }
        result
    }
}

#[cfg(any(feature = "koalabear", feature = "babybear"))]
use dt_recursion_compiler::config::SCOuterConfig;
#[cfg(any(feature = "koalabear", feature = "babybear"))]
impl CircuitConfig for SCOuterConfig {
    type Bit = Var<<Self as Config>::N>;

    fn assert_bit_zero(builder: &mut Builder<Self>, bit: Self::Bit) {
        builder.assert_var_eq(bit, Self::N::zero());
    }

    fn assert_bit_one(builder: &mut Builder<Self>, bit: Self::Bit) {
        builder.assert_var_eq(bit, Self::N::one());
    }

    fn read_bit(builder: &mut Builder<Self>) -> Self::Bit {
        builder.witness_var()
    }

    fn read_bit_val(builder: &mut Builder<Self>, val: Self::N) -> Self::Bit {
        let mut bit = builder.witness_var();
        #[cfg(feature = "verify")]
        {
            bit.val = val;
        }
        bit
    }

    fn read_felt(builder: &mut Builder<Self>) -> Felt<Self::F> {
        builder.witness_felt()
    }

    fn read_felt_val(builder: &mut Builder<Self>, val: Self::F) -> Felt<Self::F> {
        let mut felt = builder.witness_felt();
        #[cfg(feature = "verify")]
        {
            felt.val = val;
        }
        felt
    }

    fn read_ext(builder: &mut Builder<Self>) -> Ext<Self::F, Self::EF> {
        builder.witness_ext()
    }

    fn read_ext_val(builder: &mut Builder<Self>, val: Self::EF) -> Ext<Self::F, Self::EF> {
        let mut ext = builder.witness_ext();
        #[cfg(feature = "verify")]
        {
            ext.val = val;
        }
        ext
    }

    fn ext2felt(
        builder: &mut Builder<Self>,
        ext: Ext<<Self as Config>::F, <Self as Config>::EF>,
    ) -> [Felt<<Self as Config>::F>; D] {
        let felts = core::array::from_fn(|_| builder.uninit());
        builder.push_op(DslIr::CircuitExt2Felt(felts, ext));
        felts
    }

    fn exp_reverse_bits(
        builder: &mut Builder<Self>,
        input: Felt<<Self as Config>::F>,
        power_bits: Vec<Var<<Self as Config>::N>>,
    ) -> Felt<<Self as Config>::F> {
        let mut result = builder.constant(Self::F::one());
        let power_f = input;
        let bit_len = power_bits.len();

        for i in 1..=bit_len {
            let index = bit_len - i;
            let bit = power_bits[index];
            let prod = builder.eval(result * power_f);
            result = builder.select_f(bit, prod, result);
            builder.assign(power_f, power_f * power_f);
        }
        result
    }

    fn exp_reverse_bits_ext(
        builder: &mut Builder<Self>,
        input: Ext<<Self as Config>::F, <Self as Config>::EF>,
        power_bits: Vec<Var<<Self as Config>::N>>,
    ) -> Ext<<Self as Config>::F, <Self as Config>::EF> {
        let mut result = builder.constant(Self::EF::one());
        let power_f = input;
        let bit_len = power_bits.len();

        for i in 1..=bit_len {
            let index = bit_len - i;
            let bit = power_bits[index];
            let prod = builder.eval(result * power_f);
            result = builder.select_ef(bit, prod, result);
            builder.assign(power_f, power_f * power_f);
        }
        result
    }

    fn batch_fri(
        builder: &mut Builder<Self>,
        alpha_pows: Vec<Ext<<Self as Config>::F, <Self as Config>::EF>>,
        p_at_zs: Vec<Ext<<Self as Config>::F, <Self as Config>::EF>>,
        p_at_xs: Vec<Felt<<Self as Config>::F>>,
    ) -> Ext<<Self as Config>::F, <Self as Config>::EF> {
        let mut acc: Ext<_, _> = builder.uninit();
        builder.push_op(DslIr::ImmE(acc, <Self as Config>::EF::zero()));
        for (alpha_pow, p_at_z, p_at_x) in izip!(alpha_pows, p_at_zs, p_at_xs) {
            let temp_1: Ext<_, _> = builder.uninit();
            builder.push_op(DslIr::SubEF(temp_1, p_at_z, p_at_x));
            let temp_2: Ext<_, _> = builder.uninit();
            builder.push_op(DslIr::MulE(temp_2, alpha_pow, temp_1));
            let temp_3: Ext<_, _> = builder.uninit();
            builder.push_op(DslIr::AddE(temp_3, acc, temp_2));
            acc = temp_3;
        }
        acc
    }

    fn num2bits(
        builder: &mut Builder<Self>,
        num: Felt<<Self as Config>::F>,
        num_bits: usize,
    ) -> Vec<Var<<Self as Config>::N>> {
        builder.num2bits_f_circuit(num)[..num_bits].to_vec()
    }

    fn bits2num(
        builder: &mut Builder<Self>,
        bits: impl IntoIterator<Item = Var<<Self as Config>::N>>,
    ) -> Felt<<Self as Config>::F> {
        let result = builder.eval(Self::F::zero());
        for (i, bit) in bits.into_iter().enumerate() {
            let to_add: Felt<_> = builder.uninit();
            let pow2 = builder.constant(Self::F::from_canonical_u32(1 << i));
            let zero = builder.constant(Self::F::zero());
            builder.push_op(DslIr::CircuitSelectF(bit, pow2, zero, to_add));
            builder.assign(result, result + to_add);
        }
        result
    }

    fn select_chain_f(
        builder: &mut Builder<Self>,
        should_swap: Self::Bit,
        first: impl IntoIterator<Item = Felt<<Self as Config>::F>> + Clone,
        second: impl IntoIterator<Item = Felt<<Self as Config>::F>> + Clone,
    ) -> Vec<Felt<<Self as Config>::F>> {
        let id_branch = first.clone().into_iter().chain(second.clone());
        let swap_branch = second.into_iter().chain(first);
        zip(id_branch, swap_branch)
            .map(|(id_v, sw_v): (Felt<_>, Felt<_>)| -> Felt<_> {
                let result: Felt<_> = builder.uninit();
                builder.push_op(DslIr::CircuitSelectF(should_swap, sw_v, id_v, result));
                result
            })
            .collect()
    }

    fn select_chain_ef(
        builder: &mut Builder<Self>,
        should_swap: Self::Bit,
        first: impl IntoIterator<Item = Ext<<Self as Config>::F, <Self as Config>::EF>> + Clone,
        second: impl IntoIterator<Item = Ext<<Self as Config>::F, <Self as Config>::EF>> + Clone,
    ) -> Vec<Ext<<Self as Config>::F, <Self as Config>::EF>> {
        let id_branch = first.clone().into_iter().chain(second.clone());
        let swap_branch = second.into_iter().chain(first);
        zip(id_branch, swap_branch)
            .map(|(id_v, sw_v): (Ext<_, _>, Ext<_, _>)| -> Ext<_, _> {
                let result: Ext<_, _> = builder.uninit();
                builder.push_op(DslIr::CircuitSelectE(should_swap, sw_v, id_v, result));
                result
            })
            .collect()
    }

    fn exp_f_bits_precomputed(
        builder: &mut Builder<Self>,
        power_bits: &[Self::Bit],
        two_adic_powers_of_x: &[Felt<Self::F>],
    ) -> Felt<Self::F> {
        let mut result: Felt<_> = builder.eval(Self::F::one());
        let one = builder.constant(Self::F::one());
        for (&bit, &power) in power_bits.iter().zip(two_adic_powers_of_x) {
            let multiplier = builder.select_f(bit, power, one);
            result = builder.eval(multiplier * result);
        }
        result
    }
}

impl BabyBearFriConfig for BabyBearPoseidon2 {
    type ValMmcs = ValMmcs;
    type FriChallenger = <Self as StarkGenericConfig>::Challenger;
    type RowMajorProverData = <ValMmcs as Mmcs<BabyBear>>::ProverData<RowMajorMatrix<BabyBear>>;

    fn fri_config(&self) -> &FriConfig<FriMmcs<Self>> {
        self.pcs().fri_config()
    }

    fn challenger_shape(challenger: &Self::FriChallenger) -> SpongeChallengerShape {
        SpongeChallengerShape {
            input_buffer_len: challenger.input_buffer.len(),
            output_buffer_len: challenger.output_buffer.len(),
        }
    }
}

impl BabyBearFriConfig for BabyBearPoseidon2Outer {
    type ValMmcs = OuterValMmcs;
    type FriChallenger = <Self as StarkGenericConfig>::Challenger;

    type RowMajorProverData =
        <OuterValMmcs as Mmcs<BabyBear>>::ProverData<RowMajorMatrix<BabyBear>>;

    fn fri_config(&self) -> &FriConfig<FriMmcs<Self>> {
        self.pcs().fri_config()
    }

    fn challenger_shape(_challenger: &Self::FriChallenger) -> SpongeChallengerShape {
        unimplemented!("Shape not supported for outer fri challenger");
    }
}

impl<C: CircuitConfig<F = BabyBear, Bit = Felt<BabyBear>>> BabyBearFriConfigVariable<C>
    for BabyBearPoseidon2
{
    type FriChallengerVariable = DuplexChallengerVariable<C>;

    fn challenger_variable(&self, builder: &mut Builder<C>) -> Self::FriChallengerVariable {
        DuplexChallengerVariable::new(builder)
    }

    fn commit_recursion_public_values(
        builder: &mut Builder<C>,
        public_values: RecursionPublicValues<Felt<<C>::F>>,
    ) {
        builder.commit_public_values_v2(public_values);
    }
}

impl<C: CircuitConfig<F = BabyBear, N = Bn254Fr, Bit = Var<Bn254Fr>>> BabyBearFriConfigVariable<C>
    for BabyBearPoseidon2Outer
{
    type FriChallengerVariable = MultiField32ChallengerVariable<C>;

    fn challenger_variable(&self, builder: &mut Builder<C>) -> Self::FriChallengerVariable {
        MultiField32ChallengerVariable::new(builder)
    }

    fn commit_recursion_public_values(
        builder: &mut Builder<C>,
        public_values: RecursionPublicValues<Felt<<C>::F>>,
    ) {
        let committed_values_digest_bytes_felts: [Felt<_>; 32] =
            words_to_bytes(&public_values.committed_value_digest).try_into().unwrap();
        let committed_values_digest_bytes: Var<_> =
            felt_bytes_to_bn254_var(builder, &committed_values_digest_bytes_felts);
        builder.commit_committed_values_digest_circuit(committed_values_digest_bytes);

        let vkey_hash = felts_to_bn254_var(builder, &public_values.dt_vk_digest);
        builder.commit_vkey_hash_circuit(vkey_hash);
    }
}
