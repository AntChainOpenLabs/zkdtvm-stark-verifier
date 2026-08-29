//! An implementation of Poseidon2 over BN254.
#![allow(unused_mut)]
use std::{borrow::Cow, iter::repeat};

#[cfg(feature = "verify")]
use crate::ir::bits::u32_to_le_bits;
use crate::prelude::*;
use dt_recursion_core::{
    air::RecursionPublicValues, chips::poseidon2_skinny::WIDTH, D, DIGEST_SIZE, HASH_RATE,
};
use dt_stark::{
    septic_curve::SepticCurve, septic_digest::SepticDigest, septic_extension::SepticExtension,
};
use itertools::Itertools;
#[cfg(feature = "verify")]
use p3_field::PrimeField32;
use p3_field::{AbstractExtensionField, AbstractField, Field};

pub fn poly_eval_val<F: Field>(coeffs: Vec<F>, x: F) -> F {
    let mut result = F::zero();
    for coeff in coeffs.iter() {
        result = result * x + *coeff;
    }
    result
}

pub trait CircuitV2Builder<C: Config> {
    /// evaluate a field polynomial at an extension field element, i.e., y = f(x)
    fn poly_eval_v2(&mut self, coeffs: Vec<Felt<C::F>>, x: Felt<C::F>) -> Felt<C::F>;
    /// evaluate an extension field polynomial at an extension field element, i.e., y = f(x)
    fn ext_poly_eval_v2(
        &mut self,
        coeffs: Vec<Ext<C::F, C::EF>>,
        x: Ext<C::F, C::EF>,
    ) -> Ext<C::F, C::EF>;

    /// extension field exp reverse bits
    fn ext_exp_reverse_bits(
        &mut self,
        base: Ext<C::F, C::EF>,
        power_bits: Vec<Felt<C::F>>,
    ) -> Ext<C::F, C::EF>;

    /// extension field dot product
    fn ext_dot_prod(
        &mut self,
        x_vec: Vec<Ext<C::F, C::EF>>,
        y_vec: Vec<Ext<C::F, C::EF>>,
    ) -> Ext<C::F, C::EF>;

    /// eq poly eval
    fn eq_poly(
        &mut self,
        x_vec: Vec<Ext<C::F, C::EF>>,
        y_vec: Vec<Ext<C::F, C::EF>>,
    ) -> Ext<C::F, C::EF>;

    /// Sumcheck round: verify p(0)+p(1)==claim and evaluate p(challenge).
    fn sumcheck_round(
        &mut self,
        coeffs: Vec<Ext<C::F, C::EF>>,
        challenge: Ext<C::F, C::EF>,
        claim: Ext<C::F, C::EF>,
    ) -> Ext<C::F, C::EF>;

    /// Prefix sum checks: compute product of eq(x1_i, x2_i) starting from init_acc.
    fn prefix_sum_checks(
        &mut self,
        x1_vec: Vec<Ext<C::F, C::EF>>,
        x2_vec: Vec<Ext<C::F, C::EF>>,
        init_acc: Ext<C::F, C::EF>,
    ) -> Ext<C::F, C::EF>;

    fn bits2num_v2_f(
        &mut self,
        bits: impl IntoIterator<Item = Felt<<C as Config>::F>>,
    ) -> Felt<C::F>;
    fn num2bits_v2_f(&mut self, num: Felt<C::F>, num_bits: usize) -> Vec<Felt<C::F>>;
    fn exp_reverse_bits_v2(&mut self, input: Felt<C::F>, power_bits: Vec<Felt<C::F>>)
        -> Felt<C::F>;
    fn batch_fri_v2(
        &mut self,
        alphas: Vec<Ext<C::F, C::EF>>,
        p_at_zs: Vec<Ext<C::F, C::EF>>,
        p_at_xs: Vec<Felt<C::F>>,
    ) -> Ext<C::F, C::EF>;
    fn poseidon2_permute_v2(&mut self, state: [Felt<C::F>; WIDTH]) -> [Felt<C::F>; WIDTH];
    fn poseidon2_hash_v2(&mut self, array: &[Felt<C::F>]) -> [Felt<C::F>; DIGEST_SIZE];
    fn poseidon2_compress_v2(
        &mut self,
        input: impl IntoIterator<Item = Felt<C::F>>,
    ) -> [Felt<C::F>; DIGEST_SIZE];
    fn ext2felt_v2(&mut self, ext: Ext<C::F, C::EF>) -> [Felt<C::F>; D];
    fn add_curve_v2(
        &mut self,
        point1: SepticCurve<Felt<C::F>>,
        point2: SepticCurve<Felt<C::F>>,
    ) -> SepticCurve<Felt<C::F>>;
    fn assert_digest_zero_v2(&mut self, is_real: Felt<C::F>, digest: SepticDigest<Felt<C::F>>);
    fn sum_digest_v2(&mut self, digests: Vec<SepticDigest<Felt<C::F>>>)
        -> SepticDigest<Felt<C::F>>;
    fn commit_public_values_v2(&mut self, public_values: RecursionPublicValues<Felt<C::F>>);
    fn cycle_tracker_v2_enter(&mut self, name: impl Into<Cow<'static, str>>);
    fn cycle_tracker_v2_exit(&mut self);
    fn hint_ext_v2(&mut self) -> Ext<C::F, C::EF>;
    fn hint_felt_v2(&mut self) -> Felt<C::F>;
    fn hint_exts_v2(&mut self, len: usize) -> Vec<Ext<C::F, C::EF>>;
    fn hint_felts_v2(&mut self, len: usize) -> Vec<Felt<C::F>>;
    // fn single_eq_v2(&mut self, lhs: Ext<C::F, C::EF>, rhs: Ext<C::F, C::EF>) -> Ext<C::F, C::EF>;
    // fn evaluate_shift_zero(
    //     &mut self,
    //     x: &Vec<Ext<C::F, C::EF>>,
    //     y: &Vec<Ext<C::F, C::EF>>,
    // ) -> Ext<C::F, C::EF>;
    // fn evaluate_shift_one(
    //     &mut self,
    //     x: &Vec<Ext<C::F, C::EF>>,
    //     y: &Vec<Ext<C::F, C::EF>>,
    // ) -> Ext<C::F, C::EF>;
}

impl<C: Config> CircuitV2Builder<C> for Builder<C> {
    /// evaluate a field polynomial at an extension field element, i.e., y = f(x)
    fn poly_eval_v2(&mut self, coeffs: Vec<Felt<C::F>>, x: Felt<C::F>) -> Felt<C::F> {
        let y: Felt<C::F> = self.uninit();

        self.push_op(DslIr::CircuitV2PolyEval(y, coeffs, x));
        y
    }

    /// evaluate an extension field polynomial at an extension field element, i.e., y = f(x)
    fn ext_poly_eval_v2(
        &mut self,
        coeffs: Vec<Ext<C::F, C::EF>>,
        x: Ext<C::F, C::EF>,
    ) -> Ext<C::F, C::EF> {
        let mut acc: Ext<C::F, C::EF> = self.constant(C::EF::zero());
        for c in coeffs.into_iter() {
            acc = self.eval(acc * x + c);
        }
        acc
    }

    fn ext_exp_reverse_bits(
        &mut self,
        base: Ext<C::F, C::EF>,
        power_bits: Vec<Felt<C::F>>,
    ) -> Ext<C::F, C::EF> {
        let mut result: Ext<C::F, C::EF> = self.uninit();
        #[cfg(feature = "verify")]
        {
            let base_val = base.val;
            let exponent_bits = power_bits.clone().into_iter().map(|bit| bit.val).collect_vec();
            let mut out = C::EF::one();
            for val in exponent_bits.clone() {
                out *= out;
                if val.is_one() {
                    out *= base_val;
                }
            }
            result.val = out;
        }
        self.push_op(DslIr::CircuitV2ExtExpReverseBits(result, base, power_bits));
        result
    }

    fn ext_dot_prod(
        &mut self,
        x_vec: Vec<Ext<<C as Config>::F, <C as Config>::EF>>,
        y_vec: Vec<Ext<<C as Config>::F, <C as Config>::EF>>,
    ) -> Ext<<C as Config>::F, <C as Config>::EF> {
        let mut acc: Ext<C::F, C::EF> = self.constant(C::EF::zero());
        for (x, y) in x_vec.into_iter().zip(y_vec) {
            acc = self.eval(acc + x * y);
        }
        acc
    }

    fn eq_poly(
        &mut self,
        x_vec: Vec<Ext<<C as Config>::F, <C as Config>::EF>>,
        y_vec: Vec<Ext<<C as Config>::F, <C as Config>::EF>>,
    ) -> Ext<<C as Config>::F, <C as Config>::EF> {
        let init_acc = self.constant(<C::EF as AbstractField>::one());
        self.prefix_sum_checks(x_vec, y_vec, init_acc)
    }

    fn sumcheck_round(
        &mut self,
        coeffs: Vec<Ext<C::F, C::EF>>,
        challenge: Ext<C::F, C::EF>,
        claim: Ext<C::F, C::EF>,
    ) -> Ext<C::F, C::EF> {
        let mut result: Ext<C::F, C::EF> = self.uninit();
        #[cfg(feature = "verify")]
        {
            let coeffs_val = coeffs.iter().map(|c| c.val).collect::<Vec<_>>();
            let challenge_val = challenge.val;
            let out = coeffs_val[1..].iter().fold(coeffs_val[0], |acc, &x| acc * challenge_val + x);
            result.val = out;
        }
        self.push_op(DslIr::CircuitV2SumcheckRound(result, coeffs, challenge, claim));
        result
    }

    fn prefix_sum_checks(
        &mut self,
        x1_vec: Vec<Ext<C::F, C::EF>>,
        x2_vec: Vec<Ext<C::F, C::EF>>,
        init_acc: Ext<C::F, C::EF>,
    ) -> Ext<C::F, C::EF> {
        let mut result: Ext<C::F, C::EF> = self.uninit();
        #[cfg(feature = "verify")]
        {
            let x1_vals = x1_vec.iter().map(|x| x.val).collect::<Vec<_>>();
            let x2_vals = x2_vec.iter().map(|x| x.val).collect::<Vec<_>>();
            let mut acc = init_acc.val;
            for (x1, x2) in x1_vals.iter().zip(x2_vals.iter()) {
                let eq_val = *x1 * *x2 + (C::EF::one() - *x1) * (C::EF::one() - *x2);
                acc *= eq_val;
            }
            result.val = acc;
        }
        self.push_op(DslIr::CircuitV2PrefixSumChecks(result, x1_vec, x2_vec));
        result
    }

    fn bits2num_v2_f(
        &mut self,
        bits: impl IntoIterator<Item = Felt<<C as Config>::F>>,
    ) -> Felt<<C as Config>::F> {
        let mut num: Felt<_> = self.eval(C::F::zero());
        for (i, bit) in bits.into_iter().enumerate() {
            // Add `bit * 2^i` to the sum.
            num = self.eval(num + bit * C::F::from_wrapped_u32(1 << i));
        }
        num
    }

    /// Converts a felt to bits inside a circuit.
    fn num2bits_v2_f(&mut self, num: Felt<C::F>, num_bits: usize) -> Vec<Felt<C::F>> {
        let mut output: Vec<Felt<C::F>> =
            std::iter::from_fn(|| Some(self.uninit())).take(num_bits).collect::<Vec<_>>();

        #[cfg(feature = "verify")]
        {
            let mut bits = u32_to_le_bits(num.val.as_canonical_u32());
            bits.resize(num_bits, 0);
            assert_eq!(bits.len(), output.len());
            for i in 0..num_bits {
                output[i].val = C::F::from_canonical_u32(bits[i]);
            }
        }
        self.push_op(DslIr::CircuitV2HintBitsF(output.clone(), num));

        let x: SymbolicFelt<_> = output
            .iter()
            .enumerate()
            .map(|(i, &bit)| {
                self.assert_felt_eq(bit * (bit - C::F::one()), C::F::zero());
                bit * C::F::from_wrapped_u32(1 << i)
            })
            .sum();

        // Range check the bits to be less than the field modulus.
        assert!(num_bits <= 31, "num_bits must be less than or equal to 31");

        // If there are less than 31 bits, there is nothing to check.
        if num_bits > 30 {
            // BabyBear modulus: 2^31 - 2^27 + 1 -> top 4 bits, bottom 27 bits
            // KoalaBear modulus: 2^31 - 2^24 + 1 -> top 7 bits, bottom 24 bits
            //
            // If any of the top bits are zero, the number is within range.
            // If all top bits are 1, all bottom bits must be 0.
            #[cfg(feature = "babybear")]
            const TOP_BITS: usize = 4;
            #[cfg(feature = "koalabear")]
            const TOP_BITS: usize = 7;

            #[cfg(feature = "babybear")]
            const BOTTOM_BITS: usize = 27;
            #[cfg(feature = "koalabear")]
            const BOTTOM_BITS: usize = 24;

            let are_all_top_bits_one: Felt<_> = self.eval(
                output
                    .iter()
                    .rev()
                    .take(TOP_BITS)
                    .copied()
                    .map(SymbolicFelt::from)
                    .product::<SymbolicFelt<_>>(),
            );

            for bit in output.iter().take(BOTTOM_BITS).copied() {
                self.assert_felt_eq(bit * are_all_top_bits_one, C::F::zero());
            }
        }

        // Check that the original number matches the bit decomposition.
        self.assert_felt_eq(x, num);

        output
    }

    /// A version of `exp_reverse_bits_len` that uses ALU ops (square-and-multiply in reverse bit
    /// order).
    fn exp_reverse_bits_v2(&mut self, base: Felt<C::F>, power_bits: Vec<Felt<C::F>>) -> Felt<C::F> {
        let mut acc: Felt<C::F> = self.constant(C::F::one());
        for bit in power_bits.into_iter().rev() {
            let squared: Felt<C::F> = self.eval(acc * acc);
            // when bit=1: acc = squared * base; when bit=0: acc = squared
            let one: Felt<C::F> = self.constant(C::F::one());
            let not_bit: Felt<C::F> = self.eval(one - bit);
            acc = self.eval(squared * base * bit + squared * not_bit);
        }
        acc
    }

    /// A version of the `batch_fri` that uses inline arithmetic.
    fn batch_fri_v2(
        &mut self,
        alpha_pows: Vec<Ext<C::F, C::EF>>,
        p_at_zs: Vec<Ext<C::F, C::EF>>,
        p_at_xs: Vec<Felt<C::F>>,
    ) -> Ext<C::F, C::EF> {
        let mut acc: Ext<C::F, C::EF> = self.constant(C::EF::zero());
        for ((alpha, p_z), p_x) in alpha_pows.into_iter().zip(p_at_zs).zip(p_at_xs) {
            let diff: Ext<C::F, C::EF> = self.eval(p_z - p_x);
            acc = self.eval(acc + alpha * diff);
        }
        acc
    }

    /// Applies the Poseidon2 permutation to the given array.
    fn poseidon2_permute_v2(&mut self, array: [Felt<C::F>; WIDTH]) -> [Felt<C::F>; WIDTH] {
        let output: [Felt<C::F>; WIDTH] = core::array::from_fn(|_| self.uninit());
        #[cfg(feature = "babybear")]
        self.push_op(DslIr::CircuitV2Poseidon2PermuteBabyBear(Box::new((output, array))));
        #[cfg(feature = "koalabear")]
        self.push_op(DslIr::CircuitV2Poseidon2PermuteKoalaBear(Box::new((output, array))));
        output
    }

    /// Applies the Poseidon2 hash function to the given array.
    ///
    /// Reference: [p3_symmetric::PaddingFreeSponge]
    fn poseidon2_hash_v2(&mut self, input: &[Felt<C::F>]) -> [Felt<C::F>; DIGEST_SIZE] {
        // static_assert(RATE < WIDTH)
        let mut state = core::array::from_fn(|_| self.eval(C::F::zero()));
        for input_chunk in input.chunks(HASH_RATE) {
            state[..input_chunk.len()].copy_from_slice(input_chunk);
            state = self.poseidon2_permute_v2(state);
        }
        let state: [Felt<C::F>; DIGEST_SIZE] = state[..DIGEST_SIZE].try_into().unwrap();
        state
    }

    /// Applies the Poseidon2 compression function to the given array.
    ///
    /// Reference: [p3_symmetric::TruncatedPermutation]
    fn poseidon2_compress_v2(
        &mut self,
        input: impl IntoIterator<Item = Felt<C::F>>,
    ) -> [Felt<C::F>; DIGEST_SIZE] {
        // debug_assert!(DIGEST_SIZE * N <= WIDTH);
        let mut pre_iter = input.into_iter().chain(repeat(self.eval(C::F::default())));
        let pre = core::array::from_fn(move |_| pre_iter.next().unwrap());
        let post = self.poseidon2_permute_v2(pre);
        let post: [Felt<C::F>; DIGEST_SIZE] = post[..DIGEST_SIZE].try_into().unwrap();
        post
    }

    /// Decomposes an ext into its felt coordinates.
    fn ext2felt_v2(&mut self, ext: Ext<C::F, C::EF>) -> [Felt<C::F>; D] {
        let mut felts: [Felt<C::F>; D] = core::array::from_fn(|_| self.uninit());
        #[cfg(feature = "verify")]
        {
            let ef = ext.val;
            for i in 0..D {
                felts[i].val = ef.as_base_slice()[i];
            }
        }
        self.push_op(DslIr::CircuitExt2Felt(felts, ext));
        // Verify that the decomposed extension element is correct.
        let mut reconstructed_ext: Ext<C::F, C::EF> = self.constant(C::EF::zero());
        for i in 0..D {
            let felt = felts[i];
            let monomial: Ext<C::F, C::EF> = self.constant(C::EF::monomial(i));
            reconstructed_ext = self.eval(reconstructed_ext + monomial * felt);
        }
        #[cfg(feature = "verify")]
        {
            reconstructed_ext.val = ext.val;
        }

        self.assert_ext_eq(reconstructed_ext, ext);

        felts
    }

    /// Adds two septic elliptic curve points.
    fn add_curve_v2(
        &mut self,
        point1: SepticCurve<Felt<C::F>>,
        point2: SepticCurve<Felt<C::F>>,
    ) -> SepticCurve<Felt<C::F>> {
        // Hint the curve addition result.
        let point_sum_x: [Felt<C::F>; 7] = core::array::from_fn(|_| self.uninit());
        let point_sum_y: [Felt<C::F>; 7] = core::array::from_fn(|_| self.uninit());
        let point =
            SepticCurve { x: SepticExtension(point_sum_x), y: SepticExtension(point_sum_y) };
        self.push_op(DslIr::CircuitV2HintAddCurve(Box::new((point, point1, point2))));

        // Convert each point into a point over SymbolicFelt.
        let point1_symbolic = SepticCurve::convert(point1, |x| x.into());
        let point2_symbolic = SepticCurve::convert(point2, |x| x.into());
        let point_symbolic = SepticCurve::convert(point, |x| x.into());

        // Evaluate `sum_checker_x` and `sum_checker_y`.
        let sum_checker_x = SepticCurve::<SymbolicFelt<C::F>>::sum_checker_x(
            point1_symbolic,
            point2_symbolic,
            point_symbolic,
        );

        let sum_checker_y = SepticCurve::<SymbolicFelt<C::F>>::sum_checker_y(
            point1_symbolic,
            point2_symbolic,
            point_symbolic,
        );

        // Constrain `sum_checker_x` and `sum_checker_y` to be all zero.
        for limb in sum_checker_x.0 {
            self.assert_felt_eq(limb, C::F::zero());
        }

        for limb in sum_checker_y.0 {
            self.assert_felt_eq(limb, C::F::zero());
        }

        point
    }

    /// Asserts that the `digest` is the zero digest when `is_real` is non-zero.
    fn assert_digest_zero_v2(&mut self, is_real: Felt<C::F>, digest: SepticDigest<Felt<C::F>>) {
        let zero = SepticDigest::<SymbolicFelt<C::F>>::zero();
        for (digest_limb_x, zero_limb_x) in digest.0.x.0.into_iter().zip_eq(zero.0.x.0.into_iter())
        {
            self.assert_felt_eq(is_real * digest_limb_x, is_real * zero_limb_x);
        }
        for (digest_limb_y, zero_limb_y) in digest.0.y.0.into_iter().zip_eq(zero.0.y.0.into_iter())
        {
            self.assert_felt_eq(is_real * digest_limb_y, is_real * zero_limb_y);
        }
    }

    // Sums the digests into one.
    fn sum_digest_v2(
        &mut self,
        digests: Vec<SepticDigest<Felt<C::F>>>,
    ) -> SepticDigest<Felt<C::F>> {
        let mut convert_to_felt =
            |point: SepticCurve<C::F>| SepticCurve::convert(point, |value| self.eval(value));

        let start = convert_to_felt(SepticDigest::<C::F>::starting_digest_for_field().0);
        let zero_digest = convert_to_felt(SepticDigest::<C::F>::zero_for_field().0);

        if digests.is_empty() {
            return SepticDigest(zero_digest);
        }

        let neg_start = convert_to_felt(SepticDigest::<C::F>::starting_digest_for_field().0.neg());
        let neg_zero_digest = convert_to_felt(SepticDigest::<C::F>::zero_for_field().0.neg());

        let mut ret = start;
        for (i, digest) in digests.clone().into_iter().enumerate() {
            ret = self.add_curve_v2(ret, digest.0);
            if i != digests.len() - 1 {
                ret = self.add_curve_v2(ret, neg_zero_digest)
            }
        }
        SepticDigest(self.add_curve_v2(ret, neg_start))
    }

    // Commits public values.
    fn commit_public_values_v2(&mut self, public_values: RecursionPublicValues<Felt<C::F>>) {
        self.push_op(DslIr::CircuitV2CommitPublicValues(Box::new(public_values)));
    }

    fn cycle_tracker_v2_enter(&mut self, name: impl Into<Cow<'static, str>>) {
        self.push_op(DslIr::CycleTrackerV2Enter(name.into()));
    }

    fn cycle_tracker_v2_exit(&mut self) {
        self.push_op(DslIr::CycleTrackerV2Exit);
    }

    /// Hint a single felt.
    fn hint_felt_v2(&mut self) -> Felt<C::F> {
        self.hint_felts_v2(1)[0]
    }

    /// Hint a single ext.
    fn hint_ext_v2(&mut self) -> Ext<C::F, C::EF> {
        self.hint_exts_v2(1)[0]
    }

    /// Hint a vector of felts.
    fn hint_felts_v2(&mut self, len: usize) -> Vec<Felt<C::F>> {
        let arr = std::iter::from_fn(|| Some(self.uninit())).take(len).collect::<Vec<_>>();
        self.push_op(DslIr::CircuitV2HintFelts(arr[0], len));
        arr
    }

    /// Hint a vector of exts.
    fn hint_exts_v2(&mut self, len: usize) -> Vec<Ext<C::F, C::EF>> {
        let arr = std::iter::from_fn(|| Some(self.uninit())).take(len).collect::<Vec<_>>();
        self.push_op(DslIr::CircuitV2HintExts(arr[0], len));
        arr
    }
}
