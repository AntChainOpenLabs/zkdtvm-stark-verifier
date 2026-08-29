use crate::{
    sumcheck::{
        pcs::{EF, F},
        types::UniPolyVariable,
        SCBabyBearFriConfigVariable,
    },
    CircuitConfig,
};
use dt_recursion_compiler::{
    circuit::CircuitV2Builder,
    ir::{Builder, Ext, SymbolicExt},
};
use itertools::Itertools;
use p3_field::{AbstractExtensionField, AbstractField, Field};
use std::{collections::BTreeMap, marker::PhantomData};

pub struct Utils<C, SC> {
    _marker: PhantomData<(C, SC)>,
}

impl<C: CircuitConfig<F = SC::Val>, SC: SCBabyBearFriConfigVariable<C>> Utils<C, SC>
where
    Builder<C>: CircuitV2Builder<C>,
{
    pub(crate) fn extend_challenges_with_skips(
        builder: &mut Builder<C>,
        challenges: &[Ext<C::F, C::EF>],
        num_skip_rounds: usize,
        _chip_log_height_threshold: usize,
    ) -> Vec<Ext<C::F, C::EF>> {
        let m = 1usize << num_skip_rounds;

        // Precompute inv_denom as field constants -- zero builder cost.
        let inv_denom: Vec<C::EF> = (0..m)
            .map(|i| {
                let mut d = C::EF::one();
                for j in 0..m {
                    if j != i {
                        d *= C::EF::from_canonical_usize(i) - C::EF::from_canonical_usize(j);
                    }
                }
                d.inverse()
            })
            .collect();

        let mut out = Vec::with_capacity(challenges.len() * num_skip_rounds);

        for &y in challenges.iter() {
            // delta[j] = y - j
            let delta: Vec<Ext<C::F, C::EF>> = (0..m)
                .map(|j| if j == 0 { y } else { builder.eval(y - C::EF::from_canonical_usize(j)) })
                .collect();

            // Prefix products: prefix[i] = ∏_{j=0}^{i} delta[j]
            let mut prefix = Vec::with_capacity(m);
            prefix.push(delta[0]);
            for j in 1..m {
                prefix.push(builder.eval(prefix[j - 1] * delta[j]));
            }

            // Suffix products: suffix[i] = ∏_{j=i}^{m-1} delta[j]
            let mut suffix = vec![delta[0]; m];
            suffix[m - 1] = delta[m - 1];
            for j in (0..m - 1).rev() {
                suffix[j] = builder.eval(suffix[j + 1] * delta[j]);
            }

            // product_excluding[i] = ∏_{j≠i} delta[j]
            let product_excluding: Vec<Ext<C::F, C::EF>> = (0..m)
                .map(|i| {
                    if i == 0 {
                        suffix[1]
                    } else if i == m - 1 {
                        prefix[m - 2]
                    } else {
                        builder.eval(prefix[i - 1] * suffix[i + 1])
                    }
                })
                .collect();

            // L_i(y) = product_excluding[i] * inv_denom[i]
            let lvals: Vec<Ext<C::F, C::EF>> =
                (0..m).map(|i| builder.eval(product_excluding[i] * inv_denom[i])).collect();

            // M_t(y) = Σ_{i : bit t of i = 1} L_i(y)
            for t in 0..num_skip_rounds {
                let bit_index = t;
                let mut acc: Option<Ext<C::F, C::EF>> = None;
                for i in 0..m {
                    if ((i >> bit_index) & 1) == 1 {
                        acc = Some(match acc {
                            None => lvals[i],
                            Some(a) => builder.eval(a + lvals[i]),
                        });
                    }
                }
                out.push(acc.unwrap());
            }
        }
        out
    }

    pub(crate) fn calculate_eq(
        builder: &mut Builder<C>,
        a: Ext<C::F, C::EF>,
        b: Ext<C::F, C::EF>,
        degree: usize,
    ) -> Ext<C::F, C::EF> {
        let len = degree + 1;
        let roots: Vec<C::EF> = (0..len).map(C::EF::from_canonical_usize).collect();
        let prods_a = Self::product_excluding_all(builder, a, degree);
        let prods_b = Self::product_excluding_all(builder, b, degree);
        let ret_symb = roots
            .iter()
            .enumerate()
            .map(|(i, r)| {
                let mut denominator: C::EF =
                    (0..len).filter(|j| *j != i).map(|j| *r - roots[j]).product::<C::EF>();
                denominator = denominator.square();
                let numerator: Ext<_, _> = builder.eval(prods_a[i] * prods_b[i]);
                numerator * denominator.inverse()
            })
            .sum::<SymbolicExt<C::F, C::EF>>();
        builder.eval(ret_symb)
    }

    /// Compute ∏_{j≠i, j∈[0,degree]} (value − j) for ALL i in O(degree) builder ops
    /// using prefix/suffix products, instead of O(degree²).
    fn product_excluding_all(
        builder: &mut Builder<C>,
        value: Ext<C::F, C::EF>,
        degree: usize,
    ) -> Vec<Ext<C::F, C::EF>> {
        let len = degree + 1;
        if len == 1 {
            return vec![builder.constant(C::EF::one())];
        }

        let delta: Vec<Ext<C::F, C::EF>> =
            (0..len)
                .map(|j| {
                    if j == 0 {
                        value
                    } else {
                        builder.eval(value - C::EF::from_canonical_usize(j))
                    }
                })
                .collect();

        let mut prefix = Vec::with_capacity(len);
        prefix.push(delta[0]);
        for j in 1..len {
            prefix.push(builder.eval(prefix[j - 1] * delta[j]));
        }

        let mut suffix = vec![delta[0]; len];
        suffix[len - 1] = delta[len - 1];
        for j in (0..len - 1).rev() {
            suffix[j] = builder.eval(suffix[j + 1] * delta[j]);
        }

        (0..len)
            .map(|i| {
                if i == 0 {
                    suffix[1]
                } else if i == len - 1 {
                    prefix[len - 2]
                } else {
                    builder.eval(prefix[i - 1] * suffix[i + 1])
                }
            })
            .collect()
    }

    pub(crate) fn compute_eq_sum(
        builder: &mut Builder<C>,
        extend_round_challenge: EF<C>,
        open_points: &[EF<C>],
        sc_challenges: &[EF<C>],
    ) -> EF<C> {
        let one: Ext<_, _> = builder.constant(C::EF::one());
        let zero_eq: Ext<_, _> = builder.eval(one - extend_round_challenge);
        let one_eq: Ext<_, _> = extend_round_challenge;

        //for shift 0
        // let mut s_ind_p_zero: Ext<_, _> = builder.constant(C::EF::one());
        // let mut s_ind_pp_zero: Ext<_, _> = builder.constant(C::EF::zero());
        //for shift 1
        let mut s_ind_p_one: Ext<_, _> = builder.constant(C::EF::one());
        let mut s_ind_pp_one: Ext<_, _> = builder.constant(C::EF::zero());
        let max_index = open_points.len() - 1;
        //first point
        {
            let product: Ext<_, _> =
                builder.eval(open_points[max_index] * sc_challenges[max_index]);
            let eq_xy: Ext<_, _> = builder
                .eval((one - open_points[max_index]) * (one - sc_challenges[max_index]) + product);
            //for shift one, special case
            let temp_p_one: Ext<_, _> =
                builder.eval((sc_challenges[max_index] - product) * s_ind_p_one);
            let temp_pp_one: Ext<_, _> = builder
                .eval((open_points[max_index] - product) * s_ind_p_one + eq_xy * s_ind_pp_one);
            s_ind_p_one = temp_p_one;
            s_ind_pp_one = temp_pp_one;
            //for shift zero
            // let temp_pp_zero: Ext<_, _> =
            //     builder.eval((open_points[max_index] - product) * s_ind_pp_zero);
            // let temp_p_zero: Ext<_, _> = builder.eval(eq_xy * s_ind_p_zero); //+
            // (sc_challenges[max_index] - product) * s_ind_pp_zero); s_ind_p_zero =
            // temp_p_zero; s_ind_pp_zero = temp_pp_zero;
        }

        for i in 1..open_points.len() {
            let k = max_index - i;
            let product: Ext<_, _> = builder.eval(open_points[k] * sc_challenges[k]);
            let eq_xy: Ext<_, _> =
                builder.eval((one - open_points[k]) * (one - sc_challenges[k]) + product);
            let x_minus_product: Ext<_, _> = builder.eval(open_points[k] - product);
            let y_minus_product: Ext<_, _> = builder.eval(sc_challenges[k] - product);
            //for shift zero
            // let temp_pp_zero: Ext<_, _> = builder.eval(x_minus_product * s_ind_pp_zero);
            // let temp_p_zero: Ext<_, _> = builder.eval(eq_xy * s_ind_p_zero); // + y_minus_product
            // * s_ind_pp_zero); s_ind_p_zero = temp_p_zero;
            // s_ind_pp_zero = temp_pp_zero;
            //for shift one
            let temp_pp_one: Ext<_, _> = builder.eval(x_minus_product * s_ind_pp_one);
            let temp_p_one: Ext<_, _> =
                builder.eval(eq_xy * s_ind_p_one + y_minus_product * s_ind_pp_one);
            s_ind_p_one = temp_p_one;
            s_ind_pp_one = temp_pp_one;
        }

        let s_ind_p_zero = C::eq_poly(builder, sc_challenges.to_vec(), open_points.to_vec());

        let eval_one: Ext<_, _> = builder.eval(s_ind_p_one + s_ind_pp_one);
        // let eval_zero: Ext<_, _> = builder.eval(s_ind_p_zero + s_ind_pp_zero);
        builder.eval(eval_one * one_eq + s_ind_p_zero * zero_eq)
    }
    pub fn compute_eq_alpha_r_vec(
        builder: &mut Builder<C>,
        alpha_vec: Vec<EF<C>>,
        r_nomal_vec: &Vec<EF<C>>,
        num_vars: usize,
    ) -> Vec<EF<C>> {
        assert_eq!(num_vars, alpha_vec.len());
        let mut result = Vec::with_capacity(num_vars);

        for i in 0..num_vars {
            // let current_alpha = alpha_vec[i];

            // let alpha_eq_poly = EqPolyAlphaVariable::new(current_alpha);
            // let eq_alpha_r = alpha_eq_poly.evaluate(builder, &r_nomal_vec[i..]);
            // result.push(eq_alpha_r);
            let mut alpha = alpha_vec[i];
            let mut alpha_vec = vec![];
            for i in i..r_nomal_vec.len() {
                alpha_vec.push(alpha);
                alpha = builder.eval(alpha * alpha);
            }
            result.push(C::eq_poly(builder, alpha_vec, r_nomal_vec[i..].to_vec()));
        }
        result
    }
    pub fn compute_combined_eq_sum_vec(
        builder: &mut Builder<C>,
        open_points: &[EF<C>],
        sc_challenges: &[EF<C>],
        log_height_extended_rs: &BTreeMap<usize, EF<C>>,
        num_vars: usize,
    ) -> Vec<EF<C>> {
        log_height_extended_rs
            .iter()
            .rev()
            .map(|(log_height, extend_round_challenge)| {
                Self::compute_eq_sum(
                    builder,
                    *extend_round_challenge,
                    &open_points[(num_vars - log_height)..],
                    &sc_challenges[(num_vars - log_height)..],
                )
            })
            .collect::<Vec<_>>()
    }

    pub fn compute_combined_eq_sum_vec_with_univariate_skip(
        builder: &mut Builder<C>,
        open_points: &[&[EF<C>]],
        sc_challenges: &[EF<C>],
        log_height_extended_rs: &BTreeMap<usize, EF<C>>,
        num_vars: usize,
    ) -> Vec<EF<C>> {
        log_height_extended_rs
            .iter()
            .rev()
            .enumerate()
            .map(|(idx, (log_height, extend_round_challenge))| {
                Self::compute_eq_sum(
                    builder,
                    *extend_round_challenge,
                    open_points[idx],
                    &sc_challenges[(num_vars - log_height)..],
                )
            })
            .collect::<Vec<_>>()
    }

    pub fn compute_eq(builder: &mut Builder<C>, u: &[EF<C>], v: &[EF<C>]) -> EF<C> {
        assert!(u.len() == v.len());
        let n = u.len();
        let one: EF<C> = builder.constant(C::EF::one());
        let zero: EF<C> = builder.constant(C::EF::zero());
        let mut result: EF<C> = one;
        for i in 0..n {
            let product: EF<C> = builder.eval(u[i] * v[i]);
            let term: EF<C> = builder.eval((one - u[i]) * (one - v[i]) + product);
            result = builder.eval(result * term);
        }
        result
    }

    pub fn compute_combined_f_r(
        builder: &mut Builder<C>,
        combined_eq_sum_vec: Vec<EF<C>>,
        combined_f_shift: EF<C>,
        eq_alpha_r_vec: Vec<EF<C>>,
        gamma_vec: Vec<EF<C>>,
        rs_extend_vec: &Vec<EF<C>>,
        log_heights: Vec<usize>,
        num_vars: usize,
    ) -> EF<C> {
        let mut current_combined_eq_sum = combined_eq_sum_vec[0];
        let mut index_indicator = 1;
        for (idx, eq_alpha_r) in eq_alpha_r_vec.iter().enumerate() {
            let gamma_mul_eq_alpha: Ext<_, _> = builder.eval(gamma_vec[idx] * *eq_alpha_r);
            current_combined_eq_sum = builder.eval(current_combined_eq_sum + gamma_mul_eq_alpha);

            if index_indicator < log_heights.len() &&
                (num_vars - 1 - idx) == log_heights[index_indicator]
            {
                let temp: Ext<_, _> =
                    builder.eval(combined_eq_sum_vec[index_indicator] - current_combined_eq_sum);
                current_combined_eq_sum = builder
                    .eval(temp * rs_extend_vec[index_indicator - 1] + current_combined_eq_sum);
                index_indicator += 1;
            }
        }

        builder.eval(combined_f_shift / current_combined_eq_sum)
    }

    /// for single log_height
    pub fn compute_round_extend_claim(
        builder: &mut Builder<C>,
        beta: EF<C>,
        // matrices openings for current log_height
        openings: Vec<Vec<EF<C>>>,
    ) -> EF<C> {
        let coeffs = openings.into_iter().flatten().collect::<Vec<_>>();
        let unipoly = UniPolyVariable { evals: coeffs };
        unipoly.evaluate_horner(builder, &beta)
    }

    pub fn compute_dotproduct_mix(builder: &mut Builder<C>, a: &[EF<C>], b: &[F<C>]) -> EF<C> {
        let res: EF<C>;
        if a.len() == b.len() {
            let res_symbolic = a
                .iter()
                .zip(b.iter())
                .map(|(&a_i, &b_i)| a_i * b_i)
                .sum::<SymbolicExt<C::F, C::EF>>();
            res = builder.eval(res_symbolic);
        } else {
            // res = a
            //     .iter()
            //     .zip(b.chunks(C::EF::D))
            //     .map(|(&a_i, b_chunk)| a_i * builder.ext_from_base_slice(b_chunk))
            //     .sum::<SymbolicExt<C::F, C::EF>>();
            let ext_b = b
                .chunks(C::EF::D)
                .map(|b_chunk| builder.ext_from_base_slice(b_chunk))
                .collect_vec();
            res = builder.ext_dot_prod(a.to_vec(), ext_b);
        }
        res
    }
}
