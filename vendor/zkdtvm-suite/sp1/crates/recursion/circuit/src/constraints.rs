use std::ops::MulAssign;

use p3_air::{
    Air, AirBuilder, AirBuilderWithPublicValues, BaseAir, ExtensionBuilder, PairBuilder,
    PermutationAirBuilder,
};
use p3_baby_bear::BabyBear;
use p3_commit::{LagrangeSelectors, Mmcs, PolynomialSpace, TwoAdicMultiplicativeCoset};
use p3_field::{AbstractExtensionField, AbstractField, ExtensionField, Field, TwoAdicField};
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};

use dt_recursion_compiler::ir::{
    Builder, Config, Ext, ExtConst, ExtensionOperand, Felt, SymbolicExt, SymbolicFelt,
};
use dt_stark::{
    air::{EmptyMessageBuilder, MachineAir, MultiTableAirBuilder},
    AirOpenedValues, Challenge, ChipOpenedValues, GenericVerifierConstraintFolder, MachineChip,
    OpeningShapeError,
};

use crate::{
    domain::PolynomialSpaceVariable, stark::StarkVerifier, BabyBearFriConfigVariable, CircuitConfig,
};

pub type RecursiveVerifierConstraintFolder<'a, C> = GenericVerifierConstraintFolder<
    'a,
    <C as Config>::F,
    <C as Config>::EF,
    Felt<<C as Config>::F>,
    Ext<<C as Config>::F, <C as Config>::EF>,
    SymbolicExt<<C as Config>::F, <C as Config>::EF>,
>;

/// Constraint folder for the recursion circuit's sumcheck verifier path.
///
/// Mirrors the native `SumcheckVerifierConstraintFolder` but uses circuit symbolic types.
/// Unlike `RecursiveVerifierConstraintFolder` (alias for `GenericVerifierConstraintFolder`),
/// this folder uses single-row `RowMajorMatrixView` instead of two-row `VerticalPair`, and
/// has no `is_transition` field.
pub struct RecursiveSumcheckConstraintFolder<'a, C: Config> {
    pub preprocessed: RowMajorMatrixView<'a, Ext<C::F, C::EF>>,
    pub main: RowMajorMatrixView<'a, Ext<C::F, C::EF>>,
    pub permutation: RowMajorMatrixView<'a, Ext<C::F, C::EF>>,
    pub perm_challenges: &'a [Ext<C::F, C::EF>],
    pub local_cumulative_sum: &'a Ext<C::F, C::EF>,
    pub is_first_row: Ext<C::F, C::EF>,
    pub is_last_row: Ext<C::F, C::EF>,
    pub alpha: Ext<C::F, C::EF>,
    pub accumulator: SymbolicExt<C::F, C::EF>,
    pub public_values: &'a [Felt<C::F>],
}

impl<'a, C: Config> AirBuilder for RecursiveSumcheckConstraintFolder<'a, C>
where
    C::F: Field,
    C::EF: ExtensionField<C::F>,
    Ext<C::F, C::EF>: Into<SymbolicExt<C::F, C::EF>>
        + Copy
        + std::ops::Add<C::F, Output = SymbolicExt<C::F, C::EF>>
        + std::ops::Add<Ext<C::F, C::EF>, Output = SymbolicExt<C::F, C::EF>>
        + std::ops::Add<SymbolicExt<C::F, C::EF>, Output = SymbolicExt<C::F, C::EF>>
        + std::ops::Sub<C::F, Output = SymbolicExt<C::F, C::EF>>
        + std::ops::Sub<Ext<C::F, C::EF>, Output = SymbolicExt<C::F, C::EF>>
        + std::ops::Sub<SymbolicExt<C::F, C::EF>, Output = SymbolicExt<C::F, C::EF>>
        + std::ops::Mul<C::F, Output = SymbolicExt<C::F, C::EF>>
        + std::ops::Mul<Ext<C::F, C::EF>, Output = SymbolicExt<C::F, C::EF>>
        + std::ops::Mul<SymbolicExt<C::F, C::EF>, Output = SymbolicExt<C::F, C::EF>>
        + Send
        + Sync,
    SymbolicExt<C::F, C::EF>: AbstractField
        + From<C::F>
        + std::ops::Add<Ext<C::F, C::EF>, Output = SymbolicExt<C::F, C::EF>>
        + std::ops::Add<C::F, Output = SymbolicExt<C::F, C::EF>>
        + std::ops::Sub<Ext<C::F, C::EF>, Output = SymbolicExt<C::F, C::EF>>
        + std::ops::Sub<C::F, Output = SymbolicExt<C::F, C::EF>>
        + std::ops::Mul<Ext<C::F, C::EF>, Output = SymbolicExt<C::F, C::EF>>
        + std::ops::Mul<C::F, Output = SymbolicExt<C::F, C::EF>>
        + MulAssign<C::EF>,
    Felt<C::F>: Into<SymbolicExt<C::F, C::EF>> + Copy,
{
    type F = C::F;
    type Expr = SymbolicExt<C::F, C::EF>;
    type Var = Ext<C::F, C::EF>;
    type M = RowMajorMatrixView<'a, Ext<C::F, C::EF>>;

    fn main(&self) -> Self::M {
        self.main
    }

    fn is_first_row(&self) -> Self::Expr {
        self.is_first_row.into()
    }

    fn is_last_row(&self) -> Self::Expr {
        self.is_last_row.into()
    }

    fn is_transition_window(&self, size: usize) -> Self::Expr {
        panic!("Sumcheck does not support transition windows (requested size: {size})");
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        let x: SymbolicExt<C::F, C::EF> = x.into();
        self.accumulator *= self.alpha.into();
        self.accumulator += x;
    }
}

impl<C: Config> ExtensionBuilder for RecursiveSumcheckConstraintFolder<'_, C>
where
    C::F: Field,
    C::EF: ExtensionField<C::F>,
    Ext<C::F, C::EF>: Into<SymbolicExt<C::F, C::EF>>
        + Copy
        + std::ops::Add<C::F, Output = SymbolicExt<C::F, C::EF>>
        + std::ops::Add<Ext<C::F, C::EF>, Output = SymbolicExt<C::F, C::EF>>
        + std::ops::Add<SymbolicExt<C::F, C::EF>, Output = SymbolicExt<C::F, C::EF>>
        + std::ops::Sub<C::F, Output = SymbolicExt<C::F, C::EF>>
        + std::ops::Sub<Ext<C::F, C::EF>, Output = SymbolicExt<C::F, C::EF>>
        + std::ops::Sub<SymbolicExt<C::F, C::EF>, Output = SymbolicExt<C::F, C::EF>>
        + std::ops::Mul<C::F, Output = SymbolicExt<C::F, C::EF>>
        + std::ops::Mul<Ext<C::F, C::EF>, Output = SymbolicExt<C::F, C::EF>>
        + std::ops::Mul<SymbolicExt<C::F, C::EF>, Output = SymbolicExt<C::F, C::EF>>
        + Send
        + Sync,
    SymbolicExt<C::F, C::EF>: AbstractField<F = C::EF>
        + From<C::F>
        + std::ops::Add<Ext<C::F, C::EF>, Output = SymbolicExt<C::F, C::EF>>
        + std::ops::Add<C::F, Output = SymbolicExt<C::F, C::EF>>
        + std::ops::Sub<Ext<C::F, C::EF>, Output = SymbolicExt<C::F, C::EF>>
        + std::ops::Sub<C::F, Output = SymbolicExt<C::F, C::EF>>
        + std::ops::Mul<Ext<C::F, C::EF>, Output = SymbolicExt<C::F, C::EF>>
        + std::ops::Mul<C::F, Output = SymbolicExt<C::F, C::EF>>
        + MulAssign<C::EF>,
    Felt<C::F>: Into<SymbolicExt<C::F, C::EF>> + Copy,
{
    type EF = C::EF;
    type ExprEF = SymbolicExt<C::F, C::EF>;
    type VarEF = Ext<C::F, C::EF>;

    fn assert_zero_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>,
    {
        self.assert_zero(x);
    }
}

impl<'a, C: Config> PairBuilder for RecursiveSumcheckConstraintFolder<'a, C>
where
    Self: AirBuilder<M = RowMajorMatrixView<'a, Ext<C::F, C::EF>>>,
{
    fn preprocessed(&self) -> Self::M {
        self.preprocessed
    }
}

impl<'a, C: Config> PermutationAirBuilder for RecursiveSumcheckConstraintFolder<'a, C>
where
    Self: AirBuilder<Var = Ext<C::F, C::EF>, M = RowMajorMatrixView<'a, Ext<C::F, C::EF>>>,
{
    type MP = RowMajorMatrixView<'a, Ext<C::F, C::EF>>;
    type RandomVar = Ext<C::F, C::EF>;

    fn permutation(&self) -> Self::MP {
        self.permutation
    }

    fn permutation_randomness(&self) -> &[Self::RandomVar] {
        self.perm_challenges
    }
}

impl<'a, C: Config> MultiTableAirBuilder<'a> for RecursiveSumcheckConstraintFolder<'a, C>
where
    Self: PermutationAirBuilder<Expr = SymbolicExt<C::F, C::EF>, ExprEF = SymbolicExt<C::F, C::EF>>,
    Ext<C::F, C::EF>: Into<SymbolicExt<C::F, C::EF>> + Copy,
    Felt<C::F>: Into<SymbolicExt<C::F, C::EF>> + Copy,
{
    type LocalSum = Ext<C::F, C::EF>;
    fn local_cumulative_sum(&self) -> &'a Self::LocalSum {
        self.local_cumulative_sum
    }

}

impl<C: Config> EmptyMessageBuilder for RecursiveSumcheckConstraintFolder<'_, C> where
    Self: AirBuilder
{
}

impl<C: Config> AirBuilderWithPublicValues for RecursiveSumcheckConstraintFolder<'_, C>
where
    Self: AirBuilder<Expr = SymbolicExt<C::F, C::EF>>,
    Felt<C::F>: Into<SymbolicExt<C::F, C::EF>> + Copy,
{
    type PublicVar = Felt<C::F>;

    fn public_values(&self) -> &[Self::PublicVar] {
        self.public_values
    }
}

impl<C, SC, A> StarkVerifier<C, SC, A>
where
    C::F: TwoAdicField,
    SC: BabyBearFriConfigVariable<C>,
    C: CircuitConfig<F = SC::Val>,
    <SC::ValMmcs as Mmcs<BabyBear>>::ProverData<RowMajorMatrix<BabyBear>>: Clone,
    A: MachineAir<C::F> + for<'a> Air<RecursiveVerifierConstraintFolder<'a, C>>,
{
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)]
    pub fn verify_constraints(
        builder: &mut Builder<C>,
        chip: &MachineChip<SC, A>,
        opening: &ChipOpenedValues<Felt<C::F>, Ext<C::F, C::EF>>,
        trace_domain: TwoAdicMultiplicativeCoset<C::F>,
        qc_domains: Vec<TwoAdicMultiplicativeCoset<C::F>>,
        zeta: Ext<C::F, C::EF>,
        alpha: Ext<C::F, C::EF>,
        permutation_challenges: &[Ext<C::F, C::EF>],
        public_values: &[Felt<C::F>],
    ) {
        let sels = trace_domain.selectors_at_point_variable(builder, zeta);

        // Recompute the quotient at zeta from the chunks.
        let quotient = Self::recompute_quotient(builder, opening, &qc_domains, zeta);

        // Calculate the evaluations of the constraints at zeta.
        let folded_constraints = Self::eval_constraints(
            builder,
            chip,
            opening,
            &sels,
            alpha,
            permutation_challenges,
            public_values,
        );

        // Assert that the quotient times the zerofier is equal to the folded constraints.
        builder.assert_ext_eq(folded_constraints * sels.inv_zeroifier, quotient);
    }

    #[allow(clippy::type_complexity)]
    pub fn eval_constraints(
        builder: &mut Builder<C>,
        chip: &MachineChip<SC, A>,
        opening: &ChipOpenedValues<Felt<C::F>, Ext<C::F, C::EF>>,
        selectors: &LagrangeSelectors<Ext<C::F, C::EF>>,
        alpha: Ext<C::F, C::EF>,
        permutation_challenges: &[Ext<C::F, C::EF>],
        public_values: &[Felt<C::F>],
    ) -> Ext<C::F, C::EF> {
        let mut unflatten = |v: &[Ext<C::F, C::EF>]| {
            v.chunks_exact(<Challenge<SC> as AbstractExtensionField<C::F>>::D)
                .map(|chunk| {
                    builder.eval(
                        chunk
                            .iter()
                            .enumerate()
                            .map(
                                |(e_i, x): (usize, &Ext<C::F, C::EF>)| -> SymbolicExt<C::F, C::EF> {
                                    SymbolicExt::from(*x) * C::EF::monomial(e_i)
                                },
                            )
                            .sum::<SymbolicExt<_, _>>(),
                    )
                })
                .collect::<Vec<Ext<_, _>>>()
        };
        let perm_opening = AirOpenedValues {
            local: unflatten(&opening.permutation.local),
            next: unflatten(&opening.permutation.next),
        };

        let mut folder = RecursiveVerifierConstraintFolder::<C> {
            preprocessed: opening.preprocessed.view(),
            main: opening.main.view(),
            perm: perm_opening.view(),
            perm_challenges: permutation_challenges,
            local_cumulative_sum: &opening.local_cumulative_sum,
            public_values,
            is_first_row: selectors.is_first_row,
            is_last_row: selectors.is_last_row,
            is_transition: selectors.is_transition,
            alpha,
            accumulator: SymbolicExt::zero(),
            _marker: std::marker::PhantomData,
        };

        chip.eval(&mut folder);
        builder.eval(folder.accumulator)
    }

    #[allow(clippy::type_complexity)]
    pub fn recompute_quotient(
        builder: &mut Builder<C>,
        opening: &ChipOpenedValues<Felt<C::F>, Ext<C::F, C::EF>>,
        qc_domains: &[TwoAdicMultiplicativeCoset<C::F>],
        zeta: Ext<C::F, C::EF>,
    ) -> Ext<C::F, C::EF> {
        // Compute the maximum power of zeta we will need.
        let max_domain_log_n = qc_domains.iter().map(|d| d.log_n).max().unwrap();

        // Compute all powers of zeta of the form zeta^(2^i) up to `zeta^(2^max_domain_log_n)`.
        let mut zetas: Vec<Ext<_, _>> = vec![zeta];
        for _ in 1..max_domain_log_n + 1 {
            let last_zeta = zetas.last().unwrap();
            let new_zeta = builder.eval(*last_zeta * *last_zeta);
            builder.reduce_e(new_zeta);
            zetas.push(new_zeta);
        }
        let zps = qc_domains
            .iter()
            .enumerate()
            .map(|(i, domain)| {
                let (zs, zinvs) = qc_domains
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .map(|(_, other_domain)| {
                        // `shift_power` is used in the computation of
                        let shift_power =
                            other_domain.shift.exp_power_of_2(other_domain.log_n).inverse();
                        // This is `other_domain.zp_at_point_f(builder, domain.first_point())`.
                        // We compute it as a constant here.
                        let z_f = domain.first_point().exp_power_of_2(other_domain.log_n) *
                            shift_power -
                            C::F::one();
                        (
                            {
                                // We use the precomputed powers of zeta to compute (inline) the
                                // value of `other_domain.
                                // zp_at_point_variable(builder, zeta)`.
                                let z: Ext<_, _> = builder.eval(
                                    zetas[other_domain.log_n] * SymbolicFelt::from_f(shift_power) -
                                        SymbolicExt::from_f(C::EF::one()),
                                );
                                z.to_operand().symbolic()
                            },
                            builder.constant::<Felt<_>>(z_f),
                        )
                    })
                    .unzip::<_, _, Vec<SymbolicExt<C::F, C::EF>>, Vec<Felt<_>>>();
                let symbolic_prod: SymbolicFelt<_> =
                    zinvs.into_iter().map(|x| x.into()).product::<SymbolicFelt<_>>();
                (zs.into_iter().product::<SymbolicExt<_, _>>(), symbolic_prod)
            })
            .collect::<Vec<(SymbolicExt<_, _>, SymbolicFelt<_>)>>()
            .into_iter()
            .map(|(x, y)| builder.eval(x / y))
            .collect::<Vec<Ext<_, _>>>();
        zps.iter().for_each(|zp| builder.reduce_e(*zp));
        builder.eval(
            opening
                .quotient
                .iter()
                .enumerate()
                .map(|(ch_i, ch)| {
                    assert_eq!(ch.len(), C::EF::D);
                    zps[ch_i].to_operand().symbolic() *
                        ch.iter()
                            .enumerate()
                            .map(|(e_i, &c)| C::EF::monomial(e_i).cons() * SymbolicExt::from(c))
                            .sum::<SymbolicExt<_, _>>()
                })
                .sum::<SymbolicExt<_, _>>(),
        )
    }

    #[allow(clippy::type_complexity)]
    pub fn verify_opening_shape(
        chip: &MachineChip<SC, A>,
        opening: &ChipOpenedValues<Felt<C::F>, Ext<C::F, C::EF>>,
    ) -> Result<(), OpeningShapeError> {
        // Verify that the preprocessed width matches the expected value for the chip.
        if opening.preprocessed.local.len() != chip.preprocessed_width() {
            return Err(OpeningShapeError::PreprocessedWidthMismatch(
                chip.preprocessed_width(),
                opening.preprocessed.local.len(),
            ));
        }
        if opening.preprocessed.next.len() != chip.preprocessed_width() {
            return Err(OpeningShapeError::PreprocessedWidthMismatch(
                chip.preprocessed_width(),
                opening.preprocessed.next.len(),
            ));
        }

        // Verify that the main width matches the expected value for the chip.
        if opening.main.local.len() != chip.width() {
            return Err(OpeningShapeError::MainWidthMismatch(
                chip.width(),
                opening.main.local.len(),
            ));
        }
        if opening.main.next.len() != chip.width() {
            return Err(OpeningShapeError::MainWidthMismatch(chip.width(), opening.main.next.len()));
        }

        // Verify that the permutation width matches the expected value for the chip.
        if opening.permutation.local.len() !=
            chip.permutation_width() * <Challenge<SC> as AbstractExtensionField<C::F>>::D
        {
            return Err(OpeningShapeError::PermutationWidthMismatch(
                chip.permutation_width(),
                opening.permutation.local.len(),
            ));
        }
        if opening.permutation.next.len() !=
            chip.permutation_width() * <Challenge<SC> as AbstractExtensionField<C::F>>::D
        {
            return Err(OpeningShapeError::PermutationWidthMismatch(
                chip.permutation_width(),
                opening.permutation.next.len(),
            ));
        }

        // Verift that the number of quotient chunks matches the expected value for the chip.
        if opening.quotient.len() != chip.quotient_width() {
            return Err(OpeningShapeError::QuotientWidthMismatch(
                chip.quotient_width(),
                opening.quotient.len(),
            ));
        }
        // For each quotient chunk, verify that the number of elements is equal to the degree of the
        // challenge extension field over the value field.
        for slice in &opening.quotient {
            if slice.len() != <Challenge<SC> as AbstractExtensionField<C::F>>::D {
                return Err(OpeningShapeError::QuotientChunkSizeMismatch(
                    <Challenge<SC> as AbstractExtensionField<C::F>>::D,
                    slice.len(),
                ));
            }
        }

        Ok(())
    }
}
