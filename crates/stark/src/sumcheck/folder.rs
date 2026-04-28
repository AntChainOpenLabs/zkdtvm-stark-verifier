use crate::sumcheck::config::SCStarkGenericConfig;
use crate::{
    air::{EmptyMessageBuilder, MultiTableAirBuilder},
    config::{Challenge, Val},
    septic_digest::SepticDigest,
};
use p3_air::{
    AirBuilder, AirBuilderWithPublicValues, ExtensionBuilder, PairBuilder, PermutationAirBuilder,
};
use p3_matrix::dense::RowMajorMatrixView;

/// Verifier constraint folder for sumcheck V2: preprocessed, main, permutation are each one row.
pub struct SumcheckVerifierConstraintFolder<'a, SC: SCStarkGenericConfig> {
    pub preprocessed: RowMajorMatrixView<'a, Challenge<SC>>,
    pub main: RowMajorMatrixView<'a, Challenge<SC>>,
    pub permutation: RowMajorMatrixView<'a, Challenge<SC>>,
    pub perm_challenges: &'a [Challenge<SC>],
    pub local_cumulative_sum: &'a Challenge<SC>,
    pub global_cumulative_sum: &'a SepticDigest<Val<SC>>,
    pub is_first_row: Challenge<SC>,
    pub is_last_row: Challenge<SC>,
    pub alpha: Challenge<SC>,
    pub accumulator: Challenge<SC>,
    pub public_values: &'a [Val<SC>],
    pub constraint_count: usize,
}

impl<'a, SC: SCStarkGenericConfig> AirBuilder for SumcheckVerifierConstraintFolder<'a, SC> {
    type F = Val<SC>;
    type Expr = Challenge<SC>;
    type Var = Challenge<SC>;
    type M = RowMajorMatrixView<'a, Challenge<SC>>;

    fn main(&self) -> Self::M {
        self.main
    }

    fn is_first_row(&self) -> Self::Expr {
        self.is_first_row
    }

    fn is_last_row(&self) -> Self::Expr {
        self.is_last_row
    }

    fn is_transition_window(&self, size: usize) -> Self::Expr {
        panic!("Sumcheck does not support transition windows (requested size: {size})");
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        let x: Challenge<SC> = x.into();
        self.accumulator = self.accumulator * self.alpha + x;
        self.constraint_count += 1;
    }
}

impl<SC: SCStarkGenericConfig> ExtensionBuilder for SumcheckVerifierConstraintFolder<'_, SC> {
    type EF = Challenge<SC>;
    type ExprEF = Challenge<SC>;
    type VarEF = Challenge<SC>;

    fn assert_zero_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>,
    {
        let x: Challenge<SC> = x.into();
        self.accumulator = self.accumulator * self.alpha + x;
        self.constraint_count += 1;
    }
}

impl<SC: SCStarkGenericConfig> PairBuilder for SumcheckVerifierConstraintFolder<'_, SC> {
    fn preprocessed(&self) -> Self::M {
        self.preprocessed
    }
}

impl<'a, SC: SCStarkGenericConfig> PermutationAirBuilder
    for SumcheckVerifierConstraintFolder<'a, SC>
{
    type MP = RowMajorMatrixView<'a, Challenge<SC>>;
    type RandomVar = Challenge<SC>;

    fn permutation(&self) -> Self::MP {
        self.permutation
    }

    fn permutation_randomness(&self) -> &[Self::RandomVar] {
        self.perm_challenges
    }
}

impl<'a, SC: SCStarkGenericConfig> MultiTableAirBuilder<'a>
    for SumcheckVerifierConstraintFolder<'a, SC>
{
    type LocalSum = Challenge<SC>;
    type GlobalSum = Val<SC>;

    fn local_cumulative_sum(&self) -> &'a Self::LocalSum {
        self.local_cumulative_sum
    }

    fn global_cumulative_sum(&self) -> &'a SepticDigest<Self::GlobalSum> {
        self.global_cumulative_sum
    }
}

impl<SC: SCStarkGenericConfig> EmptyMessageBuilder for SumcheckVerifierConstraintFolder<'_, SC> {}

impl<SC: SCStarkGenericConfig> AirBuilderWithPublicValues
    for SumcheckVerifierConstraintFolder<'_, SC>
{
    type PublicVar = Val<SC>;

    fn public_values(&self) -> &[Self::PublicVar] {
        self.public_values
    }
}
