use crate::{
    air::{EmptyMessageBuilder, MultiTableAirBuilder},
    config::{Challenge, PackedChallenge, PackedExt, PackedVal, Val},
    sumcheck::config::SCStarkGenericConfig,
};
use p3_air::{
    AirBuilder, AirBuilderWithPublicValues, ExtensionBuilder, PairBuilder, PermutationAirBuilder,
};
use p3_field::AbstractField;
use p3_matrix::dense::RowMajorMatrixView;

/// A folder for sumcheck prover constraints using base field values.
///
/// This is a simplified version of `ProverConstraintFolder` that:
/// - Does not involve rotations/transition windows (sumcheck operates on folded traces)
/// - Operates on single rows using `RowMajorMatrixView`
///
/// Fields represent the state at a specific sumcheck round with current challenge values.
pub struct SumcheckConstraintFolder<'a, SC: SCStarkGenericConfig> {
    /// The preprocessed trace row (precomputed constants).
    pub preprocessed: RowMajorMatrixView<'a, PackedVal<SC>>,
    /// The main trace row (regular execution trace).
    pub main: RowMajorMatrixView<'a, PackedVal<SC>>,
    /// The permutation trace row (for permutation arguments).
    pub permutation: RowMajorMatrixView<'a, PackedChallenge<SC>>,
    /// The challenges for the permutation argument.
    pub permutation_challenges: &'a [PackedChallenge<SC>],
    /// The local cumulative sum for the permutation.
    pub local_cumulative_sum: &'a PackedChallenge<SC>,
    /// The selector for the first row of the original trace.
    pub is_first_row: PackedVal<SC>,
    /// The selector for the last row of the original trace.
    pub is_last_row: PackedVal<SC>,
    /// The powers of the constraint folding challenge (alpha).
    pub powers_of_alpha: &'a [Challenge<SC>],
    /// The accumulator for folded constraints.
    pub accumulator: PackedChallenge<SC>,
    /// The public values.
    pub public_values: &'a [Val<SC>],
    /// The constraint index for tracking constraint folding.
    pub constraint_index: usize,
}

impl<'a, SC: SCStarkGenericConfig> AirBuilder for SumcheckConstraintFolder<'a, SC> {
    type F = Val<SC>;
    type Expr = PackedVal<SC>;
    type Var = PackedVal<SC>;
    type M = RowMajorMatrixView<'a, PackedVal<SC>>;

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
        let x: PackedVal<SC> = x.into();
        self.accumulator +=
            PackedChallenge::<SC>::from_f(self.powers_of_alpha[self.constraint_index]) * x;
        self.constraint_index += 1;
    }
}

impl<SC: SCStarkGenericConfig> ExtensionBuilder for SumcheckConstraintFolder<'_, SC> {
    type EF = Challenge<SC>;
    type ExprEF = PackedChallenge<SC>;
    type VarEF = PackedChallenge<SC>;

    fn assert_zero_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>,
    {
        let x: PackedChallenge<SC> = x.into();
        self.accumulator +=
            PackedChallenge::<SC>::from_f(self.powers_of_alpha[self.constraint_index]) * x;
        self.constraint_index += 1;
    }
}

impl<SC: SCStarkGenericConfig> PairBuilder for SumcheckConstraintFolder<'_, SC> {
    fn preprocessed(&self) -> Self::M {
        self.preprocessed
    }
}

impl<'a, SC: SCStarkGenericConfig> PermutationAirBuilder for SumcheckConstraintFolder<'a, SC> {
    type MP = RowMajorMatrixView<'a, PackedChallenge<SC>>;
    type RandomVar = PackedChallenge<SC>;

    fn permutation(&self) -> Self::MP {
        self.permutation
    }

    fn permutation_randomness(&self) -> &[Self::RandomVar] {
        self.permutation_challenges
    }
}

impl<SC: SCStarkGenericConfig> EmptyMessageBuilder for SumcheckConstraintFolder<'_, SC> {}

impl<SC: SCStarkGenericConfig> AirBuilderWithPublicValues for SumcheckConstraintFolder<'_, SC> {
    type PublicVar = <Self as AirBuilder>::F;

    fn public_values(&self) -> &[Self::PublicVar] {
        self.public_values
    }
}

impl<'a, SC: SCStarkGenericConfig> MultiTableAirBuilder<'a> for SumcheckConstraintFolder<'a, SC> {
    type LocalSum = PackedChallenge<SC>;

    fn local_cumulative_sum(&self) -> &'a Self::LocalSum {
        self.local_cumulative_sum
    }
}

/// A folder for sumcheck prover constraints using extension field values.
///
/// This is the extension field version of `SumcheckConstraintFolder`.
pub struct SumcheckConstraintFolderExt<'a, SC: SCStarkGenericConfig> {
    /// The preprocessed trace row (precomputed constants).
    pub preprocessed: RowMajorMatrixView<'a, PackedExt<SC>>,
    /// The main trace row (regular execution trace).
    pub main: RowMajorMatrixView<'a, PackedExt<SC>>,
    /// The permutation trace row (for permutation arguments).
    pub permutation: RowMajorMatrixView<'a, PackedExt<SC>>,
    /// The challenges for the permutation argument.
    pub permutation_challenges: &'a [PackedExt<SC>],
    /// The local cumulative sum for the permutation.
    pub local_cumulative_sum: &'a PackedExt<SC>,
    /// The selector for the first row of the original trace.
    pub is_first_row: PackedExt<SC>,
    /// The selector for the last row of the original trace.
    pub is_last_row: PackedExt<SC>,
    /// The powers of the constraint folding challenge (alpha).
    pub powers_of_alpha: &'a [Challenge<SC>],
    /// The accumulator for folded constraints.
    pub accumulator: PackedExt<SC>,
    /// The public values.
    pub public_values: &'a [Challenge<SC>],
    /// The constraint index for tracking constraint folding.
    pub constraint_index: usize,
}

impl<'a, SC: SCStarkGenericConfig> AirBuilder for SumcheckConstraintFolderExt<'a, SC> {
    type F = Challenge<SC>;
    type Expr = PackedExt<SC>;
    type Var = PackedExt<SC>;
    type M = RowMajorMatrixView<'a, PackedExt<SC>>;

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
        let x: PackedExt<SC> = x.into();
        self.accumulator += PackedExt::<SC>::from(self.powers_of_alpha[self.constraint_index]) * x;
        self.constraint_index += 1;
    }
}

impl<SC: SCStarkGenericConfig> ExtensionBuilder for SumcheckConstraintFolderExt<'_, SC> {
    type EF = Challenge<SC>;
    type ExprEF = PackedExt<SC>;
    type VarEF = PackedExt<SC>;

    fn assert_zero_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>,
    {
        let x: PackedExt<SC> = x.into();
        self.accumulator += x * self.powers_of_alpha[self.constraint_index];
        self.constraint_index += 1;
    }
}

impl<SC: SCStarkGenericConfig> PairBuilder for SumcheckConstraintFolderExt<'_, SC> {
    fn preprocessed(&self) -> Self::M {
        self.preprocessed
    }
}

impl<'a, SC: SCStarkGenericConfig> PermutationAirBuilder for SumcheckConstraintFolderExt<'a, SC> {
    type MP = RowMajorMatrixView<'a, PackedExt<SC>>;
    type RandomVar = PackedExt<SC>;

    fn permutation(&self) -> Self::MP {
        self.permutation
    }

    fn permutation_randomness(&self) -> &[Self::RandomVar] {
        self.permutation_challenges
    }
}

impl<SC: SCStarkGenericConfig> EmptyMessageBuilder for SumcheckConstraintFolderExt<'_, SC> {}

impl<SC: SCStarkGenericConfig> AirBuilderWithPublicValues for SumcheckConstraintFolderExt<'_, SC> {
    type PublicVar = Challenge<SC>;

    fn public_values(&self) -> &[Self::PublicVar] {
        self.public_values
    }
}

impl<'a, SC: SCStarkGenericConfig> MultiTableAirBuilder<'a>
    for SumcheckConstraintFolderExt<'a, SC>
{
    type LocalSum = PackedExt<SC>;

    fn local_cumulative_sum(&self) -> &'a Self::LocalSum {
        self.local_cumulative_sum
    }
}

/// A folder for evaluating constraints on a single padding row.
///
/// Unlike `SumcheckConstraintFolder` which uses packed field types for SIMD parallelism,
/// this folder uses scalar `Challenge<SC>` types since it evaluates constraints on a
/// single padding row. The padding row is neither the first nor the last row, so
/// `is_first_row` and `is_last_row` are always zero.
///
/// Used during `ChipState` initialization to precompute the constraint evaluation
/// on the padding row, which can then be multiplied by the number of padding rows
/// in each sumcheck round.
pub struct PaddingRowConstraintFolder<'a, SC: SCStarkGenericConfig> {
    /// The preprocessed trace padding row (base field).
    pub preprocessed: RowMajorMatrixView<'a, Val<SC>>,
    /// The main trace padding row (base field).
    pub main: RowMajorMatrixView<'a, Val<SC>>,
    /// The permutation trace padding row (extension field).
    pub permutation: RowMajorMatrixView<'a, Challenge<SC>>,
    /// The challenges for the permutation argument.
    pub permutation_challenges: &'a [Challenge<SC>],
    /// The local cumulative sum for the permutation.
    pub local_cumulative_sum: &'a Challenge<SC>,
    /// The selector for the first row of the original trace (base field).
    pub is_first_row: Val<SC>,
    /// The selector for the last row of the original trace (base field).
    pub is_last_row: Val<SC>,
    /// The powers of the constraint folding challenge (alpha).
    pub powers_of_alpha: &'a [Challenge<SC>],
    /// The accumulator for folded constraints.
    pub accumulator: Challenge<SC>,
    /// The public values (base field).
    pub public_values: &'a [Val<SC>],
    /// The constraint index for tracking constraint folding.
    pub constraint_index: usize,
}

impl<'a, SC: SCStarkGenericConfig> AirBuilder for PaddingRowConstraintFolder<'a, SC> {
    type F = Val<SC>;
    type Expr = Val<SC>;
    type Var = Val<SC>;
    type M = RowMajorMatrixView<'a, Val<SC>>;

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
        panic!("PaddingRowConstraintFolder does not support transition windows (requested size: {size})");
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        let x: Val<SC> = x.into();
        self.accumulator += self.powers_of_alpha[self.constraint_index] * Challenge::<SC>::from(x);
        self.constraint_index += 1;
    }
}

impl<SC: SCStarkGenericConfig> ExtensionBuilder for PaddingRowConstraintFolder<'_, SC> {
    type EF = Challenge<SC>;
    type ExprEF = Challenge<SC>;
    type VarEF = Challenge<SC>;

    fn assert_zero_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>,
    {
        let x: Challenge<SC> = x.into();
        self.accumulator += self.powers_of_alpha[self.constraint_index] * x;
        self.constraint_index += 1;
    }
}

impl<SC: SCStarkGenericConfig> PairBuilder for PaddingRowConstraintFolder<'_, SC> {
    fn preprocessed(&self) -> Self::M {
        self.preprocessed
    }
}

impl<'a, SC: SCStarkGenericConfig> PermutationAirBuilder for PaddingRowConstraintFolder<'a, SC> {
    type MP = RowMajorMatrixView<'a, Challenge<SC>>;
    type RandomVar = Challenge<SC>;

    fn permutation(&self) -> Self::MP {
        self.permutation
    }

    fn permutation_randomness(&self) -> &[Self::RandomVar] {
        self.permutation_challenges
    }
}

impl<SC: SCStarkGenericConfig> EmptyMessageBuilder for PaddingRowConstraintFolder<'_, SC> {}

impl<SC: SCStarkGenericConfig> AirBuilderWithPublicValues for PaddingRowConstraintFolder<'_, SC> {
    type PublicVar = Val<SC>;

    fn public_values(&self) -> &[Self::PublicVar] {
        self.public_values
    }
}

impl<'a, SC: SCStarkGenericConfig> MultiTableAirBuilder<'a> for PaddingRowConstraintFolder<'a, SC> {
    type LocalSum = Challenge<SC>;

    fn local_cumulative_sum(&self) -> &'a Self::LocalSum {
        self.local_cumulative_sum
    }
}

// ---------------------------------------------------------------------------
// Sumcheck verifier folder: single row per trace (constraints evaluated at one opened point).
// ---------------------------------------------------------------------------

/// Verifier constraint folder for the sumcheck verifier: preprocessed, main, permutation are each
/// one row. No transition (single-row evaluation), so no `is_transition` field.
pub struct SumcheckVerifierConstraintFolder<'a, SC: SCStarkGenericConfig> {
    pub preprocessed: RowMajorMatrixView<'a, Challenge<SC>>,
    pub main: RowMajorMatrixView<'a, Challenge<SC>>,
    pub permutation: RowMajorMatrixView<'a, Challenge<SC>>,
    pub perm_challenges: &'a [Challenge<SC>],
    pub local_cumulative_sum: &'a Challenge<SC>,
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

    fn local_cumulative_sum(&self) -> &'a Self::LocalSum {
        self.local_cumulative_sum
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
