//! Trace: re-exports `CompressedMatrix` and related types from `p3_matrix::compressed`,
//! plus the zkDTVM-specific `ChipTrace` state machine.

pub use p3_matrix::compressed::{
    padding_row_sum, padding_row_to_base_vec, padding_row_to_challenge_vec, CompressedMatrix,
    FoldableBase, FoldableExt, FoldableHybrid, PaddingRow,
};

use crate::{
    config::{Challenge, Val},
    sumcheck::{config::SCStarkGenericConfig, types::BitExpandPoly},
};
use p3_field::{AbstractField, ExtensionField};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

/// Four-state representation of a single trace:
/// - `FirstRound`: reference to base-field `CompressedMatrix` (preprocessed/main at init)
///   - `padding_row` and main are both base field `Val<SC>`
/// - `FirstRoundExt`: reference to extension-field `CompressedMatrix` (permutation at init)
///   - `padding_row` and main are both extension field `Challenge<SC>`
/// - `NonFirstRound`: owned hybrid `CompressedMatrix` (after fold of preprocessed/main)
///   - `padding_row` is base `Val<SC>`, main is extension `Challenge<SC>`
/// - `NonFirstRoundExt`: owned extension-field `CompressedMatrix` (permutation after fold)
///   - `padding_row` and main are both extension field `Challenge<SC>`
pub enum ChipTrace<'a, SC: SCStarkGenericConfig> {
    FirstRound(&'a CompressedMatrix<Val<SC>, Val<SC>>),
    FirstRoundExt(&'a CompressedMatrix<Challenge<SC>, Challenge<SC>>),
    NonFirstRound(CompressedMatrix<Val<SC>, Challenge<SC>>),
    NonFirstRoundExt(CompressedMatrix<Challenge<SC>, Challenge<SC>>),
}
impl<SC: SCStarkGenericConfig> ChipTrace<'_, SC> {
    pub fn total_height(&self) -> usize {
        match self {
            ChipTrace::FirstRound(cm) => cm.total_height,
            ChipTrace::FirstRoundExt(cm) => cm.total_height,
            ChipTrace::NonFirstRound(cm) => cm.total_height,
            ChipTrace::NonFirstRoundExt(cm) => cm.total_height,
        }
    }

    pub fn get_sum_perm_rows_linear(&self, point: usize) -> Challenge<SC> {
        match point {
            0 | 1 => match self {
                ChipTrace::FirstRoundExt(mat) => Self::sum_rows_linear(point, mat),
                ChipTrace::NonFirstRoundExt(mat) => Self::sum_rows_linear(point, mat),
                _ => panic!("Invalid ChipTrace variant"),
            },
            _ => panic!("Invalid point: {point}, must be 0 (even rows) or 1 (odd rows)"),
        }
    }

    fn sum_rows_linear(
        point: usize,
        mat: &CompressedMatrix<Challenge<SC>, Challenge<SC>>,
    ) -> Challenge<SC> {
        let mut sum = Challenge::<SC>::zero();
        for row in (point..mat.total_height).step_by(2) {
            if row < mat.main.height() {
                for col in 0..mat.main.width() {
                    sum += mat.main.get(row, col);
                }
            } else {
                match &mat.padding_row {
                    PaddingRow::None | PaddingRow::Zero { .. } => {}
                    PaddingRow::Constant { value, width } => {
                        sum += *value * Challenge::<SC>::from_canonical_usize(*width);
                    }
                    PaddingRow::General(row_data) => {
                        sum += row_data.iter().copied().sum::<Challenge<SC>>();
                    }
                }
            }
        }
        sum
    }

    /// Returns summation input as a **base-field** compressed matrix.
    ///
    /// Only valid on `ChipTrace::FirstRound`. Branch order: (1) if `point <= var_degree`, extract
    /// rows at stride; (2) else if `var_degree == 1`, fold once with small integer; (3) else fold
    /// with base-field points from `BitExpandPoly::eval_all`.
    pub fn get_summation_input_base(
        &self,
        point: usize,
        var_degree: usize,
        bit_expand_poly: Option<&BitExpandPoly<Val<SC>>>,
    ) -> CompressedMatrix<Val<SC>, Val<SC>> {
        match self {
            ChipTrace::FirstRound(mat) => {
                if point <= var_degree {
                    mat.get_rows_at_stride(point, var_degree + 1)
                } else if var_degree == 1 {
                    mat.fold_base_with_small_integer(point)
                } else {
                    let poly = bit_expand_poly.expect(
                        "bit_expand_poly required when point > var_degree and var_degree > 1",
                    );
                    let points = poly.evals_all(Val::<SC>::from_canonical_usize(point));
                    FoldableBase::fold_base_with_multiple_base(mat, points)
                }
            }
            _ => panic!("get_summation_input_base only supports ChipTrace::FirstRound"),
        }
    }

    /// Returns summation input as an **extension-field** compressed matrix.
    ///
    /// Valid on `ChipTrace::FirstRoundExt` and `ChipTrace::NonFirstRoundExt`.
    pub fn get_summation_input_ext(
        &self,
        point: usize,
        var_degree: usize,
        bit_expand_poly: Option<&BitExpandPoly<Val<SC>>>,
    ) -> CompressedMatrix<Challenge<SC>, Challenge<SC>> {
        let empty_ext = || {
            CompressedMatrix::<Challenge<SC>, Challenge<SC>>::new(
                RowMajorMatrix::new(vec![], 0),
                PaddingRow::None,
                0,
            )
        };
        let do_ext = |mat: &CompressedMatrix<Challenge<SC>, Challenge<SC>>| {
            if mat.total_height == 0 {
                return empty_ext();
            }
            if point <= var_degree {
                mat.get_rows_at_stride(point, var_degree + 1)
            } else if var_degree == 1 {
                FoldableExt::<Val<SC>, Challenge<SC>>::fold_ext_with_small_integer(mat, point)
            } else {
                let poly = bit_expand_poly
                    .expect("bit_expand_poly required when point > var_degree and var_degree > 1");
                let points = poly.evals_all(Val::<SC>::from_canonical_usize(point));
                mat.fold_ext_with_multiple_base(points)
            }
        };
        match self {
            ChipTrace::FirstRoundExt(mat) => do_ext(mat),
            ChipTrace::NonFirstRoundExt(mat) => do_ext(mat),
            _ => panic!(
                "get_summation_input_ext only supports ChipTrace::FirstRoundExt and NonFirstRoundExt"
            ),
        }
    }

    /// Returns summation input as a **hybrid** (base padding, extension main) compressed matrix.
    ///
    /// Only valid on `ChipTrace::NonFirstRound`.
    pub fn get_summation_input_hybrid(
        &self,
        point: usize,
        var_degree: usize,
        bit_expand_poly: Option<&BitExpandPoly<Val<SC>>>,
    ) -> CompressedMatrix<Val<SC>, Challenge<SC>> {
        match self {
            ChipTrace::NonFirstRound(mat) => {
                if point <= var_degree {
                    mat.get_rows_at_stride(point, var_degree + 1)
                } else if var_degree == 1 {
                    mat.fold_hybrid_with_small_integer(point)
                } else {
                    let poly = bit_expand_poly.expect(
                        "bit_expand_poly required when point > var_degree and var_degree > 1",
                    );
                    let points = poly.evals_all(Val::<SC>::from_canonical_usize(point));
                    mat.fold_hybrid_with_multiple_base(points)
                }
            }
            _ => panic!("get_summation_input_hybrid only supports ChipTrace::NonFirstRound"),
        }
    }

    /// Fold the trace matrix with the given challenge, transitioning state as needed.
    ///
    /// # State transitions
    ///
    /// - `FirstRound` → `NonFirstRound`: folds `CompressedMatrix<Val, Val>` into
    ///   `CompressedMatrix<Val, Challenge>` via `fold_base_with_ext`.
    /// - `FirstRoundExt` → `NonFirstRoundExt`: clones the referenced `CompressedMatrix<Challenge,
    ///   Challenge>` and folds it with the challenge.
    /// - `NonFirstRound` → `NonFirstRound`: folds `CompressedMatrix<Val, Challenge>` in-place via
    ///   `fold_hybrid_with_ext_in_place`.
    /// - `NonFirstRoundExt` → `NonFirstRoundExt`: folds `CompressedMatrix<Challenge, Challenge>`
    ///   in-place with the challenge.
    pub fn update(&mut self, challenge: Challenge<SC>) {
        match self {
            ChipTrace::FirstRound(mat) => {
                let hybrid_mat = mat.fold_base_with_ext(challenge);
                *self = ChipTrace::NonFirstRound(hybrid_mat);
            }
            ChipTrace::FirstRoundExt(mat) => {
                let folded = <CompressedMatrix<Challenge<SC>, Challenge<SC>> as FoldableExt<
                    Val<SC>,
                    Challenge<SC>,
                >>::fold_ext_with_ext(mat, challenge);
                *self = ChipTrace::NonFirstRoundExt(folded);
            }
            ChipTrace::NonFirstRound(mat) => {
                mat.fold_hybrid_with_ext_in_place(challenge);
            }
            ChipTrace::NonFirstRoundExt(mat) => {
                FoldableExt::<Val<SC>, Challenge<SC>>::fold_ext_with_ext_in_place(mat, challenge);
            }
        }
    }

    /// Returns row `idx` as `Vec<Challenge<SC>>` for opening values.
    /// Only used after sumcheck rounds are done, so only `NonFirstRound` / `NonFirstRoundExt`
    /// occur.
    pub fn get_row_for_opening(&self, idx: usize) -> Vec<Challenge<SC>>
    where
        Challenge<SC>: ExtensionField<Val<SC>>,
    {
        match self {
            ChipTrace::NonFirstRound(mat) => mat.get_row(idx),
            ChipTrace::NonFirstRoundExt(mat) => mat.get_row(idx),
            ChipTrace::FirstRound(_) | ChipTrace::FirstRoundExt(_) => {
                unreachable!("get_row_for_opening is only used after folding; FirstRound/FirstRoundExt do not occur")
            }
        }
    }
}
