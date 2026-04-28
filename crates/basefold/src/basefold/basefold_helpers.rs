use std::collections::BTreeMap;

use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{ExtensionField, TwoAdicField};
use p3_matrix::Dimensions;
use p3_util::log2_strict_usize;

use crate::basefold::basefold_pcs::{BaseFoldError, BaseFoldPcs, DimAndNo};

impl<F, InputMmcs, FriMmcs, EF, Challenger> BaseFoldPcs<F, InputMmcs, FriMmcs, EF, Challenger>
where
    F: TwoAdicField,
    InputMmcs: Mmcs<F> + Send + Sync,
    FriMmcs: Mmcs<EF> + Send + Sync,
    EF: TwoAdicField + ExtensionField<F>,
    Challenger:
        FieldChallenger<F> + CanObserve<FriMmcs::Commitment> + GrindingChallenger<Witness = F>,
{
    pub(crate) fn validate_verify_inputs(
        &self,
        commitment_batch: &[InputMmcs::Commitment],
        matrices_size_batch: &[Vec<Dimensions>],
        opened_values_batch: &[Vec<Vec<EF>>],
    ) -> Result<(), BaseFoldError<FriMmcs::Error, InputMmcs::Error>> {
        if matrices_size_batch.len() != opened_values_batch.len()
            || commitment_batch.len() != matrices_size_batch.len()
        {
            return Err(BaseFoldError::InvalidInputError);
        }

        for (matrices_size, opened_values) in
            matrices_size_batch.iter().zip(opened_values_batch.iter())
        {
            if matrices_size.len() != opened_values.len() {
                return Err(BaseFoldError::InvalidInputError);
            }
            for (dim, vals) in matrices_size.iter().zip(opened_values.iter()) {
                if vals.len() != dim.width && vals.len() * EF::D != dim.width {
                    return Err(BaseFoldError::InvalidInputError);
                }
            }
        }
        Ok(())
    }

    pub(crate) fn group_dims_by_log_height<'a>(
        matrices_size: &'a [DimAndNo],
        flat_opened_values: &[&'a Vec<EF>],
    ) -> BTreeMap<usize, Vec<(&'a DimAndNo, &'a Vec<EF>)>> {
        let mut groups: BTreeMap<usize, Vec<(&'a DimAndNo, &'a Vec<EF>)>> = BTreeMap::new();
        for (dim_no, values) in matrices_size.iter().zip(flat_opened_values.iter()) {
            groups
                .entry(log2_strict_usize(dim_no.dim.height))
                .or_default()
                .push((dim_no, values));
        }
        groups
    }
}
