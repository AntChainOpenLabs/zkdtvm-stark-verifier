use std::collections::BTreeMap;
use std::panic::{catch_unwind, AssertUnwindSafe};

use num_traits::cast::ToPrimitive;
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_dft::dft_eval::EvalsDft;
use p3_field::{ExtensionField, TwoAdicField};
use p3_matrix::bitrev::BitReversableMatrix;
use p3_matrix::compressed::CompressedMatrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::{Dimensions, Matrix};
use p3_util::log2_strict_usize;

use crate::basefold::basefold_pcs::{BaseFoldError, BaseFoldPcs, DimAndNo};
use crate::utils::field_conversion::{flatten_to_base, reconstitute_from_base};

impl<F, InputMmcs, FriMmcs, EF, Challenger> BaseFoldPcs<F, InputMmcs, FriMmcs, EF, Challenger>
where
    F: TwoAdicField,
    InputMmcs: Mmcs<F> + Send + Sync,
    InputMmcs::ProverData<RowMajorMatrix<F>>: Sync,
    FriMmcs: Mmcs<EF> + Send + Sync,
    EF: TwoAdicField + ExtensionField<F>,
    Challenger:
        FieldChallenger<F> + CanObserve<FriMmcs::Commitment> + GrindingChallenger<Witness = F>,
{
    /// Validate that the dimensions of polynomials, prover data, and opened values are consistent.
    pub(crate) fn validate_open_inputs(
        &self,
        polynomials_batch: &[Vec<CompressedMatrix<F>>],
        prover_data_batch: &[InputMmcs::ProverData<RowMajorMatrix<F>>],
        opened_values: &[Vec<Vec<EF>>],
    ) -> Result<(), BaseFoldError<FriMmcs::Error, InputMmcs::Error>> {
        if polynomials_batch.len() != opened_values.len()
            || polynomials_batch.len() != prover_data_batch.len()
        {
            return Err(BaseFoldError::InvalidInputError);
        }

        for (batch, vals) in polynomials_batch.iter().zip(opened_values.iter()) {
            if batch.len() != vals.len() {
                return Err(BaseFoldError::InvalidInputError);
            }
            for (matrix, col_vals) in batch.iter().zip(vals.iter()) {
                if col_vals.len() != matrix.width() && col_vals.len() * EF::D != matrix.width() {
                    return Err(BaseFoldError::InvalidInputError);
                }
            }
        }
        Ok(())
    }

    /// Validate that the dimensions of commitments, matrix sizes, and opened values are consistent.
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

    /// Group dimensions with their opened values by log_height (descending key order in BTreeMap).
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

    /// Group flattened compressed matrices by their total_height, paired with their opened values.
    pub(crate) fn group_by_log_height<'a>(
        &self,
        polynomials: &'a [CompressedMatrix<F>],
        flat_opened_values: &[&'a Vec<EF>],
    ) -> Result<
        BTreeMap<usize, Vec<(&'a CompressedMatrix<F>, &'a Vec<EF>)>>,
        BaseFoldError<FriMmcs::Error, InputMmcs::Error>,
    > {
        let mut groups: BTreeMap<usize, Vec<(&'a CompressedMatrix<F>, &'a Vec<EF>)>> =
            BTreeMap::new();

        for (matrix, values) in polynomials.iter().zip(flat_opened_values.iter()) {
            let height = matrix.height();
            if !height.is_power_of_two() {
                return Err(BaseFoldError::InvalidInputError);
            }
            groups
                .entry(log2_strict_usize(height))
                .or_default()
                .push((matrix, values));
        }
        Ok(groups)
    }

    /// Encode a polynomial (given as evaluations over the hypercube) into a Reed-Solomon codeword.
    ///
    /// Steps: repeat (blowup) → twiddle-free DFT → bit-reverse output.
    /// No input bit-reverse: compatible with little-endian (even/odd) folding.
    pub(crate) fn encode_to_codeword(&self, evals: &[EF], dft: &EvalsDft<F>) -> Vec<EF> {
        let mut coeffs: Vec<EF> = evals.to_vec();

        let repeat_times = 1 << self.fri.log_blowup;
        let orig_len = coeffs.len();
        coeffs.reserve(orig_len * (repeat_times - 1));
        for _ in 1..repeat_times {
            coeffs.extend_from_within(0..orig_len);
        }

        let base_values: Vec<F> = unsafe { flatten_to_base(coeffs) };
        let dft_output = dft
            .dft_batch_by_evals(RowMajorMatrix::new(base_values, EF::D))
            .bit_reverse_rows()
            .to_row_major_matrix();
        unsafe { reconstitute_from_base(dft_output.values) }
    }

    /// Find a proof-of-work witness by trial using the given number of grinding bits.
    pub(crate) fn find_pow_witness(
        &self,
        challenger: &mut Challenger,
        grinding_bits: usize,
    ) -> Result<Vec<F>, BaseFoldError<FriMmcs::Error, InputMmcs::Error>> {
        let order = F::order().to_u64().expect("F::order() should fit in u64");

        for i in 0..order {
            let nonce = F::from_canonical_u64(i);
            if let Ok(witness) = catch_unwind(AssertUnwindSafe(|| {
                let mut trial = challenger.clone();
                trial.observe(nonce);
                trial.grind(grinding_bits)
            })) {
                challenger.observe(nonce);
                challenger.observe(witness);
                assert_eq!(challenger.sample_bits(grinding_bits), 0);
                return Ok(vec![nonce, witness]);
            }
        }
        Err(BaseFoldError::CannotFindPowWitness)
    }
}
