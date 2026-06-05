use crate::sumcheck::trace::{CompressedMatrix, PaddingRow};
use crate::{
    air::{InteractionScope, MultiTableAirBuilder},
    lookup::Interaction,
};
use hashbrown::HashMap;
use itertools::Itertools;
use p3_air::{AirBuilder, PairBuilder};
use p3_field::{AbstractExtensionField, AbstractField, ExtensionField, Field, PrimeField};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::*;
use std::borrow::Borrow;

/// Computes the width of the local permutation trace in terms of extension field elements.
#[must_use]
pub const fn local_permutation_trace_width(nb_interactions: usize, batch_size: usize) -> usize {
    if nb_interactions == 0 {
        return 0;
    }
    nb_interactions.div_ceil(batch_size)
}

/// Populates a local permutation row.
#[inline]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_pass_by_value)]
pub fn populate_local_permutation_row<F: PrimeField, EF: ExtensionField<F>>(
    row: &mut [EF],
    preprocessed_row: &[F],
    main_row: &[F],
    sends: &[Interaction<F>],
    receives: &[Interaction<F>],
    random_elements: &[EF],
    batch_size: usize,
) {
    let alpha = random_elements[0];
    let betas = random_elements[1].powers();

    let interaction_chunks = &sends
        .iter()
        .map(|int| (int, true))
        .chain(receives.iter().map(|int| (int, false)))
        .chunks(batch_size);

    // Compute the denominators \prod_{i\in B} row_fingerprint(alpha, beta).
    for (value, chunk) in row.iter_mut().zip(interaction_chunks) {
        *value = chunk
            .into_iter()
            .map(|(interaction, is_send)| {
                let mut denominator = alpha;
                let mut betas = betas.clone();
                denominator +=
                    betas.next().unwrap() * EF::from_canonical_usize(interaction.argument_index());
                for (columns, beta) in interaction.values.iter().zip(betas) {
                    denominator += beta * columns.apply::<F, F>(preprocessed_row, main_row);
                }
                let mut mult = interaction.multiplicity.apply::<F, F>(preprocessed_row, main_row);

                if !is_send {
                    mult = -mult;
                }

                EF::from_base(mult) / denominator
            })
            .sum();
    }
}

/// Returns the sends, receives, and permutation trace width grouped by scope.
#[allow(clippy::type_complexity)]
pub fn scoped_interactions<F: Field>(
    sends: &[Interaction<F>],
    receives: &[Interaction<F>],
) -> (HashMap<InteractionScope, Vec<Interaction<F>>>, HashMap<InteractionScope, Vec<Interaction<F>>>)
{
    // Create a hashmap of scope -> vec<send interactions>.
    let mut sends = sends.to_vec();
    sends.sort_by_key(|k| k.scope);
    let grouped_sends: HashMap<_, _> = sends
        .iter()
        .chunk_by(|int| int.scope)
        .into_iter()
        .map(|(k, values)| (k, values.cloned().collect_vec()))
        .collect();

    // Create a hashmap of scope -> vec<receive interactions>.
    let mut receives = receives.to_vec();
    receives.sort_by_key(|k| k.scope);
    let grouped_receives: HashMap<_, _> = receives
        .iter()
        .chunk_by(|int| int.scope)
        .into_iter()
        .map(|(k, values)| (k, values.cloned().collect_vec()))
        .collect();

    (grouped_sends, grouped_receives)
}

/// Generates the permutation trace for the given chip and main trace based on a variant of `LogUp`.
#[allow(clippy::too_many_lines)]
pub fn generate_permutation_trace<F: PrimeField, EF: ExtensionField<F>>(
    sends: &[Interaction<F>],
    receives: &[Interaction<F>],
    preprocessed: Option<&RowMajorMatrix<F>>,
    main: &RowMajorMatrix<F>,
    random_elements: &[EF],
    batch_size: usize,
) -> (RowMajorMatrix<EF>, EF) {
    let empty = vec![];
    let (scoped_sends, scoped_receives) = scoped_interactions(sends, receives);
    let local_sends = scoped_sends.get(&InteractionScope::Local).unwrap_or(&empty);
    let local_receives = scoped_receives.get(&InteractionScope::Local).unwrap_or(&empty);

    let local_permutation_width =
        local_permutation_trace_width(local_sends.len() + local_receives.len(), batch_size);

    let height = main.height();
    let permutation_trace_width = local_permutation_width;
    let mut permutation_trace = RowMajorMatrix::new(
        vec![EF::zero(); permutation_trace_width * height],
        permutation_trace_width,
    );

    let mut local_cumulative_sum = EF::zero();

    let random_elements = &random_elements[0..2];

    if !local_sends.is_empty() || !local_receives.is_empty() {
        if let Some(prep) = preprocessed {
            assert_eq!(
                prep.height(),
                main.height(),
                "preprocessed and main have different heights: main width = {}, preprocessed width = {}",
                main.width(),
                prep.width()
            );
            assert_eq!(
                permutation_trace.height(),
                main.height(),
                "permutation trace and main have different heights"
            );
            permutation_trace
                .par_rows_mut()
                .zip_eq(prep.par_row_slices())
                .zip_eq(main.par_row_slices())
                .for_each(|((row, prep_row), main_row)| {
                    populate_local_permutation_row::<F, EF>(
                        &mut row[0..local_permutation_width],
                        prep_row,
                        main_row,
                        local_sends,
                        local_receives,
                        random_elements,
                        batch_size,
                    );
                });
        } else {
            permutation_trace.par_rows_mut().zip_eq(main.par_row_slices()).for_each(
                |(row, main_row)| {
                    populate_local_permutation_row::<F, EF>(
                        &mut row[0..local_permutation_width],
                        &[],
                        main_row,
                        local_sends,
                        local_receives,
                        random_elements,
                        batch_size,
                    );
                },
            );
        }

        local_cumulative_sum = permutation_trace
            .par_rows_mut()
            .map(|row| row[0..permutation_trace_width].iter().copied().sum::<EF>())
            .sum();
    }

    (permutation_trace, local_cumulative_sum)
}

#[allow(clippy::too_many_lines)]
pub fn generate_compressed_permutation_trace<F: PrimeField, EF: ExtensionField<F>>(
    sends: &[Interaction<F>],
    receives: &[Interaction<F>],
    preprocessed: Option<&CompressedMatrix<F>>,
    main: &CompressedMatrix<F>,
    random_elements: &[EF],
    batch_size: usize,
    chip_name: &str,
) -> (CompressedMatrix<EF>, EF) {
    let empty = vec![];
    let (scoped_sends, scoped_receives) = scoped_interactions(sends, receives);
    let local_sends = scoped_sends.get(&InteractionScope::Local).unwrap_or(&empty);
    let local_receives = scoped_receives.get(&InteractionScope::Local).unwrap_or(&empty);

    let local_permutation_width =
        local_permutation_trace_width(local_sends.len() + local_receives.len(), batch_size);

    let stored_height = main.main.height();
    let total_height = main.total_height;
    let padding_count = total_height - stored_height;

    let perm_width = local_permutation_width;
    let prep_width = preprocessed.map_or(0, |prep| prep.main.width());
    let main_width = main.main.width();

    let random_elements = &random_elements[0..2];
    let mut local_cumulative_sum = EF::zero();

    let mut perm_main =
        RowMajorMatrix::new(vec![EF::zero(); perm_width * stored_height], perm_width);
    let mut perm_padding_row: PaddingRow<EF> = PaddingRow::None;

    if !local_sends.is_empty() || !local_receives.is_empty() {
        if let Some(prep) = preprocessed {
            assert_eq!(
                prep.total_height, main.total_height,
                "preprocessed and main total heights differ for chip `{}` \
                 (prep.total_height={}, main.total_height={}, \
                 prep.main.height()={}, main.main.height()={}, \
                 prep_width={}, main_width={})",
                chip_name,
                prep.total_height,
                main.total_height,
                prep.main.height(),
                main.main.height(),
                prep.main.width(),
                main.main.width(),
            );
            assert_eq!(
                prep.main.height(),
                main.main.height(),
                "preprocessed and main stored heights differ for chip `{}` \
                 (prep.main.height()={}, main.main.height()={})",
                chip_name,
                prep.main.height(),
                main.main.height(),
            );

            // process main rows
            perm_main
                .par_rows_mut()
                .zip_eq(prep.main.par_row_slices())
                .zip_eq(main.main.par_row_slices())
                .for_each(|((row, prep_row), main_row)| {
                    populate_local_permutation_row::<F, EF>(
                        &mut row[0..perm_width],
                        prep_row,
                        main_row,
                        local_sends,
                        local_receives,
                        random_elements,
                        batch_size,
                    );
                });

            // process padding row with same logic
            if padding_count > 0 {
                let prep_pad_row = match &prep.padding_row {
                    // padding_count > 0, 应该不会出现None的情况，如果是None，这里应该报错
                    PaddingRow::None => {
                        panic!("Preprocessed padding_row should not be None when padding count > 0")
                    }
                    PaddingRow::Zero { .. } => vec![F::zero(); prep_width],
                    PaddingRow::Constant { value, .. } => vec![*value; prep_width],
                    PaddingRow::General(v) => v.clone(),
                };

                let main_pad_row = match &main.padding_row {
                    PaddingRow::None => {
                        panic!("Main padding_row should not be None when padding count > 0")
                    }
                    PaddingRow::Zero { .. } => vec![F::zero(); main_width],
                    PaddingRow::Constant { value, .. } => vec![*value; main_width],
                    PaddingRow::General(v) => v.clone(),
                };

                let mut perm_pad_row = vec![EF::zero(); perm_width];

                populate_local_permutation_row::<F, EF>(
                    &mut perm_pad_row,
                    &prep_pad_row,
                    &main_pad_row,
                    local_sends,
                    local_receives,
                    random_elements,
                    batch_size,
                );

                let perm_main_sum: EF = perm_main.values.iter().copied().sum();
                let pad_row_sum: EF = perm_pad_row.iter().copied().sum();

                local_cumulative_sum =
                    perm_main_sum + pad_row_sum * EF::from_canonical_usize(padding_count);

                perm_padding_row = PaddingRow::General(perm_pad_row);
            } else {
                // padding_count = 0, i.e., no padding
                local_cumulative_sum = perm_main.values.iter().copied().sum();
                perm_padding_row = PaddingRow::None;
            }
        } else {
            // process main rows
            perm_main.par_rows_mut().zip_eq(main.main.par_row_slices()).for_each(
                |(row, main_row)| {
                    populate_local_permutation_row::<F, EF>(
                        &mut row[0..perm_width],
                        &[],
                        main_row,
                        local_sends,
                        local_receives,
                        random_elements,
                        batch_size,
                    );
                },
            );

            // process padding rows
            if padding_count > 0 {
                let main_pad_row = match &main.padding_row {
                    PaddingRow::None => {
                        panic!("Main padding_row should not be None when padding count > 0")
                    }
                    PaddingRow::Zero { .. } => vec![F::zero(); main_width],
                    PaddingRow::Constant { value, .. } => vec![*value; main_width],
                    PaddingRow::General(v) => v.clone(),
                };

                let mut perm_pad_row = vec![EF::zero(); perm_width];

                populate_local_permutation_row::<F, EF>(
                    &mut perm_pad_row,
                    &[],
                    &main_pad_row,
                    local_sends,
                    local_receives,
                    random_elements,
                    batch_size,
                );

                let pad_perm_row_sum: EF = perm_pad_row.iter().copied().sum();

                let perm_main_sum: EF = perm_main.values.iter().copied().sum();

                local_cumulative_sum =
                    perm_main_sum + pad_perm_row_sum * EF::from_canonical_usize(padding_count);

                perm_padding_row = PaddingRow::General(perm_pad_row);
            } else {
                local_cumulative_sum = perm_main.values.iter().copied().sum();
                perm_padding_row = PaddingRow::None;
            }
        }
    }

    // When there is no permutation (perm_width == 0), set total_height to 0 so that
    // decompress produces a genuinely empty matrix and perm_empty checks (total_height() == 0) work.
    let effective_total_height = if perm_width == 0 { 0 } else { total_height };

    let perm_trace =
        CompressedMatrix::<EF, EF>::new(perm_main, perm_padding_row, effective_total_height);

    (perm_trace, local_cumulative_sum)
}

/// Evaluates the permutation constraints for the given chip.
///
/// In particular, the constraints checked here are:
///     - The running sum column starts at zero.
///     - That the RLC per interaction is computed correctly.
///     - The running sum column ends at the (currently) given cumulative sum.
#[allow(clippy::too_many_lines)]
pub fn eval_permutation_constraints<'a, F, AB>(
    sends: &[Interaction<F>],
    receives: &[Interaction<F>],
    batch_size: usize,
    commit_scope: InteractionScope,
    builder: &mut AB,
) where
    F: Field,
    AB::EF: ExtensionField<F>,
    AB: MultiTableAirBuilder<'a, F = F> + PairBuilder,
    AB: 'a,
{
    let empty = vec![];
    let (scoped_sends, scoped_receives) = scoped_interactions(sends, receives);
    let local_sends = scoped_sends.get(&InteractionScope::Local).unwrap_or(&empty);
    let local_receives = scoped_receives.get(&InteractionScope::Local).unwrap_or(&empty);

    let local_permutation_width =
        local_permutation_trace_width(local_sends.len() + local_receives.len(), batch_size);

    let permutation_trace_width = local_permutation_width;
    // if permutation_trace_width == 0 {
    //     println!("permutation trace width is 0");
    // }

    let preprocessed = builder.preprocessed();
    let main = builder.main();
    let perm = builder.permutation().to_row_major_matrix();

    let preprocessed_local = preprocessed.row_slice(0);
    let main_local = main.to_row_major_matrix();
    let main_local = main_local.row_slice(0);
    let main_local: &[AB::Var] = (*main_local).borrow();
    let perm_local = perm.row_slice(0);
    let perm_local: &[AB::VarEF] = (*perm_local).borrow();
    let perm_width = perm.width();

    // Assert that the permutation trace width is correct.
    if perm_width != permutation_trace_width {
        panic!(
            "permutation trace width is incorrect: expected {permutation_trace_width}, got {perm_width}",
        );
    }

    // Get the permutation challenges.
    let permutation_challenges = builder.permutation_randomness();
    let random_elements: Vec<AB::ExprEF> =
        permutation_challenges.iter().map(|x| (*x).into()).collect();

    let random_elements = &random_elements[0..2];
    let (alpha, beta) = (&random_elements[0], &random_elements[1]);
    if !local_sends.is_empty() || !local_receives.is_empty() {
        // Ensure that each batch sum m_i/f_i is computed correctly.
        let interaction_chunks = &local_sends
            .iter()
            .map(|int| (int, true))
            .chain(local_receives.iter().map(|int| (int, false)))
            .chunks(batch_size);

        // Assert that the i-eth entry is equal to the sum_i m_i/rlc_i by constraints:
        // entry * \prod_i rlc_i = \sum_i m_i * \prod_{j!=i} rlc_j over all columns of the
        // permutation trace except the last column.
        for (entry, chunk) in perm_local[0..perm_width].iter().zip(interaction_chunks) {
            // First, we calculate the random linear combinations and multiplicities with the
            // correct sign depending on wetther the interaction is a send or a receive.
            let mut rlcs: Vec<AB::ExprEF> = Vec::with_capacity(batch_size);
            let mut multiplicities: Vec<AB::Expr> = Vec::with_capacity(batch_size);
            for (interaction, is_send) in chunk {
                let mut rlc = alpha.clone();
                let mut betas = beta.powers();

                rlc = rlc.clone()
                    + betas.next().unwrap()
                        * AB::ExprEF::from_canonical_usize(interaction.argument_index());
                for (field, beta) in interaction.values.iter().zip(betas.clone()) {
                    let elem = field.apply::<AB::Expr, AB::Var>(&preprocessed_local, main_local);
                    rlc = rlc.clone() + beta * elem;
                }
                rlcs.push(rlc);

                let send_factor = if is_send { AB::F::one() } else { -AB::F::one() };
                multiplicities.push(
                    interaction
                        .multiplicity
                        .apply::<AB::Expr, AB::Var>(&preprocessed_local, main_local)
                        * send_factor,
                );
            }

            // Now we can calculate the numerator and denominator of the combined batch.
            let mut product = AB::ExprEF::one();
            let mut numerator = AB::ExprEF::zero();
            for (i, (m, rlc)) in multiplicities.into_iter().zip(rlcs.iter()).enumerate() {
                // Calculate the running product of all rlcs.
                product = product.clone() * rlc.clone();

                // Calculate the product of all but the current rlc.
                let mut all_but_current = AB::ExprEF::one();
                for other_rlc in
                    rlcs.iter().enumerate().filter(|(j, _)| i != *j).map(|(_, rlc)| rlc)
                {
                    all_but_current = all_but_current.clone() * other_rlc.clone();
                }
                numerator = numerator.clone() + AB::ExprEF::from_base(m) * all_but_current;
            }

            // Finally, assert that the entry is equal to the numerator divided by the product.
            let entry: AB::ExprEF = (*entry).into();
            builder.assert_eq_ext(product.clone() * entry.clone(), numerator);
        }
    }

    // Handle global cumulative sums.
    // If the chip's scope is `InteractionScope::Global`, the last row's final 14 columns is equal
    // to the global cumulative sum.
    let global_cumulative_sum = builder.global_cumulative_sum();
    if commit_scope == InteractionScope::Global {
        for i in 0..7 {
            builder
                .when_last_row()
                .assert_eq(main_local[main_local.len() - 14 + i], global_cumulative_sum.0.x.0[i]);
            builder
                .when_last_row()
                .assert_eq(main_local[main_local.len() - 7 + i], global_cumulative_sum.0.y.0[i]);
        }
    }
}

/// Counts the number of permutation constraints for the given chip.
///
/// IMPORTANT: This function must be manually updated if any changes are made to
/// `eval_permutation_constraints`. The current count includes:
/// - For local interactions: `local_permutation_trace_width(sends.len() + receives.len(),
///   batch_size)` + 1 (extra alpha power for the cumulative sum constraint computed
///   separately in the sumcheck prover).
/// - For global scope: 14 additional constraints for the global cumulative sum.
pub fn count_permutation_constraints<F: Field>(
    sends: &[Interaction<F>],
    receives: &[Interaction<F>],
    batch_size: usize,
    commit_scope: InteractionScope,
) -> usize {
    let mut count = 0;

    let empty = vec![];
    let (scoped_sends, scoped_receives) = scoped_interactions(sends, receives);
    let local_sends = scoped_sends.get(&InteractionScope::Local).unwrap_or(&empty);
    let local_receives = scoped_receives.get(&InteractionScope::Local).unwrap_or(&empty);

    let num_local_interactions = local_sends.len() + local_receives.len();

    if num_local_interactions > 0 {
        let local_permutation_width =
            local_permutation_trace_width(num_local_interactions, batch_size);
        count += local_permutation_width;

        // For the cumulative sum constraint computed separately in the sumcheck prover.
        count += 1;
    }

    // If the chip's scope is `InteractionScope::Global`, 14 asserts that
    // the last row's final 14 columns is equal to the global cumulative sum.
    if commit_scope == InteractionScope::Global {
        count += 14;
    }

    count
}
