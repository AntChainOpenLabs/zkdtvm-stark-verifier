use crate::{
    air::{InteractionScope, MultiTableAirBuilder},
    lookup::Interaction,
};
use std::collections::HashMap;
use itertools::Itertools;
use p3_air::{AirBuilder, PairBuilder};
use p3_field::{AbstractExtensionField, AbstractField, ExtensionField, Field};
use p3_matrix::Matrix;
use std::borrow::Borrow;

/// Computes the width of the local permutation trace in terms of extension field elements.
#[must_use]
pub const fn local_permutation_trace_width(nb_interactions: usize, batch_size: usize) -> usize {
    if nb_interactions == 0 {
        return 0;
    }
    nb_interactions.div_ceil(batch_size)
}

/// Groups interactions by scope.
pub fn scoped_interactions<F: Field>(
    sends: &[Interaction<F>],
    receives: &[Interaction<F>],
) -> (HashMap<InteractionScope, Vec<Interaction<F>>>, HashMap<InteractionScope, Vec<Interaction<F>>>)
{
    let grouped_sends: HashMap<InteractionScope, Vec<Interaction<F>>> = sends
        .iter()
        .cloned()
        .into_group_map_by(|interaction| {
            if interaction.scope == InteractionScope::Global {
                InteractionScope::Global
            } else {
                InteractionScope::Local
            }
        });

    let grouped_receives: HashMap<InteractionScope, Vec<Interaction<F>>> = receives
        .iter()
        .cloned()
        .into_group_map_by(|interaction| {
            if interaction.scope == InteractionScope::Global {
                InteractionScope::Global
            } else {
                InteractionScope::Local
            }
        });

    (grouped_sends, grouped_receives)
}

/// Evaluates the permutation constraints for the given chip.
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

    if perm_width != permutation_trace_width {
        panic!(
            "permutation trace width is incorrect: expected {permutation_trace_width}, got {perm_width}",
        );
    }

    let permutation_challenges = builder.permutation_randomness();
    let random_elements: Vec<AB::ExprEF> =
        permutation_challenges.iter().map(|x| (*x).into()).collect();

    let random_elements = &random_elements[0..2];
    let (alpha, beta) = (&random_elements[0], &random_elements[1]);
    if !local_sends.is_empty() || !local_receives.is_empty() {
        let interaction_chunks = &local_sends
            .iter()
            .map(|int| (int, true))
            .chain(local_receives.iter().map(|int| (int, false)))
            .chunks(batch_size);

        for (entry, chunk) in perm_local[0..perm_width].iter().zip(interaction_chunks) {
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

            let mut product = AB::ExprEF::one();
            let mut numerator = AB::ExprEF::zero();
            for (i, (m, rlc)) in multiplicities.into_iter().zip(rlcs.iter()).enumerate() {
                product = product.clone() * rlc.clone();

                let mut all_but_current = AB::ExprEF::one();
                for other_rlc in
                    rlcs.iter().enumerate().filter(|(j, _)| i != *j).map(|(_, rlc)| rlc)
                {
                    all_but_current = all_but_current.clone() * other_rlc.clone();
                }
                numerator = numerator.clone() + AB::ExprEF::from_base(m) * all_but_current;
            }

            let entry: AB::ExprEF = (*entry).into();
            builder.assert_eq_ext(product.clone() * entry.clone(), numerator);
        }
    }

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
        count += 1;
    }

    if commit_scope == InteractionScope::Global {
        count += 14;
    }

    count
}
