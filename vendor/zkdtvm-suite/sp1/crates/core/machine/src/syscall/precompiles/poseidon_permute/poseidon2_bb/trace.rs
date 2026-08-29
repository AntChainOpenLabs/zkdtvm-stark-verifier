use core::mem::MaybeUninit;

use p3_field::Field;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixViewMut};
use p3_maybe_rayon::{iter::repeat, prelude::*};
use tracing::instrument;

use super::{
    columns::{num_cols, Poseidon2Cols, HFROUNDS, PROUNDS, WIDTH},
    FullRound, PartialRound, RoundConstants, SBox,
};

/// BabyBear Poseidon2 internal diagonal for width 24 (canonical u32 values).
const INTERNAL_DIAG_BB_24: [u32; 24] = {
    const O: u32 = 0x78000001; // BabyBear::ORDER_U32
    [
        O - 2,
        1,
        2,
        (O + 1) >> 1,
        3,
        4,
        (O - 1) >> 1,
        O - 3,
        O - 4,
        O - ((O - 1) >> 8),
        O - ((O - 1) >> 2),
        O - ((O - 1) >> 3),
        O - ((O - 1) >> 4),
        O - ((O - 1) >> 7),
        O - ((O - 1) >> 9),
        O - 15,
        (O - 1) >> 8,
        (O - 1) >> 2,
        (O - 1) >> 3,
        (O - 1) >> 4,
        (O - 1) >> 5,
        (O - 1) >> 6,
        (O - 1) >> 7,
        15,
    ]
};

#[instrument(name = "generate Poseidon2 BB trace", skip_all)]
pub fn generate_trace_rows<F: Field>(
    inputs: Vec<[F; WIDTH]>,
    constants: &RoundConstants<F>,
) -> RowMajorMatrix<F> {
    let n = inputs.len().next_power_of_two();

    let ncols = num_cols();
    let mut vec = Vec::with_capacity(n * ncols);
    let trace: &mut [MaybeUninit<F>] = &mut vec.spare_capacity_mut()[..n * ncols];
    let trace: RowMajorMatrixViewMut<MaybeUninit<F>> = RowMajorMatrixViewMut::new(trace, ncols);

    let (prefix, perms, suffix) =
        unsafe { trace.values.align_to_mut::<Poseidon2Cols<MaybeUninit<F>>>() };
    assert!(prefix.is_empty(), "Alignment should match");
    assert!(suffix.is_empty(), "Alignment should match");
    assert_eq!(perms.len(), n);
    let num_padding_inputs = n - inputs.len();
    let padded_inputs =
        inputs.into_par_iter().chain(repeat([F::zero(); WIDTH]).take(num_padding_inputs));
    perms.par_iter_mut().zip(padded_inputs).for_each(|(perm, input)| {
        generate_trace_rows_for_perm(perm, input, constants);
    });

    unsafe {
        vec.set_len(n * ncols);
    }

    RowMajorMatrix::new(vec, ncols)
}

fn generate_trace_rows_for_perm<F: Field>(
    perm: &mut Poseidon2Cols<MaybeUninit<F>>,
    mut state: [F; WIDTH],
    constants: &RoundConstants<F>,
) {
    perm.export.write(F::one());
    perm.inputs.iter_mut().zip(state.iter()).for_each(|(input, &x)| {
        input.write(x);
    });

    external_linear_layer(&mut state);

    for (full_round, constants) in
        perm.beginning_full_rounds.iter_mut().zip(&constants.beginning_full_round_constants)
    {
        generate_full_round(&mut state, full_round, constants);
    }

    for (partial_round, constant) in
        perm.partial_rounds.iter_mut().zip(&constants.partial_round_constants)
    {
        generate_partial_round(&mut state, partial_round, *constant);
    }

    for (full_round, constants) in
        perm.ending_full_rounds.iter_mut().zip(&constants.ending_full_round_constants)
    {
        generate_full_round(&mut state, full_round, constants);
    }
}

#[inline]
fn generate_full_round<F: Field>(
    state: &mut [F; WIDTH],
    full_round: &mut FullRound<MaybeUninit<F>>,
    round_constants: &[F; WIDTH],
) {
    for (state_i, const_i) in state.iter_mut().zip(round_constants) {
        *state_i += *const_i;
    }
    for (state_i, sbox_i) in state.iter_mut().zip(full_round.sbox.iter_mut()) {
        generate_sbox(sbox_i, state_i);
    }
    external_linear_layer(state);
    full_round.post.iter_mut().zip(*state).for_each(|(post, x)| {
        post.write(x);
    });
}

#[inline]
fn generate_partial_round<F: Field>(
    state: &mut [F; WIDTH],
    partial_round: &mut PartialRound<MaybeUninit<F>>,
    round_constant: F,
) {
    state[0] += round_constant;
    generate_sbox(&mut partial_round.sbox, &mut state[0]);
    partial_round.post_sbox.write(state[0]);
    internal_linear_layer(state);
}

/// Degree-7 S-box: x -> x^7 via intermediate x^3.
#[inline]
fn generate_sbox<F: Field>(sbox: &mut SBox<MaybeUninit<F>>, x: &mut F) {
    let x3 = x.cube();
    sbox.0[0].write(x3);
    *x = x3 * x3 * *x;
}

fn external_linear_layer<F: Field>(state: &mut [F; WIDTH]) {
    for chunk in state.chunks_exact_mut(4) {
        let t01 = chunk[0] + chunk[1];
        let t23 = chunk[2] + chunk[3];
        let t0123 = t01 + t23;
        let t01123 = t0123 + chunk[1];
        let t01233 = t0123 + chunk[3];
        chunk[3] = t01233 + chunk[0].double();
        chunk[1] = t01123 + chunk[2].double();
        chunk[0] = t01123 + t01;
        chunk[2] = t01233 + t23;
    }
    let sums: [F; 4] =
        core::array::from_fn(|k| (0..WIDTH).step_by(4).map(|j| state[j + k]).sum::<F>());
    state.iter_mut().enumerate().for_each(|(i, elem)| *elem += sums[i % 4]);
}

fn internal_linear_layer<F: Field>(input: &mut [F; WIDTH]) {
    let part_sum: F = input[1..].iter().copied().sum();
    let full_sum = part_sum + input[0];

    input[0] = part_sum - input[0];
    input[1] = full_sum + input[1];
    input[2] = full_sum + input[2].double();

    input.iter_mut().zip(INTERNAL_DIAG_BB_24).skip(3).for_each(|(val, diag_elem)| {
        *val = full_sum + *val * F::from_canonical_u32(diag_elem);
    });
}
