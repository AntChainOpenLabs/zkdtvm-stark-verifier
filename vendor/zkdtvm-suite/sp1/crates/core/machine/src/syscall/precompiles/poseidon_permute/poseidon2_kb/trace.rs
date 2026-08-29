use core::mem::MaybeUninit;

use p3_field::Field;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixViewMut};
use p3_maybe_rayon::{iter::repeat, prelude::*};
use tracing::instrument;

use super::{
    columns::{num_cols, Poseidon2Cols, HFROUNDS, NUM_EXTERNAL_ROUNDS, NUM_INTERNAL_ROUNDS, WIDTH},
    RoundConstants,
};

/// KoalaBear Poseidon2 internal diagonal for width 24 (mat_diag_minus_1 values).
const INTERNAL_DIAG_KB_24: [u32; 24] = {
    const O: u32 = 0x7F000001;
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
        O - ((O - 1) >> 5),
        O - ((O - 1) >> 6),
        O - ((O - 1) >> 24),
        (O - 1) >> 8,
        (O - 1) >> 3,
        (O - 1) >> 4,
        (O - 1) >> 5,
        (O - 1) >> 6,
        (O - 1) >> 7,
        (O - 1) >> 9,
        (O - 1) >> 24,
    ]
};

#[instrument(name = "generate Poseidon2 KB trace", skip_all)]
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
    input: [F; WIDTH],
    constants: &RoundConstants<F>,
) {
    perm.export.write(F::one());
    for (dest, &src) in perm.inputs.iter_mut().zip(input.iter()) {
        dest.write(src);
    }

    // external_rounds_state[0] = input
    write_state(&mut perm.external_rounds_state[0], &input);

    let mut state = input;

    // First half of external rounds (0 .. HFROUNDS)
    for r in 0..HFROUNDS {
        if r == 0 {
            external_linear_layer(&mut state);
        }
        for i in 0..WIDTH {
            state[i] += constants.beginning_full_round_constants[r][i];
        }
        for s in state.iter_mut() {
            *s = *s * *s * *s;
        }
        external_linear_layer(&mut state);

        if r < HFROUNDS - 1 {
            write_state(&mut perm.external_rounds_state[r + 1], &state);
        } else {
            write_state(&mut perm.internal_rounds_state, &state);
        }
    }

    // Internal rounds
    for r in 0..NUM_INTERNAL_ROUNDS {
        let add_rc = state[0] + constants.partial_round_constants[r];
        state[0] = add_rc * add_rc * add_rc;
        internal_linear_layer(&mut state);
        if r < NUM_INTERNAL_ROUNDS - 1 {
            perm.internal_rounds_s0[r].write(state[0]);
        }
    }

    // Write post-internal-rounds state as start of second half
    write_state(&mut perm.external_rounds_state[HFROUNDS], &state);

    // Second half of external rounds (HFROUNDS .. NUM_EXTERNAL_ROUNDS)
    for r in 0..HFROUNDS {
        for i in 0..WIDTH {
            state[i] += constants.ending_full_round_constants[r][i];
        }
        for s in state.iter_mut() {
            *s = *s * *s * *s;
        }
        external_linear_layer(&mut state);

        if r < HFROUNDS - 1 {
            write_state(&mut perm.external_rounds_state[HFROUNDS + r + 1], &state);
        } else {
            write_state(&mut perm.output_state, &state);
        }
    }
}

#[inline]
fn write_state<F: Copy>(dest: &mut [MaybeUninit<F>; WIDTH], src: &[F; WIDTH]) {
    for (d, &s) in dest.iter_mut().zip(src.iter()) {
        d.write(s);
    }
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

    input.iter_mut().zip(INTERNAL_DIAG_KB_24).skip(3).for_each(|(val, diag_elem)| {
        *val = full_sum + *val * F::from_canonical_u32(diag_elem);
    });
}
