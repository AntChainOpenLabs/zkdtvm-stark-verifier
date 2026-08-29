use core::borrow::Borrow;

use dt_core_machine::operations::poseidon2_kb::{
    permutation::{
        Poseidon2Cols, Poseidon2Degree3Cols, NUM_INTERNAL_ROUNDS_M1, NUM_POSEIDON2_DEGREE3_COLS,
        POSEIDON2_DEGREE3_COL_MAP,
    },
    WIDTH,
};

pub const NUM_POSEIDON2_PERMUTATION_COLS: usize = NUM_POSEIDON2_DEGREE3_COLS;
pub const POSEIDON2_MULT_COL: usize = NUM_POSEIDON2_PERMUTATION_COLS;
pub const NUM_POSEIDON2_PERMUTE_COLS: usize = NUM_POSEIDON2_PERMUTATION_COLS + 1;
pub const NUM_POSEIDON2_PERMUTE_PAYLOAD_VALUES: usize = WIDTH * 2;
pub const NUM_POSEIDON2_PERMUTE_DENOMINATOR_VALUES: usize =
    1 + NUM_POSEIDON2_PERMUTE_PAYLOAD_VALUES;

pub struct Poseidon2ColsView<'a, T: Clone> {
    permutation: &'a Poseidon2Degree3Cols<T>,
}

impl<'a, T: Clone> Poseidon2ColsView<'a, T> {
    pub fn from_slice(row: &'a [T]) -> Self {
        let permutation: &Poseidon2Degree3Cols<T> = row[..NUM_POSEIDON2_PERMUTATION_COLS].borrow();
        Self { permutation }
    }
}

impl<T: Clone> Poseidon2Cols<T> for Poseidon2ColsView<'_, T> {
    fn external_rounds_state(&self) -> &[[T; WIDTH]] {
        &self.permutation.state.external_rounds_state
    }

    fn internal_rounds_state(&self) -> &[T; WIDTH] {
        &self.permutation.state.internal_rounds_state
    }

    fn internal_rounds_s0(&self) -> &[T; NUM_INTERNAL_ROUNDS_M1] {
        &self.permutation.state.internal_rounds_s0
    }

    fn perm_output(&self) -> &[T; WIDTH] {
        &self.permutation.state.output_state
    }

    fn get_cols_mut(
        &mut self,
    ) -> (&mut [[T; WIDTH]], &mut [T; WIDTH], &mut [T; NUM_INTERNAL_ROUNDS_M1], &mut [T; WIDTH])
    {
        unreachable!("native Poseidon2 PolyAIR eval only needs an immutable column view")
    }
}

pub fn poseidon2_input_from_row<T: Clone>(row: &[T]) -> [T; WIDTH] {
    core::array::from_fn(|i| {
        row[POSEIDON2_DEGREE3_COL_MAP.state.external_rounds_state[0][i]].clone()
    })
}

pub fn poseidon2_output_from_row<T: Clone>(row: &[T]) -> [T; WIDTH] {
    core::array::from_fn(|i| row[POSEIDON2_DEGREE3_COL_MAP.state.output_state[i]].clone())
}
