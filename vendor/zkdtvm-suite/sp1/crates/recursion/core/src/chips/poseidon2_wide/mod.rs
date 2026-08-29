#![allow(clippy::needless_range_loop)]

use std::{borrow::Borrow, ops::Deref};

use dt_core_machine::operations::poseidon2::permutation::{
    Poseidon2Cols, Poseidon2Degree3Cols, Poseidon2Degree9Cols,
};

pub mod air;
pub mod columns;
pub mod trace;

/// A chip that implements addition for the opcode Poseidon2Wide.
#[derive(Default, Debug, Clone, Copy)]
pub struct Poseidon2WideChip<const DEGREE: usize>;

impl<'a, const DEGREE: usize> Poseidon2WideChip<DEGREE> {
    /// Transmute a row it to an immutable [`Poseidon2Cols`] instance.
    pub(crate) fn convert<T>(row: impl Deref<Target = [T]>) -> Box<dyn Poseidon2Cols<T> + 'a>
    where
        T: Copy + 'a,
    {
        if DEGREE == 3 {
            let convert: &Poseidon2Degree3Cols<T> = (*row).borrow();
            Box::new(*convert)
        } else if DEGREE == 9 || DEGREE == 17 {
            let convert: &Poseidon2Degree9Cols<T> = (*row).borrow();
            Box::new(*convert)
        } else {
            panic!("Unsupported degree: {DEGREE}");
        }
    }
}
