#![allow(clippy::needless_range_loop)]

use std::{borrow::Borrow, ops::Deref};

use crate::operations::poseidon2_kb::permutation::{Poseidon2Cols, Poseidon2Degree3Cols};

pub mod air;
pub mod columns;
pub mod trace;

/// KoalaBear variant of the Poseidon2 wide chip. Only supports degree 3 (SBOX_DEGREE=3).
#[derive(Default, Debug, Clone, Copy)]
pub struct Poseidon2WideKbChip<const DEGREE: usize>;

impl<'a, const DEGREE: usize> Poseidon2WideKbChip<DEGREE> {
    pub(crate) fn convert<T>(row: impl Deref<Target = [T]>) -> Box<dyn Poseidon2Cols<T> + 'a>
    where
        T: Copy + 'a,
    {
        if DEGREE == 3 {
            let convert: &Poseidon2Degree3Cols<T> = (*row).borrow();
            Box::new(*convert)
        } else {
            panic!("KoalaBear mode only supports degree 3, got degree {DEGREE}");
        }
    }
}
