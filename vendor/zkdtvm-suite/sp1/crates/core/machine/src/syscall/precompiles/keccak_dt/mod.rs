mod air;
pub mod columns;
mod controller;
pub mod keccak_cols;
pub mod keccak_controller_polyair;
pub mod keccak_polyair;
mod trace;

pub use controller::*;

pub const STATE_SIZE: usize = 25;
pub const STATE_NUM_WORDS: usize = STATE_SIZE * 2;

#[derive(Default)]
pub struct KeccakPermuteChip {}

impl KeccakPermuteChip {
    pub fn new() -> Self {
        Self {}
    }
}
