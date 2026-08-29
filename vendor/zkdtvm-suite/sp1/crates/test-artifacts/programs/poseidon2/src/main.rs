#![no_main]
dt_zkvm::entrypoint!(main);
use dt_primitives::runtime_poseidon2_init;
use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, PrimeField32};
use p3_symmetric::Permutation;
const WIDTH: usize = 24;

pub fn main() {
    let perm = runtime_poseidon2_init();
    let state_u32 = dt_zkvm::io::read::<[u32; WIDTH]>();

    let mut state: [BabyBear; WIDTH] = state_u32
        .iter()
        .map(|&x| BabyBear::from_canonical_u32(x))
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();

    perm.permute_mut(&mut state);

    let output: [u32; WIDTH] =
        state.iter().map(|x| x.as_canonical_u32()).collect::<Vec<_>>().try_into().unwrap();
    dt_zkvm::io::commit(&output);
}
