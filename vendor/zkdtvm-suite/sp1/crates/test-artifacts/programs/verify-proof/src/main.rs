//! This is a test program that takes in a dt_core vkey and a list of inputs, and then verifies the
//! zkDTVM proof for each input.

#![no_main]
dt_zkvm::entrypoint!(main);

use dt_zkvm::lib::verify::verify_dt_proof;
use sha2::{Digest, Sha256};

fn words_to_bytes(words: &[u32; 8]) -> [u8; 32] {
    let mut bytes = [0u8; 32];
    for i in 0..8 {
        let word_bytes = words[i].to_le_bytes();
        bytes[i * 4..(i + 1) * 4].copy_from_slice(&word_bytes);
    }
    bytes
}

pub fn main() {
    let vkey = dt_zkvm::io::read::<[u32; 8]>();
    println!("Read vkey: {:?}", hex::encode(words_to_bytes(&vkey)));
    let inputs = dt_zkvm::io::read::<Vec<Vec<u8>>>();
    inputs.iter().for_each(|input| {
        // Get expected pv_digest hash: sha256(input)
        let pv_digest = Sha256::digest(input);
        verify_dt_proof(&vkey, &pv_digest.into());

        println!("Verified proof for digest: {:?}", hex::encode(pv_digest));
        println!("Verified input: {:?}", hex::encode(input));
    });
}
