#![no_main]

dt_zkvm::entrypoint!(main);

use dt_zkvm::lib::bls12381::decompress_pubkey;

pub fn main() {
    let compressed_key: [u8; 48] = dt_zkvm::io::read_vec().try_into().unwrap();

    for _ in 0..4 {
        println!("before: {:?}", compressed_key);

        let decompressed_key = decompress_pubkey(&compressed_key).unwrap();

        println!("after: {:?}", decompressed_key);
        dt_zkvm::io::commit_slice(&decompressed_key);
    }
}
