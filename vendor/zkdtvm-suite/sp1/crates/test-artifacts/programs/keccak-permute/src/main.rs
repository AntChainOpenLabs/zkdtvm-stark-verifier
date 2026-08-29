#![no_main]
dt_zkvm::entrypoint!(main);

use dt_zkvm::syscalls::syscall_keccak_permute;

pub fn main() {
    for _ in 0..2 {
        let mut state = [1u64; 25];
        syscall_keccak_permute(&mut state);
        println!("{:?}", state);
    }
}
