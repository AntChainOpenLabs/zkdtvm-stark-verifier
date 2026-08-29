#![no_main]
dt_zkvm::entrypoint!(main);

use dt_zkvm::syscalls::syscall_poseidon2_permute;
pub fn main() {
    let mut state =
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23u32];
    syscall_poseidon2_permute(&mut state);
    syscall_poseidon2_permute(&mut state);
    syscall_poseidon2_permute(&mut state);
    syscall_poseidon2_permute(&mut state);
    syscall_poseidon2_permute(&mut state);
    syscall_poseidon2_permute(&mut state);
    println!("{:?}", state);
}
