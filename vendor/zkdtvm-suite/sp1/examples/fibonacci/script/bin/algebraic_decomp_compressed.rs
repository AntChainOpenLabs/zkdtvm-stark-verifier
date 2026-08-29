use dt_sdk::{include_elf, utils, DTStdin, ProverClient};
use dt_stark::sumcheck::use_algebraic_decomp;

/// The ELF we want to execute inside the zkVM.
const ELF: &[u8] = include_elf!("fibonacci-program");

fn main() {
    utils::setup_logger();

    let config_path = std::env::var("WHIR_CONFIG_PATH").unwrap_or_else(|_| "<default>".to_string());
    println!("WHIR_CONFIG_PATH={config_path}; use_algebraic_decomp={}", use_algebraic_decomp());

    let n = 500u32;
    let mut stdin = DTStdin::new();
    stdin.write(&n);

    let client = ProverClient::from_env();
    let (pk, vk) = client.setup(ELF);
    let mut proof = client.prove(&pk, &stdin).compressed().run().unwrap();

    let a = proof.public_values.read::<u32>();
    let b = proof.public_values.read::<u32>();
    println!("a: {a}, b: {b}");

    client.verify(&proof, &vk).expect("verification failed");

    println!("successfully generated and verified proof for the program!")
}
