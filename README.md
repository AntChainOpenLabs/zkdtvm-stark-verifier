# zkdtvm-stark-verifier

Standalone verifier for the zkdtvm STARK proof system.

This project extracts the verification logic from the full zkdtvm into a self-contained library and CLI tool. The goal is to provide a minimal, auditable verification path that can be independently reviewed and integrated into external systems.

## Architecture

```text
zkdtvm-stark-verifier/
├── Cargo.toml              # Workspace root
├── proof.bin               # Pre-generated compressed proof fixture
├── vk.bin                  # Verifying key fixture
├── message.bin             # Message fixture
├── crates/
│   ├── verify/             # Core verification library
│   │   └── src/
│   │       ├── lib.rs       # Public API & re-exports
│   │       ├── types.rs     # DTVerifyingKey, DTProof, HashableKey
│   │       └── verify.rs    # verify_compressed() entry point
│   ├── stark/              # STARK machine + sumcheck verifier
│   ├── basefold/           # Basefold multilinear PCS
│   ├── recursion/          # Recursion AIR chips
│   ├── primitives/         # Poseidon2, field types
│   └── derive/             # Proc macros
└── cli/                    # CLI binary
    └── src/
        └── main.rs          # Proof loading & verification
```

## Dependencies

Plonky3 crates are published on [crates.io](https://crates.io) as `dt-p3-*` (v0.2.3-dt). No internal repository access is required.

## Build

```bash
cargo build --release
```

## Usage

### As a library

```rust
use zkdtvm_stark_verifier::{verify_compressed, DTVerifyingKey, DTReduceProof};

fn verify(proof: &DTReduceProof<_>, vk: &DTVerifyingKey) {
    verify_compressed(proof, vk).expect("verification failed");
}
```

### CLI

```bash
# Verify the included fixture proof
zkdtvm-stark-verifier --proof proof.bin --vk vk.bin --message message.bin

# Or after cargo build --release:
./target/release/zkdtvm-stark-verifier --proof proof.bin --vk vk.bin --message message.bin
```

The CLI expects bincode-serialized `DTProofWithPublicValues<DTProof>` and `DTVerifyingKey` files.

## Test

```bash
# Run all tests (uses included fixture files)
cargo test --release
```

## Fixture Files

Three pre-generated binary fixtures are included in the project root:

| File          | Description                                            |
| ------------- | ------------------------------------------------------ |
| `proof.bin`   | Compressed proof (fibonacci(10))                       |
| `vk.bin`      | Verifying key                                          |
| `message.bin` | Message payload                                        |

## Design Decisions

- **Field**: KoalaBear (31-bit prime) — the only supported field
- **PCS**: Basefold — the only supported polynomial commitment scheme

## Acknowledgements

zkdtvm-stark-verifier is the standalone verification component of our zkdtvm system, a zero-knowledge virtual machine built for verifiable computation. During the development of zkdtvm, we relied heavily on and learned from several outstanding open-source projects in the ZK ecosystem. We would like to express our gratitude:

- [Plonky3](https://github.com/Plonky3/Plonky3): Our STARK proving and verification stack is built on top of the Plonky3 library. We forked and extended several of its core crates — including field arithmetic, matrix operations, FRI, and Merkle tree primitives — to support the specific needs of our proof system. We are grateful for its clean, modular architecture at the polynomial IOP level.

- [SP1](https://github.com/succinctlabs/sp1): The overall zkVM architecture of this project draws significant inspiration from SP1. Many of our design choices around the STARK machine structure, recursion framework, AIR chip layout, and proof composition pipeline were informed by studying SP1's implementation. We appreciate the Succinct team for open-sourcing their work and advancing the state of zkVM engineering.

## License

This project is licensed under the [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0).
