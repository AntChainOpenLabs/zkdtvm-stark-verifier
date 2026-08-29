# zkdtvm-stark-verifier

Standalone verifier for the zkdtvm STARK proof system.

This project extracts the verification logic from the full zkdtvm into a self-contained library and CLI tool. The goal is to provide a minimal, auditable verification path that can be independently reviewed and integrated into external systems.

This v0.8.0 release targets the native-recursion verifier stack from
`zkdtvm-suite/banjie-dev` commit
`82a57cadf6921e4fb45181d98f1a5af0148ab491`.

## Architecture

```text
zkdtvm-stark-verifier/
├── Cargo.toml              # Workspace root
├── proof.bin               # Pre-generated compressed proof fixture
├── vk-full.bin             # Full verifying key fixture
├── vk.bin                  # Verifying key digest fixture
├── fixture-metadata.json   # Suite commit and ELF SHA-256 provenance
├── message.bin             # Message fixture
├── whir_config_koalabear_ext5.json # Sole supported WHIR parameter profile
├── crates/
│   ├── verify/             # Core verification library
│   │   └── src/
│   │       ├── lib.rs       # Public API & re-exports
│   │       ├── types.rs     # DTVerifyingKey, DTProof, HashableKey
│   │       └── verify.rs    # verify_compressed() entry point
├── vendor/                 # Minimal zkdtvm-suite verifier dependency snapshot
└── cli/                    # CLI binary
    └── src/
        └── main.rs          # Proof loading & verification
```

## Dependencies

The verifier uses a vendored snapshot of the latest `zkdtvm-suite` verifier
stack. Its Plonky3 layer is provided by 20 exact crates.io dependencies named
`dt-p3-*` at version `0.8.0`; there is no local or internal Git dependency on
Plonky3. The verifier supports only the KoalaBear degree-5 challenge field and
uses the suite's `whir_config_koalabear_ext5.json`, whose key settings are:

- `NUM_SKIP_ROUNDS = 1`
- `CHIP_LOG_HEIGHT_THRESHOLD = 0`
- root shrink `log_final_poly_len = 6`

## Build

```bash
cargo build --release
```

## Usage

### As a library

```rust
use zkdtvm_stark_verifier::{verify_compressed, DTVerifyingKey, DTReduceProof, RootSC};

fn verify(proof: &DTReduceProof<RootSC>, vk: &DTVerifyingKey) {
    verify_compressed(proof, vk).expect("verification failed");
}
```

### CLI

```bash
# Verify the included fixture proof
zkdtvm-stark-verifier --proof proof.bin --vk vk-full.bin

# Or after cargo build --release:
./target/release/zkdtvm-stark-verifier --proof proof.bin --vk vk-full.bin
```

The CLI expects a bincode-serialized `DTReduceProof<RootSC>` proof and a full
bincode-serialized `DTVerifyingKey`. A 32-byte digest is not sufficient for the
native-recursion external checks, so use `vk-full.bin` for verification.

## Test

```bash
# Run verifier tests (uses included fixture files)
cargo test --release -p zkdtvm-stark-verifier -p zkdtvm-stark-verifier-cli
```

The repository also provides `scripts/verify-release.sh`, which runs the complete
release-profile fixture and CLI verification sequence.

## Regenerating fixtures

After synchronizing this repository to the approved remote build server, run:

```bash
./scripts/generate-fixtures.sh
./scripts/verify-release.sh
```

Both scripts use Cargo's release profile. The generator first builds the latest
vendored `fibonacci-program` ELF, calls `setup` with that ELF, and writes the ELF
SHA-256 to `fixture-metadata.json` alongside the regenerated proof and keys.

## Fixture Files

Pre-generated binary fixtures are included in the project root:

| File          | Description                                                  |
| ------------- | ------------------------------------------------------------ |
| `proof.bin`   | Compressed `RootSC` proof generated from the pinned suite snapshot |
| `vk-full.bin` | Full bincode-serialized `DTVerifyingKey` derived from the latest ELF |
| `vk.bin`      | Verifying key digest (`[u32; 8]`)                            |
| `message.bin` | Optional message payload                                     |
| `fixture-metadata.json` | Suite commit, program, and ELF SHA-256 provenance   |

## Design Decisions

- **Field**: KoalaBear extension degree 5
- **PCS**: mixed native-recursion stack, using Jagged + FRI and SWIRL + WHIR

## Acknowledgements

zkdtvm-stark-verifier is the standalone verification component of our zkdtvm system, a zero-knowledge virtual machine built for verifiable computation. During the development of zkdtvm, we relied heavily on and learned from several outstanding open-source projects in the ZK ecosystem. We would like to express our gratitude:

- [Plonky3](https://github.com/Plonky3/Plonky3): Our STARK proving and verification stack is built on top of the Plonky3 library. We forked and extended several of its core crates — including field arithmetic, matrix operations, FRI, and Merkle tree primitives — to support the specific needs of our proof system. We are grateful for its clean, modular architecture at the polynomial IOP level.

- [SP1](https://github.com/succinctlabs/sp1): The overall zkVM architecture of this project draws significant inspiration from SP1. Many of our design choices around the STARK machine structure, recursion framework, AIR chip layout, and proof composition pipeline were informed by studying SP1's implementation. We appreciate the Succinct team for open-sourcing their work and advancing the state of zkVM engineering.

## License

This project is licensed under the [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0).
