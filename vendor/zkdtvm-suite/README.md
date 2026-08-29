# zkdtvm-suite

A zkVM (zero-knowledge virtual machine) proving stack that proves correct
execution of arbitrary RISC-V programs. It is built as a single Cargo
workspace combining three components:

- **`sp1/`** — the zkVM (executor, AIR chips, STARK prover, recursion, SDK),
  a fork of [SP1](https://github.com/succinctlabs/sp1). Internal crates are
  named `dt-*`.
- **`WHIR-pcs/`** — a standalone WHIR-style multilinear polynomial commitment
  scheme built for this suite.
- **`Plonky3/`** — an upstream PIOP toolkit (fields, FRI, Merkle trees,
  Poseidon2), an Ant-modified fork of Plonky3.

Version `5.0.0`, Rust edition 2021.

## Layout

```
zkdtvm-suite/
├── sp1/                         # zkVM (dt-* crates) — has its own nested workspace
│   ├── crates/                  # executor, machine, stark, sc_prover, recursion, sdk_sc, ...
│   └── openspec/                # design specs
├── WHIR-pcs/pcs/                # whir-pcs crate
├── Plonky3/                     # upstream fields / FRI / hashing crates
├── whir_config_<field>_ext<4|5>.json   # runtime WHIR PCS parameters
├── ARCHITECTURE.md              # component map, proving pipeline, field/PCS config
└── SECURITY.md                  # soundness model + tracked parameter gaps
```

## Fields

The challenge field is chosen at compile time via the mutually exclusive
`koalabear` (default, production) or `babybear` cargo features. Because the
field must be selected by a top-level crate, neither is a default feature.

## Building & running

```bash
# Default build (KoalaBear).
cargo build --release

# The rsp end-to-end example (KoalaBear, default):
cd sp1/examples/rsp/script && cargo run --release --bin rsp-script

# BabyBear instead of KoalaBear:
cargo run --release --no-default-features --features babybear_whir --bin rsp-script
```

WHIR PCS parameters are loaded at runtime from `whir_config_<field>_<ext>.json`
for the active field and extension degree (overridable with the
`WHIR_CONFIG_PATH` env var). See `ARCHITECTURE.md` for
the full proving pipeline and configuration model, and `SECURITY.md` for the
soundness analysis.

## Documentation

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — how the pieces fit together.
- [`SECURITY.md`](SECURITY.md) — WHIR/SWIRL soundness model and known gaps.
- [`sp1/README.md`](sp1/README.md) — SP1 zkVM details.
- [`WHIR-pcs/README.md`](WHIR-pcs/README.md) — WHIR PCS construction.
