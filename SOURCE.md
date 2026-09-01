# Source provenance

This repository contains the complete source required to build the standalone
zkdtvm v0.8.0 verifier. The `vendor/` directory contains the verifier dependency
closure used by this release. Plonky3 packages are resolved from crates.io as
exact `dt-p3-*` version `0.8.0` dependencies.

The included release fixture is generated from the bundled
`fibonacci-program`. The guest ELF is built before `ProverClient::setup` derives
the full program verifying key. `fixture-metadata.json` records the program,
proof configuration, and ELF SHA-256 so that `proof.bin`, `vk.bin`, and
`vk-full.bin` can be checked against the ELF used during setup.
