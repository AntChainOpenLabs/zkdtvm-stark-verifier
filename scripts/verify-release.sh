#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"

cd "$ROOT_DIR"
cargo test --release --workspace
cargo run --release -p zkdtvm-stark-verifier-cli --bin inspect_fixture
cargo run --release -p zkdtvm-stark-verifier-cli --bin zkdtvm-stark-verifier -- \
  --proof proof.bin \
  --vk vk-full.bin
