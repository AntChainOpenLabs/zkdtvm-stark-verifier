#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"

cargo run \
  --release \
  --manifest-path "$ROOT_DIR/vendor/zkdtvm-suite/sp1/Cargo.toml" \
  -p fibonacci-script \
  --bin export_verifier_fixture \
  -- "$ROOT_DIR"
