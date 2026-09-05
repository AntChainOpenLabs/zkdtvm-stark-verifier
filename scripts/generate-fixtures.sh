#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"

if [ "$(uname -s)" != Linux ]; then
  echo 'Generate proof/artifact fixtures on the remote Linux host.' >&2
  exit 1
fi
if [ "$#" -ne 2 ]; then
  echo 'Usage: generate-fixtures.sh <remote-generated-q131-raw-proof> <rsp-application-elf>' >&2
  exit 2
fi
cd "$ROOT_DIR"
# This script refreshes the pinned release fixture, not an arbitrary proof/metadata pair.
proof_sha=$(sha256sum "$1")
if [ "${proof_sha%% *}" != f052ac62b49524e8b47ad76f9450b8e89afd7a12563eebd7f9a64e1b91d73908 ]; then
  echo 'Input proof does not match the pinned q131 release fixture.' >&2
  exit 1
fi
fixture_dir=$(mktemp -d)
cargo run --release -p zkdtvm-stark-verifier --bin build_l4_verifier_artifact -- \
  "$fixture_dir/artifact.bin" "$2" "$fixture_dir/vk-full.bin"
cmp crates/verify/artifacts/l4-q131-full.bin "$fixture_dir/artifact.bin"
cargo run --release -p zkdtvm-stark-verifier-cli --bin zkdtvm-stark-verifier -- \
  --proof "$1" --vk "$fixture_dir/vk-full.bin"
cp "$1" proof.bin
cp "$fixture_dir/vk-full.bin" vk-full.bin
cargo run --release -p zkdtvm-stark-verifier-cli --bin inspect_fixture -- \
  --write-vk-digest vk.bin
echo "Validated fixture source: $1; temporary exports retained at $fixture_dir"
