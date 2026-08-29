# Architecture

zkdtvm-suite is a zkVM (zero-knowledge virtual machine) proving stack built on
top of a fork of [SP1](https://github.com/succinctlabs/sp1) and
[Plonky3](https://github.com/Plonky3/Plonky3), with a custom WHIR polynomial
commitment scheme. It proves correct execution of arbitrary RISC-V programs.

## Workspace layout

The repository is a single Cargo workspace (`resolver = "2"`, version `5.0.0`,
edition 2021) with three top-level component trees:

| Directory   | Role |
|-------------|------|
| `sp1/`      | The zkVM itself: executor, AIR chips, STARK prover, recursion, SDK. SP1 fork; all internal crates are named `dt-*`. |
| `WHIR-pcs/` | Standalone WHIR-style multilinear polynomial commitment scheme (`whir-pcs` crate). `commit -> open -> verify`. |
| `Plonky3/`  | Upstream PIOP toolkit (fields, FRI, Merkle trees, Poseidon2, uni-stark). Ant-modified fork of Polygon Zero's Plonky3. |

`sp1/` also declares its own nested `[workspace]` so it can be built
standalone; the root workspace references the same crates via `sp1/crates/...`
paths. This is inherited from upstream SP1 and is intentional.

## sp1 crate map

Proving-path crates (the hot path, in rough data-flow order):

| Crate (package)                 | Responsibility |
|---------------------------------|----------------|
| `core/executor` (dt-core-executor) | RISC-V zkVM executor — runs the program, emits the execution record. |
| `core/machine` (dt-core-machine)   | CPU + precompile AIR chips: column layouts, constraints, trace generation. |
| `stark` (dt-stark)                 | STARK proving/verifying primitives; per-field Poseidon2 + PCS config. |
| `sc_prover` (dt-prover)            | End-to-end prover orchestrating the staged pipeline (see below). |
| `recursion/compiler` (dt-recursion-compiler) | Compiles recursion programs to circuit ASM. |
| `recursion/core` (dt-recursion-core)         | Recursion AIR definitions + runtime. |
| `recursion/circuit` (dt-recursion-circuit)   | Recursive verification circuit (verifies a proof inside a proof). |
| `recursion/gnark-ffi` (dt-recursion-gnark-ffi) | gnark FFI for Groth16/PLONK BN254 final wrapping. |
| `sdk_sc` (dt-sdk)                  | Public client API: `execute`, `setup`, `prove`, `verify`. |
| `verifier` (dt-verifier)          | no-std Groth16/PLONK BN254 proof verifier. |

Support crates: `primitives` (dt-primitives, shared types incl. the `SCField`
alias), `curves` (dt-curves), `derive` / `recursion/derive` (proc-macros),
`build` / `cli` / `helper` (tooling), `zkvm/entrypoint` + `zkvm/lib` (guest
runtime), `test-artifacts` + `core/test-elf` (test ELFs).

## Proving pipeline

Defined in `sp1/crates/sc_prover/src/lib.rs`. Stages:

1. **Core (shard proofs)** — split the RISC-V execution into shards and prove
   each shard.
2. **Compress** — recursively reduce the shard proofs to a single proof.
3. **Shrink / Root-shrink** — final recursion layers with dedicated FRI
   profiles tuned for proof size.
4. **Wrap** — wrap into a SNARK-friendly field, then into a Groth16/PLONK
   BN254 proof.

Each stage has its own PCS configuration (blowup rate, query counts, grinding).
See "Field & PCS configuration" below.

## Field & PCS configuration

The challenge field is selected at compile time by **mutually exclusive**
cargo features, `koalabear` or `babybear` (KoalaBear is the production
default). Because many crates depend on `dt-stark` transitively, neither field
is a default feature — the top-level crate (`dt-sdk` / `dt-prover` /
`dt-recursion-circuit`) must pick one, or `dt-primitives` raises a
`compile_error!`. `primitives::SCField` resolves to the chosen field.

The WHIR PCS uses a stacked-trace, cross-round-IOPP construction (see
`WHIR-pcs/README.md`). Per-stage parameters (blowup, `num_queries`,
`num_committed_groups`, grinding bits) are loaded at runtime from JSON:

| File | Field / extension | Target |
|------|-------------------|--------|
| `whir_config_koalabear_ext4.json` | KoalaBear⁴ (~124-bit) | 100-bit (default) |
| `whir_config_koalabear_ext5.json` | KoalaBear⁵ (~155-bit) | 128-bit |
| `whir_config_babybear_ext4.json`  | BabyBear⁴ (~124-bit)  | 100-bit |
| `whir_config_babybear_ext5.json`  | BabyBear⁵ (~155-bit)  | 128-bit |

The active file is `whir_config_<field>_<ext>.json` by default, selected from
the active field and extension-degree features, and overridable via the
`WHIR_CONFIG_PATH` env var. Missing fields fall back to hardcoded defaults
(priority: env var > JSON > code default). Query counts depend only on rate +
regime, not on the field kind, so the two ext4 files share identical
`round_query_counts`. See `SECURITY.md` for the soundness model and a known
BabyBear⁴ parameter gap.

The `path_pruning` flag (a Merkle-path-sharing switch that trims proof size)
is configured per stage. Resolution priority is `WHIR_<STAGE>_PATH_PRUNING` /
`DT_USE_PATH_PRUNING` env var (`1`/`true`/`on`/`yes` vs `0`/`false`/`off`/`no`)
> per-stage JSON `path_pruning` > `false`. The per-stage parameters above
(blowup, queries, grinding, committed groups) are read per-stage;
`num_skip_rounds` and `chip_log_height_threshold` are top-level.

An optional per-stage `stack_log_height` (KoalaBear only) tunes WHIR
commit-local stacking; priority is `WHIR_*_STACK_LOG_HEIGHT` env > JSON >
None (auto-stack to the tallest matrix). It is a performance knob, so the
shipped configs leave it unset (auto).

Each stage also has a per-stage `stacking` flag selecting the WHIR commit
path: `true` (default) uses the commit-local **stacked** path (one stacked
matrix per batch, per-round query schedule via `round_query_counts` /
`num_committed_groups`); `false` uses the legacy **non-stacking** path
(trace groups injected tallest-first via `merge_beta`, a single global
arity-2 query phase driven by `num_queries` — the per-round schedule is
ignored there). Priority is `WHIR_<STAGE>_STACKING` / `WHIR_STACKING` env
(`1`/`true`/`on`/`yes` vs `0`/`false`/`off`/`no`) > JSON `stacking` > `true`.

## Further reading

- `sp1/README.md` — SP1 zkVM overview and usage.
- `sp1/openspec/specs/` — deeper design specs (execution-engine, proof-system,
  circuit-design, recursion-compression, field-cryptography, …).
- `WHIR-pcs/README.md` — WHIR PCS construction and public interface.
- `SECURITY.md` — WHIR/SWIRL soundness model and tracked parameter gaps.
