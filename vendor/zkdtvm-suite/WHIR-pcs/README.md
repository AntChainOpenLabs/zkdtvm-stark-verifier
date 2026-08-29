# WHIR PCS

A standalone WHIR-style multilinear polynomial commitment scheme for zkDTVM.

This repository contains a transparent multilinear PCS built around stacked trace
commitments, WHIR-style out-of-domain sampling, gamma claim accumulation, and
cross-round IOPP commitments. The public PCS surface is intentionally small:

```text
commit -> open -> verify
```

## Features

- WHIR is the only supported algorithm in this repository.
- `MlPCS::{commit, open, verify}` is the core public interface.
- `commit` accepts one batch of possibly different-height `CompressedMatrix`
  values.
- `open` and `verify` support multiple committed batches, such as execution,
  preprocessed, and permutation traces.
- Commit-local stacking can reduce each batch to one stacked matrix via
  `MlCommitOptions`.
- Cross-round IOPP commitments use log foldings, for example `[5, 4]` means
  `32-to-1` followed by `16-to-1`.
- The per-round WHIR path uses reduced-rate codewords: a `k`-fold group folds
  the polynomial by `2^k` while shrinking the committed codeword domain by only
  `2`.
- FRI early-stop is controlled by `fri.log_final_poly_len`.
- Per-round query schedules can be configured with independent query counts and
  query proof-of-work per committed IOPP group.

## Stacking Model

Stacking is configured per commit, not globally. Use:

- `MlCommitOptions::default()` to keep the unstacked path.
- `MlCommitOptions::auto_stacking()` to stack to the tallest matrix in that
  commit batch.
- `MlCommitOptions::stacking_log_height(log_height)` to force an explicit
  stacked height.

In the stacked path, each committed batch is converted into one stacked matrix.
At opening time, the stacked matrices from all committed batches are randomly
linearly combined into one vector, and WHIR proves the opening claim for that
combined vector.

Extension-field columns should be flattened into base-field limbs before they
enter this PCS. The opened values may still be supplied as extension-field
values: the verifier accepts the layout where
`opened_values.len() * EF::D == matrix.width()`. The stacking layout keeps those
limbs aligned so one extension column remains a single logical column during
claim batching.

## WHIR Opening Path

When per-round queries are enabled on the stacked path, the opening protocol
uses:

- one out-of-domain value after each non-final committed group;
- gamma accumulation for OOD and in-domain query claims;
- independent query sampling per committed group;
- reduced-rate RS domains, where `poly_log -= k`, `codeword_log -= 1`, and the
  effective `log_blowup` increases by `k - 1` for a `k`-fold committed group;
- final-polynomial checking after early-stop.

This implementation also commits the linear combination of the stacked matrix
columns as the initial IOPP oracle. That differs from OpenVM-style raw-row
openings and keeps cross-round queries from opening large raw stacked blocks.

Path-pruning is available on both the legacy global-query stacked path and the
per-round WHIR path. In the per-round path, pruning is applied independently to
each committed IOPP group because each group may have its own query count.

## Parameters

`WhirConfig<FriMmcs>` owns the WHIR parameter set:

- `fri`: Plonky3 `FriConfig`, including `log_blowup`, `num_queries`,
  proof-of-work bits, `log_final_poly_len`, and `cross_round_log_foldings`.
- `path_pruning`: enables shared Merkle paths on the legacy global-query path.
- `with_cross_round_log_foldings(vec![...])`: convenience setter for the sparse
  IOPP commit schedule stored in `fri.cross_round_log_foldings`.
- `round_queries`: optional per-round query schedule. Use
  `with_round_query_counts(vec![...])` for query counts, or
  `with_round_queries(vec![...])` to set both query counts and query
  proof-of-work bits.

The compatibility constructor `WhirPcs::new(mmcs, fri)` is still available. It
creates a `WhirConfig` internally and reads `DT_USE_PATH_PRUNING=1` as the
compatibility switch for path-pruning.

Stacked matrix caching is enabled by default for `MlCommitOptions` stacking.
Set `WHIR_CACHE_STACKED_MATRIX=0` to trade lower peak memory for recomputing the
stacked matrix during opening. `PCS_CACHE_STACKED_MATRIX` is also accepted as a
generic fallback shared by standalone PCS crates.

## Repository Layout

```text
pcs/src/
├── whir/
│   ├── mlpcs.rs         # MlPCS trait and commit-local options
│   ├── sumcheck.rs      # sumcheck prover helpers
│   ├── whir_commit.rs   # commit and stacked commit implementation
│   ├── whir_helpers.rs  # stacking, validation, encoding, and PoW helpers
│   ├── whir_iopp.rs     # IOPP query opening and verification helpers
│   ├── whir_pcs.rs      # MlPCS implementation and unstacked path
│   ├── whir_stacked.rs  # stacked WHIR opening and verification path
│   ├── whir_types.rs    # config, proof, and prover-data types
│   └── mod.rs
├── utils/               # equality, multilinear, univariate, field, and math helpers
└── lib.rs
```

## Build And Test

```bash
cargo check -p whir-pcs
cargo test -p whir-pcs
```

The `whir` Cargo feature is currently a no-op compatibility flag.

## Public Release Notes

Before publishing this repository as a public open-source crate, add a license
file and matching Cargo metadata. The implementation also still depends on the
zkDTVM Plonky3 branch configured in the workspace manifest.

## References

- [WHIR: Reed-Solomon Proximity Testing with Super-Fast Verification](https://eprint.iacr.org/2024/1586)
- [BaseFold: Efficient Field-Agnostic Polynomial Commitment Schemes from Foldable Codes](https://eprint.iacr.org/2023/1705)
