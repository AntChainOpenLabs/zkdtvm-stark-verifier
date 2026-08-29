# Native Recursion

`native-recursion` implements the KoalaBear/ext5/WHIR compression ladder without running a
recursive VM. Its AIRs replay child STARK verification directly and produce the final
`DTReduceProof<RootSC>` consumed by the host verifier.

This crate contains no GPU implementation. Performance discussions use a hypothetical faster
prover only to identify host work that would otherwise dominate end-to-end wall time.

## Default route

The RSP and Fibonacci script packages select KoalaBear, extension degree 5, WHIR, and native
recursion by default. Always build, run, and test the repository remotely with `--release`.

```bash
cd examples/rsp/script
cargo run --release
cargo run --release -- core-out /tmp/rsp-core.bin
cargo run --release -- compress-from /tmp/rsp-core.bin
```

Backend selection is resolved in this order:

1. `DTProverOpts::recursion_backend`;
2. `DT_RECURSION_BACKEND=native|dsl`;
3. native recursion.

An invalid environment value is an error. Native-route errors propagate and never trigger an
automatic DSL or raw-core fallback.

The backend also requires `whir_config_koalabear_ext5.json`. It searches the working directory and
its ancestors, then checks the active stage parameters and the frozen L4 verifying-key digest.
Missing or inconsistent authority fails closed; compiled defaults are not a product fallback.

## Feature contract

| Feature | Meaning |
|---|---|
| `native-recursion` | Enables the native compression backend. It requires KoalaBear and ext5. |
| `koalabear` / `babybear` | Selects one base field. Do not enable both in one dependency graph. |
| `ext5` | Enables the degree-five extension required by the native machines. |
| `koalabear_whir_ext5` | Example-package bundle for network + KoalaBear + ext5. |

Library dependencies use `default-features = false`; top-level applications select the field.
Combining native recursion with BabyBear is rejected at compile time.

## Ladder topology

```text
ordered CoreSC shards
  -> lift nodes, at most 11 shards each
  -> either:
       at most 11 bare lifts -> L3
     or
       L2 nodes, 2..=11 lifts each -> L3
  -> one L4 root-shrink node
  -> DTReduceProof<RootSC>
```

The lift, L2, and L3 proofs use the Poseidon2 inner recursion configuration. L4 uses the SHA256
`RootSC` PCS/transcript configuration and the signed `stack_log_height = 18`. These are typed
in-memory proof transitions. The live ladder never serializes a proof or converts its
configuration to cross a layer boundary.

L2 arity one is forbidden. Child order is the core-shard order and is part of the replay contract.
With arity 11, up to 121 core shards feed L3 directly and the frozen three-level tree accepts at
most 1331 core shards.

## Request ownership and Poseidon2 memo

One non-`Clone`, non-serde `NativeRecursionRequest` owns the complete in-process operation. The
streamed route creates it before core shards are delivered; the raw route creates it while
normalizing the supplied core shards. The resulting `NativeCorePrerecordBatch` retains that same
request through lift, optional L2, L3, and L4 recording and proving. A layer cannot silently start
a replacement request.

The request owns the host-side Poseidon2 output memo. Each recording seed receives an
independently profiled view over the same request entries, so reuse survives parallel child
recording and all ladder layers. The lookup table is split into 64 independently locked shards,
and each view keeps atomic hit and miss counters for profiling. Entries are keyed by the complete
permutation input. Each key has its own single-assignment cell: distinct keys can compute
concurrently, while workers requesting the same key wait for the one computation and then share
its output.

Record merges also enforce request lineage. Views of the same request merge only their profiling
counters because they already share the table. An empty memo may adopt the other side's request
table, but two populated tables from different requests fail immediately; the live route never
performs an O(cache-size) cross-request union.

The memo stores computations only. Poseidon2 provider requests and their multiplicities are still
registered at their semantic sources and accumulated in the record's provider pool. There is no
process-global Poseidon2 output cache and no node-boundary cache-clear protocol. The process-wide
immutable permutation initialization is not a result cache. Ordinary record cloning and
deserialization do not carry memo state; the canonical request shares entries only through its
explicit recording views.

## Per-node pipeline

Each native node has five relevant phases:

1. **Record.** The recording verifier replays the child proof, Fiat-Shamir transcript, sumcheck,
   WHIR queries, Merkle visits, constraint DAG, and Poseidon2 requests. Exact multiplicities are
   accumulated at their semantic source. Transcript sponge rows are captured during this replay,
   alongside the transcript events that produced them.
2. **Finalize.** The sole `BuildingRecord -> FinalizedRecord` transition publishes statement
   values, checks publication completeness, registers the remaining source-owned provider
   requests and nonces, and seals the program authority once. A child without source-captured
   transcript rows is rejected here.
3. **Trace generation.** The prover accepts only that `FinalizedRecord`. Independent chips
   materialize `CompressedMatrix` traces in parallel from the sealed record; no chip waits for
   another chip to create or mutate proof input. Poseidon2, Merkle-path, WHIR, batch-sumcheck, and
   the large constraint DAG/fold generators fill pre-sized flat storage directly. Transcript
   tracegen concatenates the captured source rows; it cannot replay transcript events as a
   fallback.
4. **Commit.** Main and permutation traces are committed in protocol order. The permutation
   challenge depends on the main commitment, so this sequencing is semantic.
5. **Open.** The prover runs the constraint sumcheck and WHIR openings, including the stacked root
   checks at L4.

There is no live `prepare` phase. Proving does not rescan a mutable record, compute a bincode
fingerprint, or enter a compatibility fallback. A finalized-record generation token identifies the
specific sealed input for reporting, while the captured program authority rejects pairing it with
a different machine. Neither mechanism is a cross-proof content cache.

## Canonical streamed/raw handoff

The normal SDK route is:

```text
.compressed()
  -> prove_core_with_native_handoff()
       -> NativeRecursionRequest::new()
       shard production
         -> bounded ordered child recording
         -> prove each closed lift bin early
         -> record direct-L3 children as lift proofs complete
  -> NativeCoreHandoff {
       public values,
       NativeCorePrerecordBatch { request, ... },
     }
  -> compress_native(batch)
  -> DTReduceProof<RootSC>
```

`NativeCorePrerecordBatch` is the canonical compression input shared by both ingress routes. It is
request-owned, non-`Clone`, non-serde, and never stored in backend-global state.

An explicit raw or saved-core diagnostic uses:

```text
DTProver::compress_native_core_shards()
  -> normalize_core_shards()
       -> NativeRecursionRequest::new()
  -> NativeCorePrerecordBatch { request, ... }
  -> the same compress_native(batch)
```

The explicit diagnostic driver owns file loading and result wrapping. Normalization moves the
typed core shards through the same bounded `build_core_prerecord` per-child primitive used by the
streamed producer. It does not construct a temporary `SCMachineProof`, launch an unbounded bulk
recorder fan-out, clone proof payloads, serialize data, or convert proof configurations. Shard
buffers are released immediately after their individual child recording. Each closed eleven-shard
bin is dispatched to a bounded lift worker while the persistent recorder workers consume the next
bin. The public SDK has no saved-file compatibility facade, and the backend has no
raw-versus-streamed branch after the batch boundary.

## Scheduling and the 121/122 boundary

Core proof delivery, child recording, and early lift proving use bounded queues with backpressure.
Completion order is restored with node indices before records or proofs are merged.

CPU-parallel child recording, trace generation, and proving share one process-global Rayon pool.
The streamed/raw producer coordinators use separately bounded recorder and lift worker sets for
backpressure, but no worker creates an arity-wide nested OS-thread fan-out or a private Rayon pool.
There are no per-node Rayon thread-budget environment switches.

While the final core-shard count is not yet known, completed lifts may speculatively build their
direct-L3 child records. When shard 122 arrives, the coordinator atomically selects the L2 route,
drops completed and pending speculative L3 records, and causes later workers to skip them. Thus:

- 1..=121 shards retain ordered direct-L3 records;
- 122+ shards retain none and build L2 normally;
- speculative errors cannot invalidate a valid L2 route.

## Serialization policy

Bincode is appropriate for explicit proof persistence, the deterministic ladder setup cache, and
opt-in size diagnostics. It is not used to connect CoreSC to native recursion in one process, to
estimate shard weights, to fingerprint a recursion record, or to convert proof configurations.
Loading a saved core proof is an ingress concern only: once decoded by the driver, its typed shards
enter the same raw normalization boundary described above.

`RecordingSC` implements serde only because the generic SC configuration trait requires it. Its
wire representation contains the explicit `Core`, `Compress`, or `Shrink` recording stage; decode
reconstructs that stage's recording configuration. This trait adapter is never used to translate a
live CoreSC proof or verifying key into a recursion configuration, and it is not part of the
CoreSC-to-native handoff.

The obsolete Fibonacci native diagnostic bins that contained historical configuration-conversion
helpers were deleted. Saved Fibonacci and RSP bundles use different ELFs and verifying keys and
must not be interchanged.

## Setup cache

`NativeRecursionBackend::new` builds or loads the four ladder machines and their proving and
verifying keys. The cache directory is:

1. non-empty `DT_NATIVE_RECURSION_CACHE_DIR`, if set;
2. otherwise the platform cache directory under `zkdtvm-suite/native-recursion`.

Cache keys bind the schema, package version, setup constants, WHIR JSON contents, and frozen L4
digest. The complete typed artifact bundle is serialized once into one opaque byte payload, and
the cache records one hash over exactly that payload. Loading checks the payload hash before
deserializing and validating the artifacts. Verifying-key equality and setup determinism checks
compare typed fields directly, including map contents; they never use serialized bytes or map
iteration order as an equality surrogate. Writers use unique same-directory temporary files,
fsync the complete candidate, and publish it atomically; concurrent cold-start losers load the
complete winner. Torn envelopes and payload-hash corruption are rebuilt and atomically repaired,
while keyed metadata, typed-artifact, PK/VK, or digest incompatibilities fail closed. Per-proof
records are never cached there. A cache publish or post-publish cleanup failure is logged as a
warning and does not discard the freshly built, validated ladder context. Set
`DT_NATIVE_RECURSION_CACHE_REBUILD=1` (or `true`) to skip loading the keyed artifact and atomically
replace it after rebuilding; this is the recovery path for a semantic decode failure when the
cache schema was not bumped.

## Verification and diagnostics

Every returned root proof must pass the caller's native external check. Per-node post-prove
self-verification is an opt-in release diagnostic:

```bash
DT_NATIVE_RECURSION_POST_PROVE_VERIFY=1 cargo test --release \
  -p dt-prover --features native-recursion --lib
```

Native-recursion unit tests enable this check automatically. Expensive proof-size and structural
census diagnostics remain opt-in and must not be included in product timing claims.

The maintained performance and architecture record is
`docs/impl-reports/impl-native-recursion-core-to-trace-final-report.md`.
