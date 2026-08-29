# Security Notes

This document tracks known security-parameter gaps and the soundness model
used to derive the WHIR PCS parameters. It exists so that gaps documented only
in code comments do not get lost.

## WHIR / SWIRL parameter model

All `whir_config_*.json` files in the repository root are derived under the
**SWIRL round-by-round soundness model** (the same model `soundcalc` applies to
OpenVM2), not the plain DEEP-ALI FRI model. Key facts:

- **Query counts depend only on rate + regime, not on the field kind.** The
  per-stage `round_query_counts` are therefore identical between
  `whir_config_koalabear_ext4.json` and `whir_config_babybear_ext4.json`
  (and likewise for ext5).
- **Field sizes** (challenge field):
  - KoalaBear⁴ ≈ 123.95 bit, BabyBear⁴ ≈ 123.63 bit (cannot reach 128-bit)
  - KoalaBear⁵ ≈ 154.94 bit, BabyBear⁵ ≈ 154.53 bit (can reach 128-bit)
- **ext4 targets 100-bit, ext5 targets 128-bit.** ext5 is reached purely by
  raising `num_queries`; all grinding / PoW bits stay ≤ 20.

## Open gap: BabyBear⁴ 100-bit needs params not expressible in the JSON schema

**Status: OPEN — tracked here, not yet enforced in code.**

`whir_config_babybear_ext4.json` reaches 100-bit in the soundcalc SWIRL model
only with two extra SWIRL-layer grinding parameters that the current
`FriConfig` JSON schema cannot express:

| Parameter        | Required value (BabyBear⁴) | KoalaBear⁴ value | Why |
|------------------|---------------------------|------------------|-----|
| logup PoW bits   | **11**                    | 10               | BabyBear⁴ field is ~0.3 bit smaller; logup soundness `≈ \|F\| - log2(2·interactions) - msg_len + pow` falls to 99-bit at pow=10. |
| OOD grinding bits| **1** (compress, shrink)  | 0                | OOD error `L²·2^m/(2\|F\|)` is field-dependent; in the smaller BabyBear⁴ field it drops to 99-bit without 1 bit of grinding. |

### Why the JSON cannot express these

`FriConfig` (Plonky3/fri/src/config.rs) exposes only:
`log_blowup, num_queries, grinding_bits_query, grinding_bits_batching,
grinding_bits_folding, log_final_poly_len, cross_round_log_foldings,
num_committed_groups`.

- **OOD grinding** is applied inside the WHIR verifier (`_epsilon_out`) and has
  no `FriConfig` field, so it cannot be set from the JSON.
- **logup PoW** is a SWIRL/LogUp-layer parameter, outside the WHIR `FriConfig`
  entirely.

### Impact

Without these two adjustments, BabyBear⁴ stages compress/shrink land at
**99-bit** instead of the 100-bit target — a 1-bit shortfall, purely from the
~0.3-bit-smaller field. KoalaBear⁴ is unaffected (uses pow=10, OOD=0).

### Shared edge case (both fields)

`root_shrink` reaches only **99-bit** in the SWIRL model for *both* KoalaBear⁴
and BabyBear⁴, because the blowup=4 JBR folding error caps there with
`grinding_bits_folding=20`. This is a known shared 1-bit edge and is not
BabyBear-specific.

### Resolution options (not yet done)

1. Extend `FriConfig` (and the JSON schema) with `grinding_bits_ood` and plumb a
   logup-PoW knob, then set BabyBear⁴ to pow=11 / OOD=1.
2. Or accept BabyBear⁴ compress/shrink at 99-bit and document it as the chosen
   security level for that field.
3. ext5 (128-bit) does **not** hit this gap — its larger field gives ample
   margin, so this only blocks BabyBear⁴ from a clean 100-bit claim.

BabyBear is currently the non-default field (KoalaBear⁴ is the production path),
so this gap does not affect the default build.

## External audit findings (WHIR PCS native verifier)

An external audit of the WHIR (Basefold-lineage) PCS native verifier raised the
following. Status as of this branch:

- **F-018 (HIGH) — query-count not pinned (FIXED).** The standard
  (non-pruned) verify paths zipped `iopp_queries` with `query_openings`
  without first asserting both equal `num_queries`; a truncated proof made the
  `.all(..)` check vacuously pass, bypassing IOPP query soundness. Fixed by
  asserting both lengths before the dispatch in the non-stacking
  (`whir_pcs.rs`) and stacked global-query (`whir_stacked.rs`) paths, plus the
  matching circuit assertions in `recursion/circuit/.../pcs.rs`.

- **F-019 / F-020 (HIGH / k=0 HIGH) — IOPP oracle count not pinned (FIXED).**
  The non-stacking verifier did not assert
  `iopp_oracles.len() == commit_schedule.len() + (k==0)`, so a prover could
  append tail oracles (moving the final-codeword early-stop check, bypassing
  the degree bound) or insert an oracle before the k=0 final commitment
  (biasing Fiat-Shamir). Fixed by the exact oracle-count assertion in
  `whir_pcs.rs` and a `commit_phase_openings.len() == iopp_commitments.len()`
  guard in `verify_iopp_query_whir`. The stacked / round paths already had the
  equivalent guards.

- **F-017 (DISPUTED) — pruned index value binding (FIXED defensively).**
  `Mmcs::verify_batch_pruned` authenticates rows at the proof's own embedded
  `sorted_indices` and the trait doc requires callers to cross-check those
  against transcript-sampled indices by value. Added
  `Mmcs::recover_pruned_indices` (merkle-tree + ExtensionMmcs) and assert the
  recovered list equals our recomputed sampled indices in every pruned path
  (IOPP rows and PCS input openings, stacked and non-stacking).

- **F-021 (LOW) — transcript binding order (mitigated in integration, not
  changed).** The PCS samples the batching `alpha` before observing
  `commitment_batch` / `opening_point` / `opened_values`. In the integrated
  zkDTVM system the outer STARK verifier observes the commitments and public
  values before invoking the PCS (see `dt-stark` `machine.rs` / `sumcheck`),
  so `alpha` is bound to the statement. Fixing it inside the library would
  change the Fiat-Shamir transcript and break proof compatibility, so it is
  left as-is and documented; standalone library use should observe all inputs
  before sampling `alpha`.

- **F-022 (LOW / DoS) — input validation (FIXED).** `validate_verify_inputs`
  now rejects heights that are not a nonzero power of two, and the verifier
  rejects a tallest committed height that does not equal `2^opening_point.len()`
  (previously a debug-only assert), preventing `log2_strict_usize` panics and
  shift underflow on malicious proofs.

### Follow-up: dead-code proof-shape checks (`check_shard_proof` family)

`recursion/circuit/.../sumcheck/mod.rs` carries a family of host-side
proof-shape comparison helpers (`check_vk_shape`, `check_shard_proof`,
`check_basefold`, `check_sumcheck_proof`) inherited from upstream. Three
self-compare bugs in them were fixed (they compared a value with itself, e.g.
`lhs.coeffs.len() != lhs.coeffs.len()`), so they now actually compare `lhs`
vs `rhs`.

They remain **unused (dead code)** and are not yet wired into the verify path.
Wiring them safely is deferred because:
- they are typed to the concrete `SCBabyBearPoseidon2` config, while the
  production field is KoalaBear; making them field-generic is blocked by
  `MlPCS::BatchProof` being an opaque associated type with no shape accessor;
- the dummy-proof generator (`sumcheck/dummy.rs`) is stacked-shaped, whereas
  production core/compress proofs are non-stacking, so it cannot serve as the
  reference shape without first being updated;
- proof-shape binding is already enforced in the prover by the
  `input.shape()`-keyed program cache plus the witness-stream length check, so
  these helpers would be a redundant secondary guard.

Prerequisite work before wiring: (1) make the helpers field-generic via an
`MlPCS` shape-accessor, (2) regenerate the dummy to match the active
(non-stacking) production shape.
