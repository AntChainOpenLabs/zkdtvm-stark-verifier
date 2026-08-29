//! Verify-latency benchmark for the native-recursion compressed (root_shrink) proof.
//!
//! First run proves fibonacci core + native compress once and caches the
//! `(DTReduceProof, DTVerifyingKey)` pair on disk; later runs load the cache and
//! only measure verification. Every iteration prints the wall time of
//! `verify_compressed` plus the per-phase breakdown collected through
//! `pcs::whir::profile` (labels suffixed `_us` are microseconds).
//!
//! Run (remote, per AGENTS.md):
//!   cargo test --release -p dt-prover --features native-recursion \
//!     --test bench_verify_native -- --nocapture
//!
//! Env knobs:
//!   BENCH_PROOF_CACHE  — cache file path (default: <tmp>/bench_verify_native_fibo.bin)
//!   BENCH_VERIFY_ITERS — verify iterations (default: 10)
//!   BENCH_FIBO_N       — fibonacci iteration count for the proved program (default: 480000)
#![cfg(feature = "native-recursion")]

use std::time::Instant;

use dt_core_executor::DTContext;
use dt_core_machine::{io::DTStdin, reduce::DTReduceProof};
use dt_prover::{components::SCCpuProverComponents, DTProver, DTVerifyingKey, RootSC};
use dt_stark::{
    sumcheck::{keys::SCStarkVerifyingKey, proof::SCShardProof},
    DTProverOpts,
};

fn cache_path() -> std::path::PathBuf {
    std::env::var("BENCH_PROOF_CACHE")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("bench_verify_native_fibo.bin"))
}

#[test]
fn bench_verify_native_compressed() {
    let iters: usize =
        std::env::var("BENCH_VERIFY_ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(10);

    let prover_init = Instant::now();
    let prover = DTProver::<SCCpuProverComponents>::new();
    println!("prover_init_ms={}", prover_init.elapsed().as_millis());

    // Cache the proof by components with their plain derive serde: the DTReduceProof
    // compact wire format does not round-trip the root proof today (its Deserialize
    // substitutes vk.chip_ordering — prep-only chips — for the shard's full
    // chip_ordering, so machine verify fails with ChipOpeningLengthMismatch).
    type CachedPair = ((SCStarkVerifyingKey<RootSC>, SCShardProof<RootSC>), DTVerifyingKey);
    let path = cache_path();
    let (reduce_proof, vk): (DTReduceProof<RootSC>, DTVerifyingKey) = if path.exists() {
        let bytes = std::fs::read(&path).expect("read proof cache");
        let deser_start = Instant::now();
        let ((proof_vk, shard_proof), vk): CachedPair =
            bincode::deserialize(&bytes).expect("deserialize proof cache");
        println!(
            "loaded cached proof: {} bytes, deserialize_ms={}",
            bytes.len(),
            deser_start.elapsed().as_millis()
        );
        let pair = (DTReduceProof { vk: proof_vk, proof: shard_proof }, vk);
        println!(
            "chip_ordering: proof_len={} vk_len={} content_equal={}",
            pair.0.proof.chip_ordering.len(),
            pair.0.vk.chip_ordering.len(),
            pair.0.proof.chip_ordering == pair.0.vk.chip_ordering
        );
        pair
    } else {
        let n: u32 =
            std::env::var("BENCH_FIBO_N").ok().and_then(|v| v.parse().ok()).unwrap_or(480_000);
        let elf = test_artifacts::FIBONACCI_ELF;
        let (pk, pk_d, program, vk) = prover.setup(elf);
        let _ = &pk;
        let stdin =
            DTStdin { buffer: vec![bincode::serialize(&n).unwrap()], ptr: 0, proofs: vec![] };

        let core_start = Instant::now();
        let core_proof = prover
            .prove_core(&pk_d, program, &stdin, DTProverOpts::default(), DTContext::default())
            .expect("core prove");
        println!("core_prove_s={:.1}", core_start.elapsed().as_secs_f64());

        let compress_start = Instant::now();
        let reduce_proof = prover
            .compress(&vk, core_proof, vec![], DTProverOpts::default())
            .expect("native compress");
        println!("compress_s={:.1}", compress_start.elapsed().as_secs_f64());

        let bytes = bincode::serialize(&((&reduce_proof.vk, &reduce_proof.proof), &vk))
            .expect("serialize proof cache");
        std::fs::write(&path, &bytes).expect("write proof cache");
        println!("cached proof to {} ({} bytes)", path.display(), bytes.len());
        (reduce_proof, vk)
    };

    // Optional byte-identity check against a previously cached proof
    // (BENCH_COMPARE_PROOF=<path to an older cache file>). Compares the proof
    // COMPONENTS and typed vk content — a whole-file cmp is invalid
    // because the vk/chip_ordering hash maps serialize in per-instance random
    // order even for identical content.
    if let Ok(other_path) = std::env::var("BENCH_COMPARE_PROOF") {
        let other_bytes = std::fs::read(&other_path).expect("read BENCH_COMPARE_PROOF");
        let ((other_vk, other_proof), _other_core_vk): CachedPair =
            bincode::deserialize(&other_bytes).expect("deserialize BENCH_COMPARE_PROOF");
        let components = |p: &SCShardProof<RootSC>| {
            bincode::serialize(&(
                &p.commitment,
                &p.opened_values,
                &p.opening_proof,
                &p.sumcheck_proof,
                &p.dimensions,
                &p.public_values,
            ))
            .expect("serialize proof components")
        };
        println!(
            "proof_components_identical={} proof_vk_typed_identical={} \
chip_ordering_identical={}",
            components(&reduce_proof.proof) == components(&other_proof),
            native_recursion::compress_dt::verifying_keys_equal(&reduce_proof.vk, &other_vk),
            reduce_proof.proof.chip_ordering == other_proof.chip_ordering,
        );
    }

    // Wire round trip: verify the proof exactly as an external consumer would
    // receive it — through the DTReduceProof compact wire format (v4 carries the
    // shard chip_ordering; v3 substituted the prep-only vk ordering and failed).
    let wire_bytes = bincode::serialize(&reduce_proof).expect("wire serialize");
    let reduce_proof: DTReduceProof<RootSC> =
        bincode::deserialize(&wire_bytes).expect("wire deserialize");
    println!("wire_roundtrip_bytes={}", wire_bytes.len());

    // Warm the backend (ladder disk-cache load / build) outside the timed loop:
    // this is once-per-process setup, not per-verify latency.
    let backend_init = Instant::now();
    prover.native_backend().expect("native backend init");
    println!("native_backend_init_ms={}", backend_init.elapsed().as_millis());

    let mut totals_ms = Vec::with_capacity(iters);
    for iter in 0..iters {
        pcs::whir::profile::reset();
        let start = Instant::now();
        prover.verify_compressed(&reduce_proof, &vk).expect("verify_compressed");
        let total_ms = start.elapsed().as_secs_f64() * 1e3;
        totals_ms.push(total_ms);
        let phases = pcs::whir::profile::take();
        let mut line = String::new();
        let mut accounted_us: u128 = 0;
        for (label, value) in &phases {
            if let Some(name) = label.strip_prefix("verify.") {
                line.push_str(&format!(" {name}={value}"));
                accounted_us += value;
            }
        }
        println!("iter={iter} verify_total_ms={total_ms:.2} accounted_us={accounted_us}{line}");
    }
    totals_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min = totals_ms.first().copied().unwrap_or(0.0);
    let median = totals_ms[totals_ms.len() / 2];
    let max = totals_ms.last().copied().unwrap_or(0.0);
    println!("verify_total_ms min={min:.2} median={median:.2} max={max:.2} iters={iters}");
}
