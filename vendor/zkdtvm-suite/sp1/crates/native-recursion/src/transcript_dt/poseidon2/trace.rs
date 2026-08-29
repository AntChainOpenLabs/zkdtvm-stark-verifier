use core::borrow::{Borrow, BorrowMut};
use std::{
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex, OnceLock,
    },
};

use dt_core_machine::operations::poseidon2_kb::{
    permutation::Poseidon2Degree3Cols, trace::populate_perm_deg3, WIDTH,
};
use dt_stark::{
    koalabear_poseidon2::koala_bear_poseidon2::{my_perm, Perm},
    sumcheck::trace::{CompressedMatrix, PaddingRow},
};
use hashbrown::HashMap;
use p3_field::{AbstractField, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;
use p3_symmetric::Permutation;

use crate::{
    config::F,
    system_dt::RecursionPoseidon2Pool,
    transcript_dt::poseidon2::columns::{
        NUM_POSEIDON2_PERMUTATION_COLS, NUM_POSEIDON2_PERMUTE_COLS, POSEIDON2_MULT_COL,
    },
    Instant,
};

static POSEIDON2_PERM: OnceLock<Perm> = OnceLock::new();

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
pub struct RecursionPoseidon2MemoSnapshot {
    pub hits: u64,
    pub misses: u64,
    pub proof_local_hits: u64,
    pub cross_proof_hits: u64,
    pub unique_inputs: u64,
    pub retained_bytes: u64,
    pub canonicalization_nanos: u64,
    pub map_lookup_lock_nanos: u64,
    pub cell_wait_nanos: u64,
    pub permutation_compute_nanos: u64,
    pub audit_enabled: bool,
}

type Poseidon2MemoKey = [u32; WIDTH];
type Poseidon2MemoOutput = [F; WIDTH];
const POSEIDON2_MEMO_SHARDS: usize = 64;
const POSEIDON2_MEMO_AUDIT_ENV: &str = "DT_NATIVE_POSEIDON2_MEMO_AUDIT";

struct Poseidon2MemoCell {
    owner_scope: u64,
    output: OnceLock<Poseidon2MemoOutput>,
}

type SharedPoseidon2MemoCell = Arc<Poseidon2MemoCell>;

struct Poseidon2MemoTable {
    shards: [Mutex<HashMap<Poseidon2MemoKey, SharedPoseidon2MemoCell>>; POSEIDON2_MEMO_SHARDS],
    next_scope: AtomicU64,
    unique_inputs: AtomicU64,
    retained_bucket_bytes: AtomicU64,
    audit_enabled: bool,
}

impl Default for Poseidon2MemoTable {
    fn default() -> Self {
        let audit_enabled = crate::env_var(POSEIDON2_MEMO_AUDIT_ENV)
            .is_ok_and(|value| value != "0" && !value.eq_ignore_ascii_case("false"));
        Self::new(audit_enabled)
    }
}

impl Poseidon2MemoTable {
    fn new(audit_enabled: bool) -> Self {
        Self {
            shards: core::array::from_fn(|_| Mutex::new(HashMap::new())),
            next_scope: AtomicU64::new(1),
            unique_inputs: AtomicU64::new(0),
            retained_bucket_bytes: AtomicU64::new(0),
            audit_enabled,
        }
    }

    fn next_scope(&self) -> u64 {
        let scope = self.next_scope.fetch_add(1, Ordering::Relaxed);
        assert_ne!(scope, u64::MAX, "Poseidon2 memo proof scope overflow");
        scope
    }

    fn shard(
        &self,
        key: &Poseidon2MemoKey,
    ) -> &Mutex<HashMap<Poseidon2MemoKey, SharedPoseidon2MemoCell>> {
        &self.shards[poseidon2_memo_shard(key)]
    }

    fn is_empty(&self) -> bool {
        self.shards
            .iter()
            .all(|shard| shard.lock().expect("recursion Poseidon2 memo shard lock").is_empty())
    }

    fn len(&self) -> usize {
        usize::try_from(self.unique_inputs.load(Ordering::Relaxed)).unwrap_or(usize::MAX)
    }

    fn retained_bytes(&self) -> u64 {
        let unique = self.unique_inputs.load(Ordering::Relaxed);
        if unique == 0 {
            return 0;
        }
        let table_bytes = u64::try_from(core::mem::size_of::<Self>()).unwrap_or(u64::MAX);
        let cell_bytes = u64::try_from(core::mem::size_of::<Poseidon2MemoCell>())
            .unwrap_or(u64::MAX)
            .saturating_mul(unique);
        table_bytes
            .saturating_add(cell_bytes)
            .saturating_add(self.retained_bucket_bytes.load(Ordering::Relaxed))
    }
}

type Poseidon2MemoEntries = Arc<Poseidon2MemoTable>;

/// Request-owned memo for host-side Poseidon2 output computation.
///
/// This cache never records provider requests or multiplicities. Cloning intentionally starts a
/// new empty request cache, while [`Self::append`] transfers entries and counters when two
/// request-owned records are merged.
pub struct RecursionPoseidon2Memo {
    entries: Poseidon2MemoEntries,
    request_lineage: Arc<()>,
    scope_id: u64,
    hits: AtomicU64,
    misses: AtomicU64,
    proof_local_hits: AtomicU64,
    cross_proof_hits: AtomicU64,
    canonicalization_nanos: AtomicU64,
    map_lookup_lock_nanos: AtomicU64,
    cell_wait_nanos: AtomicU64,
    permutation_compute_nanos: AtomicU64,
}

impl Default for RecursionPoseidon2Memo {
    fn default() -> Self {
        let entries = Arc::new(Poseidon2MemoTable::default());
        let scope_id = entries.next_scope();
        Self {
            entries,
            request_lineage: Arc::new(()),
            scope_id,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            proof_local_hits: AtomicU64::new(0),
            cross_proof_hits: AtomicU64::new(0),
            canonicalization_nanos: AtomicU64::new(0),
            map_lookup_lock_nanos: AtomicU64::new(0),
            cell_wait_nanos: AtomicU64::new(0),
            permutation_compute_nanos: AtomicU64::new(0),
        }
    }
}

/// Minimal output contract used by semantic trace builders. Mandatory proof
/// verification uses the output-only memo; CPU tracegen uses the full-witness
/// cache below so a Merkle/statement walk and the Poseidon2 provider never
/// evaluate the same permutation twice.
pub trait RecursionPoseidon2Output {
    fn permute_output(&self, input: [F; WIDTH]) -> [F; WIDTH];
}

impl RecursionPoseidon2Output for RecursionPoseidon2Memo {
    fn permute_output(&self, input: [F; WIDTH]) -> [F; WIDTH] {
        self.permute(input)
    }
}

/// Tracegen-only cache of complete Poseidon2 permutation columns.
///
/// It is empty throughout preparation, fills lazily while derived rows expose
/// chained permutation inputs, and is consumed by the provider trace. The
/// multiplicity column is deliberately excluded because direct mode may emit
/// the same input in more than one immutable segment with different counts.
#[derive(Default)]
pub struct RecursionPoseidon2TracegenCache {
    rows: Mutex<HashMap<Poseidon2MemoKey, Arc<[F]>>>,
    generated_rows: AtomicU64,
    generation_nanos: AtomicU64,
}

impl core::fmt::Debug for RecursionPoseidon2TracegenCache {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("RecursionPoseidon2TracegenCache")
            .field("rows", &self.rows.lock().expect("Poseidon2 tracegen cache lock").len())
            .field("generated_rows", &self.generated_rows())
            .finish()
    }
}

impl Clone for RecursionPoseidon2TracegenCache {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl PartialEq for RecursionPoseidon2TracegenCache {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl Eq for RecursionPoseidon2TracegenCache {}

impl RecursionPoseidon2TracegenCache {
    fn witness_row(&self, input: [F; WIDTH]) -> Arc<[F]> {
        let key = canonical_poseidon2_input(input);
        if let Some(row) =
            self.rows.lock().expect("Poseidon2 tracegen cache lock").get(&key).cloned()
        {
            return row;
        }

        let start = Instant::now();
        let row: Arc<[F]> = build_permutation_row(input).into();
        let elapsed = start.elapsed().as_nanos();
        let mut rows = self.rows.lock().expect("Poseidon2 tracegen cache lock");
        if let Some(existing) = rows.get(&key) {
            return Arc::clone(existing);
        }
        rows.insert(key, Arc::clone(&row));
        self.generated_rows.fetch_add(1, Ordering::Relaxed);
        self.generation_nanos
            .fetch_add(u64::try_from(elapsed).unwrap_or(u64::MAX), Ordering::Relaxed);
        row
    }

    fn witness_rows_batch<I>(&self, inputs: I) -> Vec<Arc<[F]>>
    where
        I: IntoIterator<Item = [F; WIDTH]>,
        I::IntoIter: ExactSizeIterator,
    {
        let inputs = inputs.into_iter();
        let mut output = Vec::with_capacity(inputs.len());
        let started = Instant::now();
        let mut rows = self.rows.lock().expect("Poseidon2 tracegen cache lock");
        let prior_len = rows.len();
        rows.reserve(inputs.len());
        for input in inputs {
            let key = canonical_poseidon2_input(input);
            if let Some(row) = rows.get(&key) {
                output.push(Arc::clone(row));
                continue;
            }
            let row: Arc<[F]> = build_permutation_row(input).into();
            rows.insert(key, Arc::clone(&row));
            output.push(row);
        }
        let generated = rows.len() - prior_len;
        drop(rows);
        self.generated_rows.fetch_add(
            u64::try_from(generated).expect("Poseidon2 generated row count exceeds u64"),
            Ordering::Relaxed,
        );
        self.generation_nanos.fetch_add(
            u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        output
    }

    pub fn generated_rows(&self) -> u64 {
        self.generated_rows.load(Ordering::Relaxed)
    }

    pub fn generation_nanos(&self) -> u64 {
        self.generation_nanos.load(Ordering::Relaxed)
    }

    pub fn retained_rows(&self) -> usize {
        self.rows.lock().expect("Poseidon2 tracegen cache lock").len()
    }

    pub fn clear(&self) {
        self.rows.lock().expect("Poseidon2 tracegen cache lock").clear();
    }
}

impl RecursionPoseidon2Output for RecursionPoseidon2TracegenCache {
    fn permute_output(&self, input: [F; WIDTH]) -> [F; WIDTH] {
        output_from_trace_row(&self.witness_row(input))
    }
}

impl core::fmt::Debug for RecursionPoseidon2Memo {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let snapshot = self.snapshot();
        formatter
            .debug_struct("RecursionPoseidon2Memo")
            .field("entries", &self.entries.len())
            .field("hits", &snapshot.hits)
            .field("misses", &snapshot.misses)
            .finish()
    }
}

impl Clone for RecursionPoseidon2Memo {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl PartialEq for RecursionPoseidon2Memo {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl Eq for RecursionPoseidon2Memo {}

impl RecursionPoseidon2Memo {
    /// Starts an independently-profiled child view over this request's shared memo entries.
    /// Ordinary [`Clone`] intentionally does not share entries.
    pub(crate) fn fork(&self) -> Self {
        let scope_id = self.entries.next_scope();
        Self {
            entries: Arc::clone(&self.entries),
            request_lineage: Arc::clone(&self.request_lineage),
            scope_id,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            proof_local_hits: AtomicU64::new(0),
            cross_proof_hits: AtomicU64::new(0),
            canonicalization_nanos: AtomicU64::new(0),
            map_lookup_lock_nanos: AtomicU64::new(0),
            cell_wait_nanos: AtomicU64::new(0),
            permutation_compute_nanos: AtomicU64::new(0),
        }
    }

    /// Starts a proof-local memo table while retaining the request-lineage token used by node
    /// admission. This is the no-shared-memo performance candidate; independent child tables are
    /// discarded rather than unioned when their records merge.
    pub(crate) fn fork_isolated(&self) -> Self {
        let entries = Arc::new(Poseidon2MemoTable::new(self.entries.audit_enabled));
        let scope_id = entries.next_scope();
        Self {
            entries,
            request_lineage: Arc::clone(&self.request_lineage),
            scope_id,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            proof_local_hits: AtomicU64::new(0),
            cross_proof_hits: AtomicU64::new(0),
            canonicalization_nanos: AtomicU64::new(0),
            map_lookup_lock_nanos: AtomicU64::new(0),
            cell_wait_nanos: AtomicU64::new(0),
            permutation_compute_nanos: AtomicU64::new(0),
        }
    }

    #[cfg(test)]
    pub(crate) fn shares_entries_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.entries, &other.entries)
    }

    pub(crate) fn shares_request_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.request_lineage, &other.request_lineage)
    }

    pub fn permute(&self, input: [F; WIDTH]) -> [F; WIDTH] {
        let canonical_started = self.entries.audit_enabled.then(Instant::now);
        let key = canonical_poseidon2_input(input);
        if let Some(started) = canonical_started {
            add_nanos(&self.canonicalization_nanos, started.elapsed().as_nanos());
        }
        let lookup_started = self.entries.audit_enabled.then(Instant::now);
        let (cell, inserted) = {
            let mut entries =
                self.entries.shard(&key).lock().expect("recursion Poseidon2 memo shard lock");
            if let Some(cell) = entries.get(&key) {
                (Arc::clone(cell), false)
            } else {
                let prior_capacity = entries.capacity();
                let cell = Arc::new(Poseidon2MemoCell {
                    owner_scope: self.scope_id,
                    output: OnceLock::new(),
                });
                entries.insert(key, Arc::clone(&cell));
                let new_capacity = entries.capacity();
                if new_capacity > prior_capacity {
                    let bucket_bytes = new_capacity.saturating_sub(prior_capacity).saturating_mul(
                        core::mem::size_of::<(Poseidon2MemoKey, SharedPoseidon2MemoCell)>() + 1,
                    );
                    add_u64(
                        &self.entries.retained_bucket_bytes,
                        u64::try_from(bucket_bytes).unwrap_or(u64::MAX),
                        "Poseidon2 memo retained-byte counter overflow",
                    );
                }
                add_u64(
                    &self.entries.unique_inputs,
                    1,
                    "Poseidon2 memo unique-input counter overflow",
                );
                (cell, true)
            }
        };
        if let Some(started) = lookup_started {
            add_nanos(&self.map_lookup_lock_nanos, started.elapsed().as_nanos());
        }

        // Distinct inputs run without the map lock. Same-input workers share this cell, so the
        // permutation is evaluated exactly once and all other workers wait for that output.
        let cell_started = self.entries.audit_enabled.then(Instant::now);
        let mut computed_here = false;
        let output = *cell.output.get_or_init(|| {
            computed_here = true;
            let compute_started = self.entries.audit_enabled.then(Instant::now);
            let output = poseidon2_permute(input);
            if let Some(started) = compute_started {
                add_nanos(&self.permutation_compute_nanos, started.elapsed().as_nanos());
            }
            output
        });
        if !computed_here {
            if let Some(started) = cell_started {
                add_nanos(&self.cell_wait_nanos, started.elapsed().as_nanos());
            }
        }
        if inserted {
            add_u64(&self.misses, 1, "Poseidon2 memo misses overflow");
        } else {
            add_u64(&self.hits, 1, "Poseidon2 memo hits overflow");
            if cell.owner_scope == self.scope_id {
                add_u64(&self.proof_local_hits, 1, "Poseidon2 proof-local hit counter overflow");
            } else {
                add_u64(&self.cross_proof_hits, 1, "Poseidon2 cross-proof hit counter overflow");
            }
        }
        output
    }

    pub fn snapshot(&self) -> RecursionPoseidon2MemoSnapshot {
        RecursionPoseidon2MemoSnapshot {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            proof_local_hits: self.proof_local_hits.load(Ordering::Relaxed),
            cross_proof_hits: self.cross_proof_hits.load(Ordering::Relaxed),
            unique_inputs: self.entries.unique_inputs.load(Ordering::Relaxed),
            retained_bytes: self.entries.retained_bytes(),
            canonicalization_nanos: self.canonicalization_nanos.load(Ordering::Relaxed),
            map_lookup_lock_nanos: self.map_lookup_lock_nanos.load(Ordering::Relaxed),
            cell_wait_nanos: self.cell_wait_nanos.load(Ordering::Relaxed),
            permutation_compute_nanos: self.permutation_compute_nanos.load(Ordering::Relaxed),
            audit_enabled: self.entries.audit_enabled,
        }
    }

    /// Merges independently-profiled views from one request and consumes the other counters.
    /// Two populated, independent stores are a request-lineage violation, not a compatibility
    /// case: the canonical pipeline never performs an O(cache-size) cross-request union.
    pub(crate) fn append(&mut self, other: &mut Self) {
        let same_request = Arc::ptr_eq(&self.entries, &other.entries);
        let same_lineage = Arc::ptr_eq(&self.request_lineage, &other.request_lineage);
        let (self_is_empty, other_is_empty) = if same_request {
            (false, false)
        } else {
            (self.entries.is_empty(), other.entries.is_empty())
        };
        assert!(
            same_lineage || self_is_empty || other_is_empty,
            "cannot merge populated Poseidon2 memos from different recursion requests"
        );

        for (target, source, overflow) in [
            (&self.hits, &other.hits, "Poseidon2 memo hits overflow"),
            (&self.misses, &other.misses, "Poseidon2 memo misses overflow"),
            (
                &self.proof_local_hits,
                &other.proof_local_hits,
                "Poseidon2 proof-local hit counter overflow",
            ),
            (
                &self.cross_proof_hits,
                &other.cross_proof_hits,
                "Poseidon2 cross-proof hit counter overflow",
            ),
            (
                &self.canonicalization_nanos,
                &other.canonicalization_nanos,
                "Poseidon2 canonicalization timer overflow",
            ),
            (
                &self.map_lookup_lock_nanos,
                &other.map_lookup_lock_nanos,
                "Poseidon2 lookup/lock timer overflow",
            ),
            (&self.cell_wait_nanos, &other.cell_wait_nanos, "Poseidon2 cell-wait timer overflow"),
            (
                &self.permutation_compute_nanos,
                &other.permutation_compute_nanos,
                "Poseidon2 compute timer overflow",
            ),
        ] {
            add_u64(target, source.swap(0, Ordering::Relaxed), overflow);
        }

        if same_request {
            return;
        }

        if same_lineage {
            // Proof-local tables are deliberately not unioned. After a node merge, subsequent
            // statement work starts from one empty node-local table, preserving correctness while
            // making cross-proof child-preflight reuse impossible in the candidate mode.
            self.entries = Arc::new(Poseidon2MemoTable::new(self.entries.audit_enabled));
            self.scope_id = self.entries.next_scope();
            return;
        }

        if self_is_empty {
            self.entries = Arc::clone(&other.entries);
            self.request_lineage = Arc::clone(&other.request_lineage);
            self.scope_id = self.entries.next_scope();
        }
    }
}

fn add_nanos(counter: &AtomicU64, nanos: u128) {
    add_u64(
        counter,
        u64::try_from(nanos).unwrap_or(u64::MAX),
        "Poseidon2 memo timing counter overflow",
    );
}

fn add_u64(counter: &AtomicU64, value: u64, overflow: &'static str) {
    let prior = counter.fetch_add(value, Ordering::Relaxed);
    assert!(prior.checked_add(value).is_some(), "{overflow}");
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Poseidon2PermuteTraceGenerator;

impl Poseidon2PermuteTraceGenerator {
    pub fn trace_height(pool: &RecursionPoseidon2Pool) -> usize {
        pool.unique_count().max(1).next_power_of_two()
    }

    pub fn generate_trace_row_major(
        pool: &RecursionPoseidon2Pool,
        cache: &RecursionPoseidon2TracegenCache,
    ) -> RowMajorMatrix<F> {
        let height = Self::trace_height(pool);
        let active_rows = pool.unique_count();
        let mut requests: Vec<_> = pool.requests().collect();
        requests.sort_unstable_by_key(|request| canonical_poseidon2_input(request.input));
        let mut trace = vec![F::zero(); height * NUM_POSEIDON2_PERMUTE_COLS];
        let witnesses = cache.witness_rows_batch(requests.iter().map(|request| request.input));

        for ((request, witness), row) in requests.iter().zip(witnesses).zip(
            trace[..active_rows * NUM_POSEIDON2_PERMUTE_COLS]
                .chunks_exact_mut(NUM_POSEIDON2_PERMUTE_COLS),
        ) {
            fill_trace_row_from_witness(row, &witness, request.count);
        }

        if active_rows < height {
            let padding = &mut trace[active_rows * NUM_POSEIDON2_PERMUTE_COLS..];
            let (first, remaining) = padding.split_at_mut(NUM_POSEIDON2_PERMUTE_COLS);
            fill_trace_row(first, [F::zero(); WIDTH], 0, cache);
            for row in remaining.chunks_exact_mut(NUM_POSEIDON2_PERMUTE_COLS) {
                row.copy_from_slice(first);
            }
        }

        RowMajorMatrix::new(trace, NUM_POSEIDON2_PERMUTE_COLS)
    }

    pub fn generate_trace_compressed(
        pool: &RecursionPoseidon2Pool,
        cache: &RecursionPoseidon2TracegenCache,
    ) -> CompressedMatrix<F> {
        // Instrumentation: print the pool's request/unique split.
        if crate::debug_prints_enabled() {
            println!(
                "native_pose2_pool unique={} total_requests={}",
                pool.unique_count(),
                pool.total_count_usize(),
            );
        }
        if pool.unique_count() == 0 {
            let main =
                RowMajorMatrix::new(zero_input_padding_row(cache), NUM_POSEIDON2_PERMUTE_COLS);
            return CompressedMatrix::new(main, PaddingRow::None, 1);
        }

        let height = Self::trace_height(pool);
        let mut requests: Vec<_> = pool.requests().collect();
        requests.sort_unstable_by_key(|request| canonical_poseidon2_input(request.input));
        let mut trace = vec![F::zero(); requests.len() * NUM_POSEIDON2_PERMUTE_COLS];
        let witnesses = cache.witness_rows_batch(requests.iter().map(|request| request.input));
        for ((request, witness), row) in
            requests.iter().zip(witnesses).zip(trace.chunks_exact_mut(NUM_POSEIDON2_PERMUTE_COLS))
        {
            fill_trace_row_from_witness(row, &witness, request.count);
        }

        let main = RowMajorMatrix::new(trace, NUM_POSEIDON2_PERMUTE_COLS);
        let padding = if pool.unique_count() < height {
            PaddingRow::General(zero_input_padding_row(cache))
        } else {
            PaddingRow::None
        };
        let trace = CompressedMatrix::new(main, padding, height);
        cache.clear();
        trace
    }
}

pub fn poseidon2_permute(input: [F; WIDTH]) -> [F; WIDTH] {
    POSEIDON2_PERM.get_or_init(my_perm).permute(input)
}

pub(super) fn canonical_poseidon2_input(input: [F; WIDTH]) -> [u32; WIDTH] {
    input.map(|value| value.as_canonical_u32())
}

pub(super) fn poseidon2_memo_shard(key: &Poseidon2MemoKey) -> usize {
    let mut mixed =
        key[0] ^ key[5].rotate_left(7) ^ key[10].rotate_left(13) ^ key[15].rotate_left(19);
    mixed ^= mixed >> 16;
    mixed = mixed.wrapping_mul(0x7feb_352d);
    mixed ^= mixed >> 15;
    (mixed as usize) & (POSEIDON2_MEMO_SHARDS - 1)
}

fn zero_input_padding_row(cache: &RecursionPoseidon2TracegenCache) -> Vec<F> {
    let mut row = vec![F::zero(); NUM_POSEIDON2_PERMUTE_COLS];
    fill_trace_row(&mut row, [F::zero(); WIDTH], 0, cache);
    row
}

fn build_permutation_row(input: [F; WIDTH]) -> Vec<F> {
    let mut row = vec![F::zero(); NUM_POSEIDON2_PERMUTATION_COLS];
    let op = populate_perm_deg3(input, None);
    let permutation: &mut Poseidon2Degree3Cols<F> = row.as_mut_slice().borrow_mut();
    *permutation = op.permutation;
    row
}

fn fill_trace_row(
    row: &mut [F],
    input: [F; WIDTH],
    count: u32,
    cache: &RecursionPoseidon2TracegenCache,
) {
    let permutation = cache.witness_row(input);
    fill_trace_row_from_witness(row, &permutation, count);
}

fn fill_trace_row_from_witness(row: &mut [F], permutation: &[F], count: u32) {
    debug_assert_eq!(row.len(), NUM_POSEIDON2_PERMUTE_COLS);
    debug_assert_eq!(permutation.len(), NUM_POSEIDON2_PERMUTATION_COLS);
    row[..NUM_POSEIDON2_PERMUTATION_COLS].copy_from_slice(permutation);
    row[POSEIDON2_MULT_COL] = F::from_canonical_u32(count);
}

pub fn output_from_trace_row(row: &[F]) -> [F; WIDTH] {
    let permutation: &Poseidon2Degree3Cols<F> = row[..NUM_POSEIDON2_PERMUTATION_COLS].borrow();
    permutation.state.output_state
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system_dt::RecursionPoseidon2Pool;
    use dt_stark::koalabear_poseidon2::koala_bear_poseidon2::my_perm;
    use p3_matrix::Matrix;
    use p3_symmetric::Permutation;

    fn memo_input(offset: u32) -> [F; WIDTH] {
        core::array::from_fn(|i| F::from_canonical_u32(offset + i as u32))
    }

    fn assert_memo_counts(memo: &RecursionPoseidon2Memo, hits: u64, misses: u64) {
        let snapshot = memo.snapshot();
        assert_eq!(snapshot.hits, hits);
        assert_eq!(snapshot.misses, misses);
        assert_eq!(snapshot.hits, snapshot.proof_local_hits + snapshot.cross_proof_hits);
    }

    #[test]
    fn preflight_and_trace_match_active_perm() {
        let input = core::array::from_fn(|i| F::from_canonical_u32((i as u32) + 1));
        let expected = my_perm().permute(input);

        let mut record = crate::system_dt::RecursionRecord::default();
        record.poseidon2.record_poseidon2(input);
        record.poseidon2.record_poseidon2(input);
        assert_eq!(record.poseidon2.unique_count(), 2);
        record.reduce_provider_inputs().expect("reduce Poseidon2 provider input");
        let pool = &record.poseidon2;
        assert_eq!(pool.unique_count(), 1);
        assert_eq!(pool.total_count(), 2);

        let cache = RecursionPoseidon2TracegenCache::default();
        let trace = Poseidon2PermuteTraceGenerator::generate_trace_row_major(pool, &cache);
        assert_eq!(trace.width(), NUM_POSEIDON2_PERMUTE_COLS);
        assert_eq!(trace.height(), 1);
        let row = trace.row_slice(0);
        assert_eq!(output_from_trace_row(row.as_ref()), expected);
        assert_eq!(row[POSEIDON2_MULT_COL], F::from_canonical_u32(2));
    }

    #[test]
    fn trace_height_crosses_exact_power_of_two_boundaries() {
        let mut pool = RecursionPoseidon2Pool::default();
        assert_eq!(Poseidon2PermuteTraceGenerator::trace_height(&pool), 1);

        pool.record_poseidon2(memo_input(1));
        assert_eq!(Poseidon2PermuteTraceGenerator::trace_height(&pool), 1);
        pool.record_poseidon2(memo_input(101));
        assert_eq!(Poseidon2PermuteTraceGenerator::trace_height(&pool), 2);
        pool.record_poseidon2(memo_input(201));
        assert_eq!(Poseidon2PermuteTraceGenerator::trace_height(&pool), 4);
        pool.record_poseidon2(memo_input(301));
        assert_eq!(Poseidon2PermuteTraceGenerator::trace_height(&pool), 4);
        pool.record_poseidon2(memo_input(401));
        assert_eq!(Poseidon2PermuteTraceGenerator::trace_height(&pool), 8);
    }

    #[test]
    fn tracegen_cache_reuses_derived_row_witness_and_releases_it_after_provider_trace() {
        let input = memo_input(500);
        let cache = RecursionPoseidon2TracegenCache::default();
        assert_eq!(cache.permute_output(input), poseidon2_permute(input));
        assert_eq!(cache.generated_rows(), 1);
        assert_eq!(cache.retained_rows(), 1);

        let mut pool = RecursionPoseidon2Pool::default();
        pool.record_poseidon2(input);
        let trace = Poseidon2PermuteTraceGenerator::generate_trace_compressed(&pool, &cache);
        assert_eq!(trace.height(), 1);
        assert_eq!(cache.generated_rows(), 1);
        assert_eq!(cache.retained_rows(), 0);
    }

    #[test]
    fn tracegen_batch_witness_preserves_order_and_distincts_canonical_inputs() {
        let first = memo_input(600);
        let second = memo_input(700);
        let cache = RecursionPoseidon2TracegenCache::default();
        let rows = cache.witness_rows_batch([first, second, first]);

        assert_eq!(rows.len(), 3);
        assert_eq!(output_from_trace_row(&rows[0]), poseidon2_permute(first));
        assert_eq!(output_from_trace_row(&rows[1]), poseidon2_permute(second));
        assert_eq!(rows[0], rows[2]);
        assert_eq!(cache.generated_rows(), 2);
        assert_eq!(cache.retained_rows(), 2);
    }

    #[test]
    fn request_memo_reports_hits_and_misses() {
        let memo = RecursionPoseidon2Memo::default();
        let input = memo_input(1);
        let expected = poseidon2_permute(input);

        assert_memo_counts(&memo, 0, 0);
        assert_eq!(memo.permute(input), expected);
        assert_memo_counts(&memo, 0, 1);
        assert_eq!(memo.snapshot().unique_inputs, 1);
        assert!(memo.snapshot().retained_bytes > 0);
        assert_eq!(memo.permute(input), expected);
        assert_memo_counts(&memo, 1, 1);
        assert_eq!(memo.snapshot().proof_local_hits, 1);
    }

    #[test]
    fn same_key_is_single_flight_across_forked_workers() {
        let memo = RecursionPoseidon2Memo::default();
        let children = (0..8).map(|_| memo.fork()).collect::<Vec<_>>();
        let input = memo_input(10);
        std::thread::scope(|scope| {
            let handles = children
                .iter()
                .map(|child| scope.spawn(move || child.permute(input)))
                .collect::<Vec<_>>();
            for handle in handles {
                assert_eq!(handle.join().expect("memo worker"), poseidon2_permute(input));
            }
        });
        let aggregate = children.iter().map(RecursionPoseidon2Memo::snapshot).fold(
            RecursionPoseidon2MemoSnapshot::default(),
            |mut total, snapshot| {
                total.hits += snapshot.hits;
                total.misses += snapshot.misses;
                total
            },
        );
        assert_eq!((aggregate.hits, aggregate.misses), (7, 1));
        assert_memo_counts(&memo, 0, 0);
        assert_eq!(children.iter().map(|child| child.snapshot().cross_proof_hits).sum::<u64>(), 7);
    }

    #[test]
    fn distinct_keys_run_through_one_shared_request_store() {
        let seed = RecursionPoseidon2Memo::default();
        let children = (0..8).map(|_| seed.fork()).collect::<Vec<_>>();
        std::thread::scope(|scope| {
            let handles = children
                .iter()
                .enumerate()
                .map(|(idx, child)| {
                    scope.spawn(move || {
                        let input = memo_input(200 + idx as u32 * 20);
                        (input, child.permute(input))
                    })
                })
                .collect::<Vec<_>>();
            for handle in handles {
                let (input, output) = handle.join().expect("memo worker");
                assert_eq!(output, poseidon2_permute(input));
            }
        });
        assert_eq!(
            children.iter().map(|child| child.snapshot().misses).sum::<u64>(),
            children.len() as u64
        );
    }

    #[test]
    fn request_memos_are_isolated() {
        let left = RecursionPoseidon2Memo::default();
        let right = RecursionPoseidon2Memo::default();
        let input = memo_input(20);

        left.permute(input);
        assert_memo_counts(&left, 0, 1);
        assert_memo_counts(&right, 0, 0);
        right.permute(input);
        assert_memo_counts(&right, 0, 1);
    }

    #[test]
    fn cloning_starts_an_empty_request_memo() {
        let original = RecursionPoseidon2Memo::default();
        let input = memo_input(40);
        original.permute(input);

        let cloned = original.clone();
        assert_eq!(original, cloned);
        assert!(!original.shares_entries_with(&cloned));
        assert_memo_counts(&cloned, 0, 0);
        cloned.permute(input);
        assert_memo_counts(&cloned, 0, 1);
    }

    #[test]
    #[should_panic(expected = "different recursion requests")]
    fn append_rejects_populated_independent_requests() {
        let mut left = RecursionPoseidon2Memo::default();
        let mut right = RecursionPoseidon2Memo::default();
        left.permute(memo_input(60));
        right.permute(memo_input(80));
        left.append(&mut right);
    }

    #[test]
    fn append_adopts_a_populated_store_into_an_empty_accumulator() {
        let mut left = RecursionPoseidon2Memo::default();
        let mut right = RecursionPoseidon2Memo::default();
        let input = memo_input(80);

        right.permute(input);
        left.append(&mut right);

        assert!(left.shares_entries_with(&right));
        assert_memo_counts(&left, 0, 1);
        assert_memo_counts(&right, 0, 0);
        left.permute(input);
        assert_memo_counts(&left, 1, 1);
    }

    #[test]
    fn append_of_forked_views_keeps_one_entry_store() {
        let seed = RecursionPoseidon2Memo::default();
        let mut left = seed.fork();
        let mut right = seed.fork();
        let input = memo_input(100);
        left.permute(input);
        right.permute(input);
        assert!(left.shares_entries_with(&right));
        left.append(&mut right);
        assert_memo_counts(&left, 1, 1);
        assert_memo_counts(&right, 0, 0);
    }

    #[test]
    fn isolated_proof_views_share_lineage_but_not_entries_or_cross_proof_hits() {
        let request = RecursionPoseidon2Memo::default();
        let mut left = request.fork_isolated();
        let mut right = request.fork_isolated();
        let input = memo_input(120);
        left.permute(input);
        right.permute(input);
        assert!(left.shares_request_with(&right));
        assert!(!left.shares_entries_with(&right));
        assert_eq!(left.snapshot().cross_proof_hits, 0);
        assert_eq!(right.snapshot().cross_proof_hits, 0);

        left.append(&mut right);
        assert_memo_counts(&left, 0, 2);
        assert_eq!(left.snapshot().unique_inputs, 0, "isolated proof tables are not unioned");
        assert_eq!(left.snapshot().retained_bytes, 0);
        assert_memo_counts(&right, 0, 0);
    }
}
