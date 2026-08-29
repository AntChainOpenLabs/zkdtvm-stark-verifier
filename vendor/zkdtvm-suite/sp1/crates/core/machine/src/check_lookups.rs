//! Lightweight lookup (interaction) consistency checker.
//!
//! For a valid STARK proof, all **Local scope** send/receive interactions within a single
//! shard must balance: for every unique `(kind, values)` tuple, the total send multiplicity
//! must equal the total receive multiplicity.
//!
//! This module evaluates each chip's `Interaction` definitions on concrete trace values
//! and verifies this balance without requiring the full `StarkMachine` or `ProvingKey`.

use hashbrown::HashMap;
use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, Field, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use dt_core_executor::ExecutionRecord;
use dt_stark::{
    air::{InteractionScope, MachineAir},
    Interaction, InteractionKind,
};
use p3_air::BaseAir;

use crate::riscv::RiscvAir;

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

/// A source that contributed to a lookup entry (for diagnostics).
#[derive(Debug, Clone)]
pub struct LookupSource {
    pub chip_name: String,
    pub row_index: usize,
    pub is_send: bool,
    pub mult: u32,
}

/// An accumulated lookup entry keyed by `(kind, values)`.
#[derive(Debug, Clone)]
pub struct LookupEntry {
    pub kind: InteractionKind,
    pub values: Vec<u32>,
    /// Net multiplicity: positive means more sends, negative means more receives.
    pub net_mult: i64,
    /// Sources (kept for diagnostics; capped to avoid memory explosion).
    pub sources: Vec<LookupSource>,
}

/// The result of a lookup consistency check.
#[derive(Debug)]
pub struct LookupCheckResult {
    /// Total number of unique (kind, values) entries seen.
    pub total_entries: usize,
    /// Number of entries where net_mult == 0 (balanced).
    pub balanced: usize,
    /// Entries where net_mult != 0 (mismatched).
    pub mismatches: Vec<LookupEntry>,
}

// ---------------------------------------------------------------------------
// Core implementation
// ---------------------------------------------------------------------------

type F = BabyBear;

/// Key type for the lookup map: (kind as u8, values as Vec<u32>).
type LookupKey = (u8, Vec<u32>);

/// Maximum number of sources to keep per entry (to bound memory).
const MAX_SOURCES_PER_ENTRY: usize = 10;

/// Per-chip, per-kind send/receive summary.
#[derive(Debug, Default, Clone)]
pub struct ChipKindSummary {
    pub chip_name: String,
    pub kind: String,
    /// Total send multiplicity.
    pub total_send: i64,
    /// Total receive multiplicity.
    pub total_recv: i64,
    /// Number of distinct (kind, values) entries sent.
    pub send_entries: usize,
    /// Number of distinct (kind, values) entries received.
    pub recv_entries: usize,
}

/// Collect per-chip, per-kind send/receive totals across ALL entries (balanced and mismatched).
pub fn collect_chip_kind_summaries(
    record: &ExecutionRecord,
    skip_chips: &[&str],
) -> Vec<ChipKindSummary> {
    let (chips, _) = RiscvAir::<F>::get_chips_and_costs();
    let mut result = Vec::new();

    for chip in &chips {
        let name = chip.name();
        if skip_chips.contains(&name.as_str()) || !chip.included(record) {
            continue;
        }

        let trace_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut output = ExecutionRecord::default();
            let main_trace = chip.generate_trace(record, &mut output).decompress();
            let prep_trace = if chip.preprocessed_width() > 0 {
                chip.generate_preprocessed_trace(&record.program).map(|c| c.decompress())
            } else {
                None
            };
            (main_trace, prep_trace)
        }));

        let (main_trace, prep_trace) = match trace_result {
            Ok(t) => t,
            Err(_) => continue,
        };

        let sends = chip.sends();
        let receives = chip.receives();
        let height = main_trace.height();
        let empty_prep = vec![F::zero(); 1];

        // Track per-kind totals for this chip
        let mut kind_send: HashMap<u8, (i64, usize)> = HashMap::new();
        let mut kind_recv: HashMap<u8, (i64, usize)> = HashMap::new();

        for row in 0..height {
            let main_row = main_trace.row_slice(row);
            let prep_row: Vec<F> = if let Some(pt) = prep_trace.as_ref() {
                pt.row_slice(row).to_vec()
            } else {
                empty_prep.clone()
            };

            for interaction in sends.iter().filter(|i| i.scope == InteractionScope::Local) {
                let mult: F = interaction.multiplicity.apply::<F, F>(&prep_row, &main_row);
                if !mult.is_zero() {
                    let e = kind_send.entry(interaction.kind as u8).or_insert((0, 0));
                    e.0 += mult.as_canonical_u32() as i64;
                    e.1 += 1;
                }
            }

            for interaction in receives.iter().filter(|i| i.scope == InteractionScope::Local) {
                let mult: F = interaction.multiplicity.apply::<F, F>(&prep_row, &main_row);
                if !mult.is_zero() {
                    let e = kind_recv.entry(interaction.kind as u8).or_insert((0, 0));
                    e.0 += mult.as_canonical_u32() as i64;
                    e.1 += 1;
                }
            }
        }

        // Merge send and recv into summaries
        let mut all_kinds: std::collections::BTreeSet<u8> = std::collections::BTreeSet::new();
        all_kinds.extend(kind_send.keys());
        all_kinds.extend(kind_recv.keys());

        for kind_u8 in all_kinds {
            let kind_str = match kind_u8 {
                k if k == InteractionKind::Memory as u8 => "Memory",
                k if k == InteractionKind::Program as u8 => "Program",
                k if k == InteractionKind::Instruction as u8 => "Instruction",
                k if k == InteractionKind::Alu as u8 => "Alu",
                k if k == InteractionKind::Byte as u8 => "Byte",
                k if k == InteractionKind::Range as u8 => "Range",
                k if k == InteractionKind::Field as u8 => "Field",
                k if k == InteractionKind::Global as u8 => "Global",
                k if k == InteractionKind::Syscall as u8 => "Syscall",
                k if k == InteractionKind::ShaExtend as u8 => "ShaExtend",
                k if k == InteractionKind::ShaCompress as u8 => "ShaCompress",
                k if k == InteractionKind::Keccak as u8 => "Keccak",
                k if k == InteractionKind::State as u8 => "State",
                k if k == InteractionKind::MemoryGlobalAddr as u8 => "MemoryGlobalAddr",
                k if k == InteractionKind::BitVec as u8 => "BitVec",
                k if k == InteractionKind::Recursion as u8 => "Recursion",
                _ => "Unknown",
            };
            let (s_tot, s_cnt) = kind_send.get(&kind_u8).copied().unwrap_or((0, 0));
            let (r_tot, r_cnt) = kind_recv.get(&kind_u8).copied().unwrap_or((0, 0));
            result.push(ChipKindSummary {
                chip_name: name.clone(),
                kind: kind_str.to_string(),
                total_send: s_tot,
                total_recv: r_tot,
                send_entries: s_cnt,
                recv_entries: r_cnt,
            });
        }
    }

    result
}

/// Evaluate all interactions of a single chip on its trace, and accumulate into `map`.
///
/// `sends` and `receives` come from `Chip::sends()` / `Chip::receives()`.
/// Only **Local scope** interactions are processed.
fn collect_chip_lookups(
    chip_name: &str,
    sends: &[Interaction<F>],
    receives: &[Interaction<F>],
    main_trace: &RowMajorMatrix<F>,
    preprocessed_trace: Option<&RowMajorMatrix<F>>,
    map: &mut HashMap<LookupKey, LookupEntry>,
) {
    let height = main_trace.height();
    let empty_prep = vec![F::zero(); 1]; // fallback if no preprocessed trace

    for row in 0..height {
        let main_row = main_trace.row_slice(row);

        let prep_row: Vec<F> = if let Some(pt) = preprocessed_trace {
            pt.row_slice(row).to_vec()
        } else {
            empty_prep.clone()
        };

        // Process sends (Local scope only)
        for interaction in sends.iter().filter(|i| i.scope == InteractionScope::Local) {
            process_interaction(
                chip_name,
                interaction,
                &prep_row,
                &main_row,
                row,
                true, // is_send
                map,
            );
        }

        // Process receives (Local scope only)
        for interaction in receives.iter().filter(|i| i.scope == InteractionScope::Local) {
            process_interaction(
                chip_name,
                interaction,
                &prep_row,
                &main_row,
                row,
                false, // is_receive
                map,
            );
        }
    }
}

/// Evaluate a single interaction on one row and update the map.
fn process_interaction(
    chip_name: &str,
    interaction: &Interaction<F>,
    prep_row: &[F],
    main_row: &[F],
    row: usize,
    is_send: bool,
    map: &mut HashMap<LookupKey, LookupEntry>,
) {
    let mult: F = interaction.multiplicity.apply::<F, F>(prep_row, main_row);

    // Skip zero-multiplicity (padding rows, inactive interactions)
    if mult.is_zero() {
        return;
    }

    let mult_u32 = mult.as_canonical_u32();

    let values: Vec<u32> = interaction
        .values
        .iter()
        .map(|v| {
            let val: F = v.apply::<F, F>(prep_row, main_row);
            val.as_canonical_u32()
        })
        .collect();

    let key: LookupKey = (interaction.kind as u8, values.clone());

    let entry = map.entry(key).or_insert_with(|| LookupEntry {
        kind: interaction.kind,
        values: values.clone(),
        net_mult: 0,
        sources: Vec::new(),
    });

    if is_send {
        entry.net_mult += mult_u32 as i64;
    } else {
        entry.net_mult -= mult_u32 as i64;
    }

    // Record source for diagnostics (capped)
    if entry.sources.len() < MAX_SOURCES_PER_ENTRY {
        entry.sources.push(LookupSource {
            chip_name: chip_name.to_string(),
            row_index: row,
            is_send,
            mult: mult_u32,
        });
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Check all Local-scope lookup interactions for a single shard.
///
/// Iterates over all RiscvAir chips, generates their traces, evaluates all interactions,
/// and checks that every `(kind, values)` entry has net_mult == 0.
///
/// `skip_chips` is a list of chip names to skip (e.g., precompiles without events).
pub fn check_all_lookups_local(record: &ExecutionRecord, skip_chips: &[&str]) -> LookupCheckResult {
    let (chips, _) = RiscvAir::<F>::get_chips_and_costs();

    let mut map: HashMap<LookupKey, LookupEntry> = HashMap::new();

    for chip in &chips {
        let name = chip.name();

        if skip_chips.contains(&name.as_str()) {
            continue;
        }

        // Skip chips with no events for this record
        if !chip.included(record) {
            continue;
        }

        // Generate traces
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut output = ExecutionRecord::default();
            let main_trace = chip.generate_trace(record, &mut output).decompress();

            let prep_width = chip.preprocessed_width();
            let prep_trace = if prep_width > 0 {
                chip.generate_preprocessed_trace(&record.program).map(|c| c.decompress())
            } else {
                None
            };

            (main_trace, prep_trace)
        }));

        let (main_trace, prep_trace) = match result {
            Ok(traces) => traces,
            Err(e) => {
                let msg = if let Some(s) = e.downcast_ref::<&str>() {
                    s.to_string()
                } else if let Some(s) = e.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "unknown panic".to_string()
                };
                eprintln!("  [PANIC] {name} — {msg}");
                continue;
            }
        };

        let sends = chip.sends();
        let receives = chip.receives();

        // Sanity check: trace width must match chip width
        let expected_width = chip.width();
        let actual_width = main_trace.width();
        if expected_width != actual_width {
            eprintln!("  [WIDTH MISMATCH] {name} — expected {expected_width}, got {actual_width}");
        }
        if let Some(ref pt) = prep_trace {
            let expected_prep = chip.preprocessed_width();
            let actual_prep = pt.width();
            if expected_prep != actual_prep {
                eprintln!(
                    "  [PREP WIDTH MISMATCH] {name} — expected {expected_prep}, got {actual_prep}"
                );
            }
        }

        collect_chip_lookups(&name, sends, receives, &main_trace, prep_trace.as_ref(), &mut map);
    }

    // Partition into balanced and mismatched
    let total_entries = map.len();
    let mut balanced = 0;
    let mut mismatches = Vec::new();

    for (_, entry) in map {
        if entry.net_mult == 0 {
            balanced += 1;
        } else {
            mismatches.push(entry);
        }
    }

    // Sort mismatches by kind for readability
    mismatches.sort_by(|a, b| a.kind.cmp(&b.kind).then_with(|| a.values.cmp(&b.values)));

    LookupCheckResult { total_entries, balanced, mismatches }
}
