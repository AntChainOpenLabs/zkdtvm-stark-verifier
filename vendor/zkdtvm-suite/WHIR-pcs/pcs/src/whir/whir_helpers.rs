use std::any::{Any, TypeId};
use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, HashMap};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::rc::Rc;

use num_traits::cast::ToPrimitive;
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_dft::dft_eval::EvalsDft;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::compressed::CompressedMatrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::{Dimensions, Matrix};
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;
use serde::{Deserialize, Serialize};

use crate::utils::eqpoly::EqPolynomial;
use crate::utils::field_conversion::{flatten_to_base, reconstitute_from_base};
use crate::whir::whir_types::{
    DimAndNo, DimGroupsByLogHeight, MatrixGroupsByLogHeight, WhirError, WhirPcs,
};

thread_local! {
    static THREAD_EVALS_DFT: RefCell<HashMap<TypeId, Box<dyn Any>>> =
        RefCell::new(HashMap::new());
    static THREAD_EVALS_DFT_IN_USE: Cell<bool> = const { Cell::new(false) };
}

const STACKED_LAYOUT_AUDIT_ENV: &str = "DT_WHIR_STACKED_LAYOUT_AUDIT";

struct ThreadEvalsDftUseGuard;

impl Drop for ThreadEvalsDftUseGuard {
    fn drop(&mut self) {
        THREAD_EVALS_DFT_IN_USE.with(|in_use| in_use.set(false));
    }
}

pub(crate) fn with_thread_local_evals_dft<F, R>(f: impl FnOnce(&EvalsDft<F>) -> R) -> R
where
    F: TwoAdicField + 'static,
{
    let already_in_use = THREAD_EVALS_DFT_IN_USE.with(|in_use| {
        let value = in_use.get();
        if !value {
            in_use.set(true);
        }
        value
    });
    if already_in_use {
        let dft = EvalsDft::<F>::default();
        return f(&dft);
    }

    let _guard = ThreadEvalsDftUseGuard;
    let dft = THREAD_EVALS_DFT.with(|cache| {
        let mut cache = cache.borrow_mut();
        cache
            .entry(TypeId::of::<F>())
            .or_insert_with(|| Box::new(Rc::new(EvalsDft::<F>::default())))
            .downcast_ref::<Rc<EvalsDft<F>>>()
            .expect("thread-local EvalsDft type mismatch")
            .clone()
    });
    f(&dft)
}

fn audit_stacked_layout(
    dimensions: &[Dimensions],
    log_height: usize,
    column_alignment: usize,
    layout_width: usize,
) {
    let enabled = match std::env::var(STACKED_LAYOUT_AUDIT_ENV) {
        Ok(value) => value != "0" && !value.eq_ignore_ascii_case("false"),
        Err(_) => false,
    };
    if !enabled {
        return;
    }

    let Some(stack_height) = 1u128.checked_shl(log_height as u32) else {
        return;
    };
    let total_cells = dimensions.iter().fold(0u128, |acc, dim| {
        acc + dim.width as u128 * dim.height as u128
    });
    let lower_bound = ((total_cells + stack_height - 1) / stack_height) as usize;
    let gap = layout_width.saturating_sub(lower_bound);
    let gap_bps = if layout_width == 0 {
        0
    } else {
        ((gap as u128 * 10_000) / layout_width as u128) as usize
    };
    eprintln!(
        "whir_stacked_layout_audit log_height={log_height} column_alignment={column_alignment} matrices={} total_cells={total_cells} layout_width={layout_width} lower_bound={lower_bound} gap={gap} gap_bps={gap_bps}",
        dimensions.len()
    );
}

/// One source column placement in a stacked WHIR commitment.
///
/// This is public so a device backend can construct the exact same stacked
/// matrix without materializing it on the host.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StackedSource {
    pub matrix_idx: usize,
    pub base_col: usize,
    pub stacked_col: usize,
    pub slot: usize,
    pub selector_bits: usize,
}

/// Canonical layout shared by host and device WHIR commitment backends.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StackedBatchLayout {
    pub log_height: usize,
    pub width: usize,
    pub sources: Vec<StackedSource>,
}

#[derive(Debug, Clone, Copy)]
struct StackedCopyGroup {
    matrix_idx: usize,
    base_col: usize,
    stacked_col: usize,
    width: usize,
}

/// Blocked-slot bitmap for one stacked column, kept at the finest selector
/// depth probed so far. Bit `s` set means depth-`depth` slot `s` overlaps an
/// existing allocation. Items are placed in ascending `selector_bits` order,
/// so every existing allocation has depth <= the probe depth; refining the
/// bitmap one level is then exactly self-concatenation (a depth-d allocation
/// blocks the whole `s ≡ slot (mod 2^d)` congruence class), which keeps
/// first-zero-bit identical to the reference scan's first free slot.
#[derive(Debug, Clone)]
struct ColumnBitmap {
    words: Vec<u64>,
    depth: usize,
    free: usize,
}

impl ColumnBitmap {
    fn new() -> Self {
        Self {
            words: vec![0],
            depth: 0,
            free: 1,
        }
    }

    fn ensure_depth(&mut self, depth: usize) {
        while self.depth < depth {
            let slots = 1usize << self.depth;
            if slots < 64 {
                self.words[0] |= self.words[0] << slots;
            } else {
                let len = self.words.len();
                self.words.extend_from_within(0..len);
            }
            self.depth += 1;
            self.free *= 2;
        }
    }

    fn first_free_slot(&self) -> Option<usize> {
        let slots = 1usize << self.depth;
        for (idx, &word) in self.words.iter().enumerate() {
            let mut blocked = word;
            if slots < 64 {
                blocked |= !0u64 << slots;
            }
            if blocked != u64::MAX {
                return Some(idx * 64 + (!blocked).trailing_zeros() as usize);
            }
        }
        None
    }

    fn reserve(&mut self, slot: usize) {
        let (word, bit) = (slot / 64, slot % 64);
        debug_assert_eq!(self.words[word] >> bit & 1, 0, "slot already reserved");
        self.words[word] |= 1u64 << bit;
        self.free -= 1;
    }
}

/// First-fit slot allocator over stacked columns, output-identical to the
/// original per-pattern scan (same column order, same slot order) but with
/// per-column bitmaps instead of O(allocations) overlap walks.
#[derive(Debug, Default)]
struct SlotAllocator {
    columns: Vec<ColumnBitmap>,
}

impl SlotAllocator {
    fn new() -> Self {
        Self::default()
    }

    fn width(&self) -> usize {
        self.columns.len()
    }

    fn place_column(&mut self, selector_bits: usize) -> Option<(usize, usize)> {
        1usize.checked_shl(selector_bits as u32)?;
        for col in 0..self.columns.len() {
            let column = &mut self.columns[col];
            column.ensure_depth(selector_bits);
            debug_assert_eq!(
                column.depth, selector_bits,
                "items must be placed in ascending selector_bits order"
            );
            if column.free == 0 {
                continue;
            }
            let slot = column
                .first_free_slot()
                .expect("free > 0 must yield a slot");
            column.reserve(slot);
            return Some((col, slot));
        }
        let mut column = ColumnBitmap::new();
        column.ensure_depth(selector_bits);
        column.reserve(0);
        self.columns.push(column);
        Some((self.columns.len() - 1, 0))
    }

    fn place_column_group(
        &mut self,
        width: usize,
        selector_bits: usize,
        column_alignment: usize,
    ) -> Option<(usize, usize)> {
        let slots = 1usize.checked_shl(selector_bits as u32)?;
        let max_start = self.columns.len() + column_alignment.saturating_sub(1);
        for start_col in 0..=max_start {
            if start_col % column_alignment != 0 {
                continue;
            }
            for col in start_col..(start_col + width).min(self.columns.len()) {
                self.columns[col].ensure_depth(selector_bits);
                debug_assert_eq!(
                    self.columns[col].depth, selector_bits,
                    "items must be placed in ascending selector_bits order"
                );
            }
            if let Some(slot) = self.first_common_free_slot(start_col, width, slots) {
                self.reserve_group(start_col, width, selector_bits, slot);
                return Some((start_col, slot));
            }
        }
        None
    }

    fn first_common_free_slot(
        &self,
        start_col: usize,
        width: usize,
        slots: usize,
    ) -> Option<usize> {
        let num_words = slots.div_ceil(64);
        for word_idx in 0..num_words {
            let mut blocked = 0u64;
            for col in start_col..start_col + width {
                if let Some(column) = self.columns.get(col) {
                    blocked |= column.words[word_idx];
                }
            }
            if slots < 64 {
                blocked |= !0u64 << slots;
            }
            if blocked != u64::MAX {
                return Some(word_idx * 64 + (!blocked).trailing_zeros() as usize);
            }
        }
        None
    }

    fn reserve_group(&mut self, start_col: usize, width: usize, selector_bits: usize, slot: usize) {
        while self.columns.len() < start_col + width {
            self.columns.push(ColumnBitmap::new());
        }
        for col in start_col..start_col + width {
            let column = &mut self.columns[col];
            column.ensure_depth(selector_bits);
            column.reserve(slot);
        }
    }
}

impl StackedBatchLayout {
    pub fn max_log_height(dimensions: &[Dimensions]) -> Result<usize, ()> {
        dimensions
            .iter()
            .filter(|dim| dim.width > 0)
            .map(|dim| {
                if dim.height == 0 || !dim.height.is_power_of_two() {
                    Err(())
                } else {
                    Ok(log2_strict_usize(dim.height))
                }
            })
            .try_fold(None, |max_height, log_height| {
                let log_height = log_height?;
                Ok(Some(
                    max_height.map_or(log_height, |max: usize| max.max(log_height)),
                ))
            })?
            .ok_or(())
    }

    pub fn from_dimensions(
        dimensions: &[Dimensions],
        log_height: usize,
        column_alignment: usize,
    ) -> Result<Self, ()> {
        let column_alignment = column_alignment.max(1);
        let mut items = Vec::new();
        for (matrix_idx, dim) in dimensions.iter().enumerate() {
            if dim.width == 0 {
                continue;
            }
            if dim.height == 0 || !dim.height.is_power_of_two() {
                return Err(());
            }
            let log_matrix_height = log2_strict_usize(dim.height);
            if log_matrix_height > log_height {
                return Err(());
            }
            let selector_bits = log_height - log_matrix_height;
            let mut base_col = 0;
            while base_col < dim.width {
                let remaining = dim.width - base_col;
                // Keep complete extension-limb chunks consecutive so flattened
                // opened values still match the stacked column batching layout.
                let group_width = if column_alignment > 1 && remaining >= column_alignment {
                    column_alignment
                } else {
                    1
                };
                items.push((selector_bits, matrix_idx, base_col, group_width));
                base_col += group_width;
            }
        }

        items.sort_by_key(|&(selector_bits, matrix_idx, base_col, _)| {
            (selector_bits, matrix_idx, base_col)
        });

        let mut allocator = SlotAllocator::new();
        let mut sources = Vec::new();

        for (selector_bits, matrix_idx, base_col, group_width) in items {
            if group_width == 1 {
                let (stacked_col, slot) = allocator.place_column(selector_bits).ok_or(())?;
                sources.push(StackedSource {
                    matrix_idx,
                    base_col,
                    stacked_col,
                    slot,
                    selector_bits,
                });
            } else {
                let (start_col, slot) = allocator
                    .place_column_group(group_width, selector_bits, column_alignment)
                    .ok_or(())?;
                for offset in 0..group_width {
                    sources.push(StackedSource {
                        matrix_idx,
                        base_col: base_col + offset,
                        stacked_col: start_col + offset,
                        slot,
                        selector_bits,
                    });
                }
            }
        }

        if sources.is_empty() {
            return Err(());
        }

        let width = allocator.width();
        audit_stacked_layout(dimensions, log_height, column_alignment, width);

        Ok(Self {
            log_height,
            width,
            sources,
        })
    }

    pub fn from_matrices<F: Field>(
        matrices: &[&CompressedMatrix<F>],
        log_height: usize,
        column_alignment: usize,
    ) -> Result<Self, ()> {
        let dimensions = matrices
            .iter()
            .map(|matrix| Dimensions {
                width: matrix.width(),
                height: matrix.height(),
            })
            .collect::<Vec<_>>();
        Self::from_dimensions(&dimensions, log_height, column_alignment)
    }

    pub fn max_log_height_from_matrices<F: Field>(
        matrices: &[&CompressedMatrix<F>],
    ) -> Result<usize, ()> {
        let dimensions = matrices
            .iter()
            .map(|matrix| Dimensions {
                width: matrix.width(),
                height: matrix.height(),
            })
            .collect::<Vec<_>>();
        Self::max_log_height(&dimensions)
    }

    pub fn stacked_dimensions(&self, log_blowup: usize) -> [Dimensions; 1] {
        [Dimensions {
            width: self.width,
            height: (1usize << self.log_height) << log_blowup,
        }]
    }
}

fn stacked_copy_groups_by_selector(
    layout: &StackedBatchLayout,
) -> Vec<HashMap<usize, Vec<StackedCopyGroup>>> {
    let mut groups_by_selector: Vec<HashMap<usize, Vec<StackedCopyGroup>>> =
        (0..=layout.log_height).map(|_| HashMap::new()).collect();

    for source in &layout.sources {
        let slot_groups = groups_by_selector[source.selector_bits]
            .entry(source.slot)
            .or_default();
        if let Some(group) = slot_groups.last_mut() {
            if group.matrix_idx == source.matrix_idx
                && group.base_col + group.width == source.base_col
                && group.stacked_col + group.width == source.stacked_col
            {
                group.width += 1;
                continue;
            }
        }
        slot_groups.push(StackedCopyGroup {
            matrix_idx: source.matrix_idx,
            base_col: source.base_col,
            stacked_col: source.stacked_col,
            width: 1,
        });
    }

    groups_by_selector
}

pub fn build_stacked_evaluations<F: Field>(
    matrices: &[&CompressedMatrix<F>],
    layout: &StackedBatchLayout,
) -> RowMajorMatrix<F> {
    let stack_height = 1usize << layout.log_height;
    let mut stacked_values = vec![F::zero(); stack_height * layout.width];

    let groups_by_selector = stacked_copy_groups_by_selector(layout);

    stacked_values
        .par_chunks_mut(layout.width)
        .enumerate()
        .for_each(|(stacked_row, out_row)| {
            for (selector_bits, groups_by_slot) in groups_by_selector.iter().enumerate() {
                let slot = if selector_bits == 0 {
                    0
                } else {
                    stacked_row & ((1usize << selector_bits) - 1)
                };
                let Some(groups) = groups_by_slot.get(&slot) else {
                    continue;
                };
                let source_row = stacked_row >> selector_bits;
                for group in groups {
                    let src_row = matrices[group.matrix_idx].row_slice(source_row);
                    out_row[group.stacked_col..group.stacked_col + group.width]
                        .copy_from_slice(&src_row[group.base_col..group.base_col + group.width]);
                }
            }
        });

    RowMajorMatrix::new(stacked_values, layout.width)
}

#[derive(Clone)]
pub(crate) struct StackedBatchCoefficients<EF> {
    pub(crate) column_coeffs: Vec<EF>,
    // reserved for future per-chunk batching
    pub(crate) chunk_coeffs: Vec<EF>,
}

/// Index structure for mapping a flat global matrix index back to its `(batch_idx, mat_idx)` pair.
///
/// Given `matrices_size: Vec<Vec<Dimensions>>`, each inner `Vec<Dimensions>` corresponds to one
/// batch. This struct builds a prefix-sum array so that a global index (across all batches) can
/// be efficiently mapped to the batch and matrix it belongs to via binary search.
pub(crate) struct MatricesSizeIndex {
    /// Prefix sum array: `prefix_sums[i]` is the total number of matrices in the first `i` batches.
    prefix_sums: Vec<usize>,
}

impl MatricesSizeIndex {
    /// Create index structure from matrices_size
    pub(crate) fn new(matrices_size: &[Vec<Dimensions>]) -> Self {
        let mut prefix_sums = Vec::with_capacity(matrices_size.len() + 1);
        prefix_sums.push(0);

        let mut sum = 0;
        for vec in matrices_size {
            sum += vec.len();
            prefix_sums.push(sum);
        }

        Self { prefix_sums }
    }

    /// Find the position (i, j) corresponding to the global index
    /// - i: index of the outer Vec
    /// - j: index of the inner Vec
    pub(crate) fn find_position(&self, index: usize) -> (usize, usize) {
        // Use binary search to determine the outer index i
        let i = match self.prefix_sums.binary_search(&index) {
            Ok(exact) => exact,
            Err(insert_pos) => insert_pos - 1,
        };

        // Calculate the inner index j
        let j = index - self.prefix_sums[i];

        (i, j)
    }
}

impl<F, InputMmcs, FriMmcs, EF, Challenger> WhirPcs<F, InputMmcs, FriMmcs, EF, Challenger>
where
    F: TwoAdicField,
    InputMmcs: Mmcs<F> + Send + Sync,
    InputMmcs::ProverData<RowMajorMatrix<F>>: Sync,
    FriMmcs: Mmcs<EF> + Send + Sync,
    EF: TwoAdicField + ExtensionField<F>,
    Challenger:
        FieldChallenger<F> + CanObserve<FriMmcs::Commitment> + GrindingChallenger<Witness = F>,
{
    /// Validate that the dimensions of polynomials, prover data, and opened values are consistent.
    pub(crate) fn validate_open_inputs<P>(
        &self,
        polynomials_batch: &[Vec<CompressedMatrix<F>>],
        prover_data_batch: &[P],
        opened_values: &[Vec<Vec<EF>>],
    ) -> Result<(), WhirError<FriMmcs::Error, InputMmcs::Error>> {
        if polynomials_batch.len() != opened_values.len()
            || polynomials_batch.len() != prover_data_batch.len()
        {
            return Err(WhirError::InvalidInputError);
        }

        for (batch, vals) in polynomials_batch.iter().zip(opened_values.iter()) {
            if batch.len() != vals.len() {
                return Err(WhirError::InvalidInputError);
            }
            for (matrix, col_vals) in batch.iter().zip(vals.iter()) {
                if col_vals.len() != matrix.width() && col_vals.len() * EF::D != matrix.width() {
                    return Err(WhirError::InvalidInputError);
                }
            }
        }
        Ok(())
    }

    /// Validate that the dimensions of commitments, matrix sizes, and opened values are consistent.
    pub(crate) fn validate_verify_inputs(
        &self,
        commitment_batch: &[InputMmcs::Commitment],
        matrices_size_batch: &[Vec<Dimensions>],
        opened_values_batch: &[Vec<Vec<EF>>],
    ) -> Result<(), WhirError<FriMmcs::Error, InputMmcs::Error>> {
        if matrices_size_batch.len() != opened_values_batch.len()
            || commitment_batch.len() != matrices_size_batch.len()
        {
            return Err(WhirError::InvalidInputError);
        }

        for (matrices_size, opened_values) in
            matrices_size_batch.iter().zip(opened_values_batch.iter())
        {
            if matrices_size.len() != opened_values.len() {
                return Err(WhirError::InvalidInputError);
            }
            for (dim, vals) in matrices_size.iter().zip(opened_values.iter()) {
                if vals.len() != dim.width && vals.len() * EF::D != dim.width {
                    return Err(WhirError::InvalidInputError);
                }
                // [F-022] Reject heights that are not a nonzero power of two.
                // Downstream `log2_strict_usize(dim.height)` and the
                // `index >> shift` arithmetic assume this; an untrusted proof
                // with height 0 or a non-power-of-two would otherwise panic
                // (verifier DoS).
                if dim.width != 0 && (dim.height == 0 || !dim.height.is_power_of_two()) {
                    return Err(WhirError::InvalidInputError);
                }
            }
        }
        Ok(())
    }

    /// Group dimensions with their opened values by log_height (descending key order in BTreeMap).
    pub(crate) fn group_dims_by_log_height<'a>(
        matrices_size: &'a [DimAndNo],
        flat_opened_values: &[&'a Vec<EF>],
    ) -> DimGroupsByLogHeight<'a, EF> {
        let mut groups: DimGroupsByLogHeight<'a, EF> = BTreeMap::new();
        for (dim_no, values) in matrices_size.iter().zip(flat_opened_values.iter()) {
            groups
                .entry(log2_strict_usize(dim_no.dim.height))
                .or_default()
                .push((dim_no, values));
        }
        groups
    }

    /// Group flattened compressed matrices by their total_height, paired with their opened values.
    pub(crate) fn group_by_log_height<'a>(
        &self,
        polynomials: &'a [CompressedMatrix<F>],
        flat_opened_values: &[&'a Vec<EF>],
    ) -> Result<MatrixGroupsByLogHeight<'a, F, EF>, WhirError<FriMmcs::Error, InputMmcs::Error>>
    {
        let mut groups: MatrixGroupsByLogHeight<'a, F, EF> = BTreeMap::new();

        for (matrix, values) in polynomials.iter().zip(flat_opened_values.iter()) {
            let height = matrix.height();
            if !height.is_power_of_two() {
                return Err(WhirError::InvalidInputError);
            }
            groups
                .entry(log2_strict_usize(height))
                .or_default()
                .push((matrix, values));
        }
        Ok(groups)
    }

    /// Encode a polynomial (given as evaluations over the hypercube) into a Reed-Solomon codeword.
    ///
    /// Steps: repeat (blowup) → twiddle-free DFT → bit-reverse output.
    /// No input bit-reverse: compatible with little-endian (even/odd) folding.
    pub(crate) fn encode_to_codeword(
        &self,
        evals: &[EF],
        log_blowup: usize,
        dft: &EvalsDft<F>,
    ) -> Vec<EF> {
        let mut coeffs: Vec<EF> = evals.to_vec();

        let repeat_times = 1 << log_blowup;
        let orig_len = coeffs.len();
        coeffs.reserve(orig_len * (repeat_times - 1));
        for _ in 1..repeat_times {
            coeffs.extend_from_within(0..orig_len);
        }

        let base_values: Vec<F> = unsafe { flatten_to_base(coeffs) };
        let dft_output = dft
            .dft_batch_by_evals_skip(RowMajorMatrix::new(base_values, EF::D), log_blowup)
            .to_row_major_matrix();
        unsafe { reconstitute_from_base(dft_output.values) }
    }

    /// Find a proof-of-work witness by trial using the given number of grinding bits.
    pub(crate) fn find_pow_witness(
        &self,
        challenger: &mut Challenger,
        grinding_bits: usize,
    ) -> Result<Vec<F>, WhirError<FriMmcs::Error, InputMmcs::Error>> {
        let order = F::order().to_u64().expect("F::order() should fit in u64");

        for i in 0..order {
            let nonce = F::from_canonical_u64(i);
            if let Ok(witness) = catch_unwind(AssertUnwindSafe(|| {
                let mut trial = challenger.clone();
                trial.observe(nonce);
                trial.grind(grinding_bits)
            })) {
                challenger.observe(nonce);
                challenger.observe(witness);
                assert_eq!(challenger.sample_bits(grinding_bits), 0);
                return Ok(vec![nonce, witness]);
            }
        }
        Err(WhirError::CannotFindPowWitness)
    }

    pub(crate) fn extend_stacked_opening_point(
        &self,
        opening_point: &[EF],
        stack_log_height: usize,
        challenger: &mut Challenger,
    ) -> Result<Vec<EF>, WhirError<FriMmcs::Error, InputMmcs::Error>> {
        if opening_point.len() > stack_log_height {
            return Err(WhirError::InvalidInputError);
        }
        let mut point = opening_point.to_vec();
        while point.len() < stack_log_height {
            point.push(challenger.sample_ext_element());
        }
        Ok(point)
    }

    pub(crate) fn batch_uses_flattened_ext_dims(
        &self,
        dimensions: &[Dimensions],
        opened_values: &[Vec<EF>],
    ) -> bool {
        dimensions
            .iter()
            .zip(opened_values.iter())
            .any(|(dim, values)| values.len() != dim.width && values.len() * EF::D == dim.width)
    }
}

// ── Stacking reduction helpers ──────────────────────────────────────────────
//
// These implement the Q_j coefficient polynomial construction for the stacking
// reduction sumcheck: T = Σ_{x ∈ {0,1}^L} Σ_c F_c(x) · Q_c(x).
//
// λ assignment: sources are numbered globally across batches in the order
// (batch_0.source_0, batch_0.source_1, ..., batch_1.source_0, ...).
// For ext-flattened batches, D consecutive base-column sources sharing the same
// (matrix_idx, slot, selector_bits) count as one logical source that consumes
// one λ power; the D base limbs each get the same λ^i multiplied by the
// appropriate basis element.

/// Compute per-source λ coefficients for a batch.
///
/// Returns `(per_source_coeff, raw_lambda_powers, lambdas_consumed, next_lambda_power)`.
/// - `per_source_coeff[i]` is the full Q-polynomial coefficient for source `i`:
///   non-flattened: `λ^{...}`, flattened: `λ^{...} · basis(limb)`.
/// - `raw_lambda_powers[i]` is the plain `λ^{...}` without basis (for T computation).
fn source_lambda_coeffs<EF: ExtensionField<F>, F: Field>(
    layout: &StackedBatchLayout,
    lambda_power_start: EF,
    lambda: EF,
    uses_flattened_ext: bool,
) -> (Vec<EF>, Vec<EF>, usize, EF) {
    let mut coeffs = Vec::with_capacity(layout.sources.len());
    let mut raw_powers = Vec::with_capacity(layout.sources.len());
    let mut current_power = lambda_power_start;
    let mut lambdas_consumed = 0usize;
    let mut prev_logical: Option<(usize, usize, usize)> = None;

    for source in &layout.sources {
        if uses_flattened_ext {
            let logical_key = (source.matrix_idx, source.slot, source.base_col / EF::D);
            let is_new = prev_logical.map_or(true, |prev| prev != logical_key);
            if is_new {
                if lambdas_consumed > 0 || prev_logical.is_some() {
                    current_power *= lambda;
                }
                prev_logical = Some(logical_key);
                lambdas_consumed += 1;
            }
            let limb = source.base_col % EF::D;
            let basis = EF::from_base_fn(|i| if i == limb { F::one() } else { F::zero() });
            coeffs.push(current_power * basis);
            raw_powers.push(current_power);
        } else {
            if lambdas_consumed > 0 {
                current_power *= lambda;
            }
            coeffs.push(current_power);
            raw_powers.push(current_power);
            lambdas_consumed += 1;
        }
    }

    let next_power = if lambdas_consumed > 0 {
        current_power * lambda
    } else {
        current_power
    };
    (coeffs, raw_powers, lambdas_consumed, next_power)
}

/// Build the Q_c evaluation matrix for one batch.
///
/// The output is row-major with shape `2^L × layout.width`. For each source in
/// `layout` belonging to stacked column `col`, this adds
/// `λ^{global_idx} · selector · eq_prefix` to that column.
///
/// Returns `(lambdas_consumed, next_lambda_power)`.
pub(crate) fn build_q_matrix_for_batch<EF: ExtensionField<F>, F: Field>(
    layout: &StackedBatchLayout,
    full_opening_point: &[EF],
    lambda: EF,
    lambda_power_start: EF,
    uses_flattened_ext: bool,
) -> (RowMajorMatrix<EF>, usize, EF) {
    let l = layout.log_height;
    let stack_height = 1usize << l;
    let mut q_values = vec![EF::zero(); stack_height * layout.width];

    let (src_coeffs, _raw_powers, consumed, next_power) =
        source_lambda_coeffs::<EF, F>(layout, lambda_power_start, lambda, uses_flattened_ext);

    let mut eq_prefix_cache: HashMap<usize, Vec<EF>> = HashMap::new();

    for (src_idx, source) in layout.sources.iter().enumerate() {
        let b = source.selector_bits;
        let eq_prefix = eq_prefix_cache.entry(b).or_insert_with(|| {
            let prefix_len = l - b;
            if prefix_len == 0 {
                vec![EF::one()]
            } else {
                EqPolynomial::new(full_opening_point[0..prefix_len].to_vec()).evals()
            }
        });

        let coeff = src_coeffs[src_idx];
        let col = source.stacked_col;
        let slot = source.slot;

        if b == 0 {
            for row in 0..stack_height {
                q_values[row * layout.width + col] += coeff * eq_prefix[row];
            }
        } else {
            for (prefix, eq) in eq_prefix.iter().enumerate() {
                let row = (prefix << b) | slot;
                q_values[row * layout.width + col] += coeff * *eq;
            }
        }
    }

    (
        RowMajorMatrix::new(q_values, layout.width),
        consumed,
        next_power,
    )
}

/// Compute the stacking reduction target T = Σ_i λ^i · original_claim_i
/// for one batch.  Returns `(T_batch, lambdas_consumed, next_lambda_power)`.
pub(crate) fn reduction_target_for_batch<EF: ExtensionField<F>, F: Field>(
    layout: &StackedBatchLayout,
    dimensions: &[Dimensions],
    opened_values: &[Vec<EF>],
    lambda: EF,
    lambda_power_start: EF,
    uses_flattened_ext: bool,
) -> (EF, usize, EF) {
    let (_src_coeffs, raw_powers, consumed, next_power) =
        source_lambda_coeffs::<EF, F>(layout, lambda_power_start, lambda, uses_flattened_ext);

    let mut t = EF::zero();
    for (src_idx, source) in layout.sources.iter().enumerate() {
        let dim = &dimensions[source.matrix_idx];
        let values = &opened_values[source.matrix_idx];

        if uses_flattened_ext && values.len() * EF::D == dim.width {
            // For flattened ext, only the first limb (base_col % D == 0) adds the
            // full extension-field opened_value with the raw λ power. Other limbs
            // are skipped because the D base contributions sum to the full ext
            // value through Σ_k basis(k) · base_k = identity.
            if source.base_col % EF::D != 0 {
                continue;
            }
            let ext_col = source.base_col / EF::D;
            t += raw_powers[src_idx] * values[ext_col];
        } else {
            t += raw_powers[src_idx] * values[source.base_col];
        }
    }

    (t, consumed, next_power)
}

/// Evaluate Q_c(u) for all stacked columns in one batch.
/// Returns `(q_values, lambdas_consumed, next_lambda_power)`.
pub(crate) fn compute_q_at_point_for_batch<EF: ExtensionField<F>, F: Field>(
    layout: &StackedBatchLayout,
    full_opening_point: &[EF],
    u: &[EF],
    lambda: EF,
    lambda_power_start: EF,
    uses_flattened_ext: bool,
) -> (Vec<EF>, usize, EF) {
    let l = layout.log_height;
    let mut q_values = vec![EF::zero(); layout.width];

    let (src_coeffs, _raw_powers, consumed, next_power) =
        source_lambda_coeffs::<EF, F>(layout, lambda_power_start, lambda, uses_flattened_ext);

    let mut selector_cache: HashMap<(usize, usize), EF> = HashMap::new();
    let mut eq_prefix_cache: HashMap<usize, EF> = HashMap::new();

    for (src_idx, source) in layout.sources.iter().enumerate() {
        let b = source.selector_bits;

        let selector_val = *selector_cache.entry((b, source.slot)).or_insert_with(|| {
            if b == 0 {
                EF::one()
            } else {
                let start = l - b;
                (0..b)
                    .map(|i| {
                        let bit = (source.slot >> (b - 1 - i)) & 1;
                        if bit == 0 {
                            EF::one() - u[start + i]
                        } else {
                            u[start + i]
                        }
                    })
                    .product()
            }
        });

        let eq_prefix_val = *eq_prefix_cache.entry(b).or_insert_with(|| {
            let prefix_len = l - b;
            if prefix_len == 0 {
                EF::one()
            } else {
                EqPolynomial::new(full_opening_point[0..prefix_len].to_vec())
                    .evaluate(&u[0..prefix_len])
            }
        });

        q_values[source.stacked_col] += src_coeffs[src_idx] * selector_val * eq_prefix_val;
    }

    (q_values, consumed, next_power)
}

#[cfg(test)]
mod slot_allocator_tests {
    use super::*;

    // ── Reference implementation: the original per-pattern scan allocator ──
    // Kept verbatim so the bitmap allocator can be checked for exact output
    // equality (the layout defines the committed stacking, so any placement
    // drift would change commitments and break every frozen vk).

    #[derive(Debug, Clone, Copy)]
    struct SlotPattern {
        depth: usize,
        slot: usize,
    }

    impl SlotPattern {
        fn overlaps(self, other: Self) -> bool {
            if self.depth <= other.depth {
                let mask = (1usize << self.depth) - 1;
                (other.slot & mask) == self.slot
            } else {
                let mask = (1usize << other.depth) - 1;
                (self.slot & mask) == other.slot
            }
        }
    }

    #[derive(Debug, Clone, Default)]
    struct RefAllocator {
        allocations: Vec<SlotPattern>,
    }

    fn can_place_pattern(allocators: &[RefAllocator], col: usize, pattern: SlotPattern) -> bool {
        allocators.get(col).is_none_or(|allocator| {
            allocator
                .allocations
                .iter()
                .all(|&existing| !existing.overlaps(pattern))
        })
    }

    fn reserve_pattern(allocators: &mut Vec<RefAllocator>, col: usize, pattern: SlotPattern) {
        while allocators.len() <= col {
            allocators.push(RefAllocator::default());
        }
        allocators[col].allocations.push(pattern);
    }

    fn ref_place_column(
        allocators: &mut Vec<RefAllocator>,
        selector_bits: usize,
    ) -> Option<(usize, usize)> {
        let num_slots = 1usize.checked_shl(selector_bits as u32)?;
        let max_col = allocators.len();
        for col in 0..=max_col {
            for slot in 0..num_slots {
                let pattern = SlotPattern {
                    depth: selector_bits,
                    slot,
                };
                if can_place_pattern(allocators, col, pattern) {
                    reserve_pattern(allocators, col, pattern);
                    return Some((col, slot));
                }
            }
        }
        None
    }

    fn ref_place_column_group(
        allocators: &mut Vec<RefAllocator>,
        width: usize,
        selector_bits: usize,
        column_alignment: usize,
    ) -> Option<(usize, usize)> {
        let num_slots = 1usize.checked_shl(selector_bits as u32)?;
        let max_start = allocators.len() + column_alignment.saturating_sub(1);
        for start_col in 0..=max_start {
            if start_col % column_alignment != 0 {
                continue;
            }
            for slot in 0..num_slots {
                let pattern = SlotPattern {
                    depth: selector_bits,
                    slot,
                };
                if (0..width).all(|off| can_place_pattern(allocators, start_col + off, pattern)) {
                    for col in start_col..start_col + width {
                        reserve_pattern(allocators, col, pattern);
                    }
                    return Some((start_col, slot));
                }
            }
        }
        None
    }

    fn ref_from_dimensions(
        dimensions: &[Dimensions],
        log_height: usize,
        column_alignment: usize,
    ) -> Result<StackedBatchLayout, ()> {
        let column_alignment = column_alignment.max(1);
        let mut items = Vec::new();
        for (matrix_idx, dim) in dimensions.iter().enumerate() {
            if dim.width == 0 {
                continue;
            }
            if dim.height == 0 || !dim.height.is_power_of_two() {
                return Err(());
            }
            let log_matrix_height = log2_strict_usize(dim.height);
            if log_matrix_height > log_height {
                return Err(());
            }
            let selector_bits = log_height - log_matrix_height;
            let mut base_col = 0;
            while base_col < dim.width {
                let remaining = dim.width - base_col;
                let group_width = if column_alignment > 1 && remaining >= column_alignment {
                    column_alignment
                } else {
                    1
                };
                items.push((selector_bits, matrix_idx, base_col, group_width));
                base_col += group_width;
            }
        }
        items.sort_by_key(|&(selector_bits, matrix_idx, base_col, _)| {
            (selector_bits, matrix_idx, base_col)
        });

        let mut allocators: Vec<RefAllocator> = Vec::new();
        let mut sources = Vec::new();
        for (selector_bits, matrix_idx, base_col, group_width) in items {
            if group_width == 1 {
                let (stacked_col, slot) =
                    ref_place_column(&mut allocators, selector_bits).ok_or(())?;
                sources.push(StackedSource {
                    matrix_idx,
                    base_col,
                    stacked_col,
                    slot,
                    selector_bits,
                });
            } else {
                let (start_col, slot) = ref_place_column_group(
                    &mut allocators,
                    group_width,
                    selector_bits,
                    column_alignment,
                )
                .ok_or(())?;
                for offset in 0..group_width {
                    sources.push(StackedSource {
                        matrix_idx,
                        base_col: base_col + offset,
                        stacked_col: start_col + offset,
                        slot,
                        selector_bits,
                    });
                }
            }
        }
        if sources.is_empty() {
            return Err(());
        }
        Ok(StackedBatchLayout {
            log_height,
            width: allocators.len(),
            sources,
        })
    }

    // Simple deterministic LCG so the sweep is reproducible without a rand dep.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 33
        }
        fn below(&mut self, bound: u64) -> u64 {
            self.next() % bound
        }
    }

    #[test]
    fn bitmap_allocator_matches_reference() {
        let mut rng = Lcg(0x5eed_1234_5678_9abc);
        for case in 0..200 {
            let log_height = 1 + (rng.below(10) as usize); // 1..=10
            let num_matrices = 1 + rng.below(12) as usize;
            let dims: Vec<Dimensions> = (0..num_matrices)
                .map(|_| {
                    let h_log = rng.below(log_height as u64 + 1) as usize;
                    Dimensions {
                        width: rng.below(40) as usize, // width 0 allowed: skipped path
                        height: 1usize << h_log,
                    }
                })
                .collect();
            for alignment in [1usize, 2, 5] {
                let fast = StackedBatchLayout::from_dimensions(&dims, log_height, alignment);
                let reference = ref_from_dimensions(&dims, log_height, alignment);
                match (&fast, &reference) {
                    (Ok(a), Ok(b)) => assert_eq!(
                        a, b,
                        "layout mismatch: case={case} alignment={alignment} \
                         log_height={log_height} dims={dims:?}"
                    ),
                    (Err(()), Err(())) => {}
                    _ => panic!(
                        "ok/err mismatch: case={case} alignment={alignment} \
                         log_height={log_height} dims={dims:?} fast_ok={} ref_ok={}",
                        fast.is_ok(),
                        reference.is_ok()
                    ),
                }
            }
        }
    }

    #[test]
    fn bitmap_allocator_matches_reference_deep_selectors() {
        // Deep selector regime: tiny matrices stacked very high (the L4 root
        // case that made the O(cols·slots·allocs) reference explode).
        let dims = [
            Dimensions {
                width: 7,
                height: 1 << 12,
            },
            Dimensions {
                width: 23,
                height: 1 << 8,
            },
            Dimensions {
                width: 3,
                height: 1 << 2,
            },
            Dimensions {
                width: 11,
                height: 1,
            },
            Dimensions {
                width: 5,
                height: 1 << 5,
            },
            Dimensions {
                width: 40,
                height: 1 << 12,
            },
        ];
        for alignment in [1usize, 5] {
            let fast = StackedBatchLayout::from_dimensions(&dims, 12, alignment).unwrap();
            let reference = ref_from_dimensions(&dims, 12, alignment).unwrap();
            assert_eq!(fast, reference, "alignment={alignment}");
        }
    }
}

// Mirrors the equivalence test that lives next to `CompressedMatrix::
// flatten_to_base` in p3-matrix; duplicated here because the standalone
// Plonky3 workspace does not currently resolve on the remote toolchain, and
// this invariant (P1: the padding-preserving perm-trace flatten is logically
// identical to the old decompress+flatten detour, hence commit-identical)
// must stay covered by a suite that actually runs in CI.
#[cfg(test)]
mod compressed_flatten_tests {
    use p3_baby_bear::BabyBear;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{AbstractExtensionField, AbstractField};
    use p3_matrix::compressed::{CompressedMatrix, PaddingRow};
    use p3_matrix::dense::RowMajorMatrix;
    use p3_matrix::Matrix;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    fn ef(i: u32) -> EF {
        EF::from_base_slice(&[
            F::from_canonical_u32(i),
            F::from_canonical_u32(i.wrapping_mul(7).wrapping_add(3)),
            F::from_canonical_u32(i.wrapping_mul(13).wrapping_add(11)),
            F::from_canonical_u32(i.wrapping_mul(31).wrapping_add(1)),
        ])
    }

    #[test]
    fn compressed_flatten_to_base_matches_decompress_detour() {
        let width = 3;
        for (stored_rows, total_height, padding_row) in [
            (
                5usize,
                8usize,
                PaddingRow::General(vec![ef(100), ef(200), ef(300)]),
            ),
            (3, 8, PaddingRow::Zero { width: 3 }),
            (
                6,
                16,
                PaddingRow::Constant {
                    value: ef(42),
                    width: 3,
                },
            ),
            (4, 4, PaddingRow::None),
        ] {
            let main_values: Vec<EF> = (0..stored_rows * width).map(|i| ef(i as u32 + 1)).collect();
            let main = RowMajorMatrix::new(main_values, width);
            let compressed: CompressedMatrix<EF, EF> =
                CompressedMatrix::new(main, padding_row.clone(), total_height);

            let fast: CompressedMatrix<F, F> = compressed.flatten_to_base::<F>();
            let detour: CompressedMatrix<F, F> = CompressedMatrix::from_full_matrix_no_padding(
                compressed.decompress().flatten_to_base(),
            );

            assert_eq!(fast.width(), detour.width(), "width: {padding_row:?}");
            assert_eq!(
                fast.total_height, detour.total_height,
                "height: {padding_row:?}"
            );
            assert_eq!(
                fast.decompress().values,
                detour.decompress().values,
                "decompressed content: {padding_row:?}"
            );
            // The fast path must actually keep the padding compressed.
            assert_eq!(fast.main.height(), stored_rows.min(total_height));
        }
    }
}
