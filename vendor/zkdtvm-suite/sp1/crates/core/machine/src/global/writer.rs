use std::{borrow::BorrowMut, ops::Range};

use dt_core_executor::{
    events::{ByteLookupEvent, ByteRecord, GlobalInteractionEvent, GlobalSourceId},
    ExecutionRecord, PrepareGlobalProgramError, PreparedGlobalProgram,
};
use dt_stark::{
    air::{GlobalClaim, GlobalState, GlobalStateInterval},
    global_d11::{
        apply_direction, construct_map, fixed_padding_dummy, D11ProjectivePointV1, D11Sparse7,
        program_global_seed, GlobalBoundaryError, GlobalMapErrorV1, GlobalMapWitnessV1,
        GlobalPackInputV1, GlobalSignedMapRowV1, D11, HALF_BASE_MINUS_ONE,
    },
    sumcheck::trace::{CompressedMatrix, PaddingRow},
    InteractionKind,
};
use hashbrown::HashMap;
use p3_field::{Field, PrimeField32};
use p3_koala_bear::KoalaBear;
use p3_matrix::dense::RowMajorMatrix;
use rayon::prelude::*;

use crate::{global::sources::GLOBAL_PRODUCER_SCHEDULE, utils::padded_rows_threshold};

use super::{
    columns::{
        D11PointCols, GlobalCols, GlobalTileReducerCols, NUM_GLOBAL_COLS,
        NUM_GLOBAL_TILE_REDUCER_COLS, REDUCER_LEAF_END_VALUE, REDUCER_LEAF_GAP_BITS,
        REDUCER_LEAF_GAP_BITS_START, REDUCER_LEAF_K_VALUE, REDUCER_LEAF_P_BITS,
        REDUCER_LEAF_P_BITS_START, REDUCER_PRODUCT_CONTINUE_VALUE,
        REDUCER_PRODUCT_INFINITY_VALUE, REDUCER_PRODUCT_REBASE_VALUE,
        REDUCER_PRODUCT_TO_MIDDLE_VALUE, REDUCER_PRODUCT_TO_NODE_VALUE,
        REDUCER_PRODUCT_TO_ROOT_VALUE, REDUCER_PRODUCT_WAVE_VALUE,
    },
    constraints::{
        constraint_residuals, header, map_quotient, mixed_output_with_quotients,
        mul_with_quotient, packed_x,
    },
};

pub type GlobalByteLookupDelta = HashMap<ByteLookupEvent, usize>;
const MAX_GLOBAL_LOG_HEIGHT: usize = 22;
const MEMORY_SOURCE_CHUNK_ROWS: usize = 4096;

#[derive(Clone, Copy, Debug)]
struct PreparationOptions {
    memory_chunk_rows: usize,
}

impl Default for PreparationOptions {
    fn default() -> Self {
        Self { memory_chunk_rows: MEMORY_SOURCE_CHUNK_ROWS }
    }
}

#[derive(Clone, Debug)]
struct GlobalSourceChunk {
    source: GlobalSourceId,
    batch: crate::global::sources::GlobalProducerBatch,
    source_range: Range<usize>,
}

/// One immutable count/range authority shared by admission, allocation, and population.
#[derive(Clone, Debug)]
struct GlobalSourcePlan {
    raw_rows: usize,
    chunks: Vec<GlobalSourceChunk>,
}

impl GlobalSourcePlan {
    fn build(
        input: &ExecutionRecord,
        memory_chunk_rows: usize,
        local_memory_event_count: usize,
    ) -> Result<Self, GlobalPrepareError> {
        if memory_chunk_rows == 0 {
            return Err(GlobalPrepareError::InvalidChunkSize);
        }

        let mut raw_rows = 0usize;
        let mut chunks = Vec::new();
        for batch in GLOBAL_PRODUCER_SCHEDULE {
            let count = if batch.source_id == GlobalSourceId::MemoryLocal {
                local_memory_event_count
                    .checked_mul(2)
                    .ok_or(GlobalPrepareError::SourceCountOverflow)?
            } else {
                batch.endpoint_count(input)
            };
            raw_rows =
                raw_rows.checked_add(count).ok_or(GlobalPrepareError::SourceCountOverflow)?;
            let chunk_rows = if matches!(
                batch.source_id,
                GlobalSourceId::MemoryInitialize
                    | GlobalSourceId::MemoryFinalize
                    | GlobalSourceId::MemoryLocal
            ) {
                memory_chunk_rows
            } else {
                count.max(1)
            };
            for start in (0..count).step_by(chunk_rows) {
                let end = start
                    .checked_add(chunk_rows)
                    .ok_or(GlobalPrepareError::SourceTaskRangeOverflow)?
                    .min(count);
                chunks.push(GlobalSourceChunk {
                    source: batch.source_id,
                    batch,
                    source_range: start..end,
                });
            }
        }

        Ok(Self { raw_rows, chunks })
    }
}

/// Failure to admit or construct the canonical Global trace artifact.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GlobalPrepareError {
    SourceCountOverflow,
    InvalidChunkSize,
    TraceElementCountOverflow,
    TraceByteCountOverflow,
    TraceAllocationFailed {
        elements: usize,
    },
    SourceTaskRangeOverflow,
    SourceTaskRowCountMismatch {
        expected: usize,
        actual: usize,
    },
    ByteMultiplicityOverflow,
    Program(PrepareGlobalProgramError),
    Boundary(GlobalBoundaryError),
    Map {
        source: GlobalSourceId,
        source_ordinal: usize,
        endpoint: GlobalInteractionEvent,
        cause: GlobalMapErrorV1,
    },
    HeightExceeded {
        raw_rows: usize,
        padded_rows: usize,
        maximum_rows: usize,
    },
}

impl From<PrepareGlobalProgramError> for GlobalPrepareError {
    fn from(value: PrepareGlobalProgramError) -> Self {
        Self::Program(value)
    }
}

impl From<GlobalBoundaryError> for GlobalPrepareError {
    fn from(value: GlobalBoundaryError) -> Self {
        Self::Boundary(value)
    }
}

/// Single retained artifact: one real-row SoA matrix plus one repeated row.
pub struct PreparedGlobalTrace<F: Field = KoalaBear> {
    pub trace: CompressedMatrix<F>,
    pub reducer_trace: CompressedMatrix<F>,
    pub byte_delta: GlobalByteLookupDelta,
    pub raw_rows: usize,
    pub start: D11ProjectivePointV1<F>,
    pub terminal: D11ProjectivePointV1<F>,
    pub log_height: u8,
    pub claim: GlobalClaim<F>,
}

/// The retained half of the artifact after its compact Byte delta is consumed.
pub struct RetainedGlobalTrace<F: Field = KoalaBear> {
    pub trace: CompressedMatrix<F>,
    pub reducer_trace: CompressedMatrix<F>,
    pub raw_rows: usize,
    pub start: D11ProjectivePointV1<F>,
    pub terminal: D11ProjectivePointV1<F>,
    pub log_height: u8,
    pub claim: GlobalClaim<F>,
}

impl<F: Field> PreparedGlobalTrace<F> {
    /// Consume the dependency delta exactly once while retaining the already-built matrix.
    #[must_use]
    pub fn consume_byte_delta(self, record: &mut ExecutionRecord) -> RetainedGlobalTrace<F> {
        let Self { trace, reducer_trace, byte_delta, raw_rows, start, terminal, log_height, claim } =
            self;
        record.add_byte_lookup_events_from_maps(vec![&byte_delta]);
        RetainedGlobalTrace { trace, reducer_trace, raw_rows, start, terminal, log_height, claim }
    }
}

fn state_from_point<F: Field>(point: &D11ProjectivePointV1<F>) -> GlobalState<F> {
    GlobalState {
        x: *point.x.coefficients(),
        y: *point.y.coefficients(),
        z: *point.z.coefficients(),
    }
}

impl<F: Field> PreparedGlobalTrace<F> {
    #[must_use]
    pub fn claim(&self) -> GlobalClaim<F> {
        self.claim
    }
}

impl<F: Field> RetainedGlobalTrace<F> {
    #[must_use]
    pub fn claim(&self) -> GlobalClaim<F> {
        self.claim
    }
}

fn merge_delta(
    mut left: GlobalByteLookupDelta,
    mut right: GlobalByteLookupDelta,
) -> Result<GlobalByteLookupDelta, GlobalPrepareError> {
    if left.len() < right.len() {
        core::mem::swap(&mut left, &mut right);
    }
    for (event, count) in right {
        let entry = left.entry(event).or_default();
        *entry = entry.checked_add(count).ok_or(GlobalPrepareError::ByteMultiplicityOverflow)?;
    }
    Ok(left)
}

pub fn global_padded_rows(raw_rows: usize) -> Result<usize, GlobalPrepareError> {
    let maximum_rows = 1usize << MAX_GLOBAL_LOG_HEIGHT;
    let padded_rows = raw_rows.checked_next_power_of_two().unwrap_or(usize::MAX);
    if raw_rows > maximum_rows || padded_rows > maximum_rows {
        return Err(GlobalPrepareError::HeightExceeded { raw_rows, padded_rows, maximum_rows });
    }
    Ok(padded_rows_threshold(padded_rows))
}

fn cached_program_map<F: PrimeField32>(
    event: &GlobalInteractionEvent,
    source: GlobalSourceId,
    program: &PreparedGlobalProgram,
) -> Option<GlobalSignedMapRowV1<F>> {
    if source != GlobalSourceId::MemoryFinalize
        || !event.is_receive
        || event.kind != InteractionKind::Memory as u8
        || event.message[0] != 0
        || event.message[1] != 0
    {
        return None;
    }
    let addr = event.message[2];
    let word = event.message[3]
        | (event.message[4] << 8)
        | (event.message[5] << 16)
        | (event.message[6] << 24);
    let entry = program.entry(addr).filter(|entry| entry.word == word)?;
    let unsigned = entry.unsigned_point::<F>();
    Some(GlobalSignedMapRowV1 {
        packed_x: unsigned.x,
        signed_y: apply_direction(unsigned.y, true),
        is_receive: true,
        witness: GlobalMapWitnessV1 {
            tweak: entry.tweak,
            canonical_y: entry.canonical_y,
            candidate_rounds: 0,
            zero_top_residue_skips: 0,
        },
    })
}

fn record_byte_delta<F: PrimeField32>(
    event: &GlobalInteractionEvent,
    mapped: &GlobalSignedMapRowV1<F>,
    delta: &mut GlobalByteLookupDelta,
) {
    let canonical_top = mapped.witness.canonical_y[10];
    let w = if event.is_receive { canonical_top - 1 } else { HALF_BASE_MINUS_ONE - canonical_top };
    delta.add_u16_range_check(
        u16::try_from(event.message[0] & 0xffff).expect("low 16-bit mask must fit u16"),
    );
    delta.add_u8_range_check(
        u8::try_from(event.message[0] >> 16).expect("validated message[0] high limb must fit u8"),
        u8::try_from(event.message[5]).expect("validated message[5] must fit u8"),
    );
    delta.add_u8_range_check(
        u8::try_from(event.message[6]).expect("validated message[6] must fit u8"),
        event.kind,
    );
    delta.add_bit_range_check(mapped.witness.tweak, 9);
    let w_low = u16::try_from(w & 0xffff).expect("low 16-bit mask must fit u16");
    let w_high = u16::try_from(w >> 16).expect("validated sign witness high limb must fit u16");
    delta.add_u16_range_check(w_low);
    delta.add_u16_range_check(w_high);
    delta.add_u16_range_check(16_255 - w_high);
}

fn populate_header<F: PrimeField32>(
    row: &mut [F],
    event: &GlobalInteractionEvent,
    mapped: &GlobalSignedMapRowV1<F>,
) {
    let cols: &mut GlobalCols<F> = row.borrow_mut();
    cols.message_rest = core::array::from_fn(|i| F::from_wrapped_u32(event.message[i + 1]));
    cols.x6 = mapped.packed_x.coefficients()[6];
    cols.x5 = mapped.packed_x.coefficients()[5];
    cols.m0_lo16 = F::from_canonical_u32(event.message[0] & 0xffff);
    cols.m0_hi8 = F::from_canonical_u32(event.message[0] >> 16);
    cols.y_lower.copy_from_slice(&mapped.signed_y.coefficients()[..10]);
    let canonical_top = mapped.witness.canonical_y[10];
    let w = if event.is_receive { canonical_top - 1 } else { HALF_BASE_MINUS_ONE - canonical_top };
    cols.w_lo16 = F::from_canonical_u32(w & 0xffff);
    cols.w_hi = F::from_canonical_u32(w >> 16);
    cols.is_receive = F::from_bool(event.is_receive);
    cols.is_real = F::one();
    let h = header(cols);
    cols.quotient.map = map_quotient(&packed_x(cols, &h), &h.signed_y);
}

fn write_point<F: Field>(target: &mut D11PointCols<F>, point: &D11ProjectivePointV1<F>) {
    target.x.copy_from_slice(point.x.coefficients());
    target.y.copy_from_slice(point.y.coefficients());
    target.z.copy_from_slice(point.z.coefficients());
}

fn sparse_affine_from_header<F: Field>(cols: &GlobalCols<F>) -> (D11Sparse7<F>, D11<F>) {
    let h = header(cols);
    let x = D11Sparse7::new([
        h.message[0],
        h.message[1],
        h.message[2],
        h.message[3],
        h.message[4],
        cols.x5,
        cols.x6,
    ]);
    (x, D11::new(h.signed_y))
}

fn populate_chain_row<F: PrimeField32>(
    row: &mut [F],
    index: usize,
    running: &mut D11ProjectivePointV1<F>,
) {
    let cols: &mut GlobalCols<F> = row.borrow_mut();
    let (affine_x, affine_y) = sparse_affine_from_header(cols);
    let input = *running;
    let (cumulative, products) = input.add_mixed_complete_sparse(&affine_x, &affine_y);
    cols.index = F::from_canonical_usize(index);
    write_point(&mut cols.input, &input);
    for (target, product) in [
        (&mut cols.products.u0, &products[0]),
        (&mut cols.products.u1, &products[1]),
        (&mut cols.products.u3, &products[2]),
        (&mut cols.products.u4, &products[3]),
        (&mut cols.products.u5, &products[4]),
    ] {
        target.copy_from_slice(product.coefficients());
    }
    let h = header(cols);
    let input_coefficients = [
        *input.x.coefficients(),
        *input.y.coefficients(),
        *input.z.coefficients(),
    ];
    let mixed = mixed_output_with_quotients(
        &input_coefficients,
        &[packed_x(cols, &h), h.signed_y],
        &cols.products,
    );
    cols.quotient.u0.copy_from_slice(&mixed.product_quotients[0][..6]);
    cols.quotient.u1 = mixed.product_quotients[1];
    cols.quotient.u3 = mixed.product_quotients[2];
    cols.quotient.u4.copy_from_slice(&mixed.product_quotients[3][..6]);
    cols.quotient.u5 = mixed.product_quotients[4];
    cols.quotient.output_x = mixed.output_quotients[0];
    cols.quotient.output_y = mixed.output_quotients[1];
    cols.quotient.output_z = mixed.output_quotients[2];
    write_point(&mut cols.cumulative, &cumulative);
    *running = cumulative;
}

fn padding_row<F: PrimeField32>(raw_rows: usize, terminal: D11ProjectivePointV1<F>) -> Vec<F> {
    let mapped = fixed_padding_dummy::<F>();
    let mut row = vec![F::zero(); NUM_GLOBAL_COLS];
    let cols: &mut GlobalCols<F> = row.as_mut_slice().borrow_mut();
    cols.x6 = mapped.packed_x.coefficients()[6];
    cols.x5 = mapped.packed_x.coefficients()[5];
    cols.y_lower.copy_from_slice(&mapped.signed_y.coefficients()[..10]);
    let w = mapped.signed_y.coefficients()[10].as_canonical_u32();
    cols.w_lo16 = F::from_canonical_u32(w & 0xffff);
    cols.w_hi = F::from_canonical_u32(w >> 16);
    cols.index = F::from_canonical_usize(raw_rows);
    write_point(&mut cols.input, &terminal);
    let sparse_x = D11Sparse7::new(
        mapped.packed_x.coefficients()[..7].try_into().expect("padding X is sparse-seven"),
    );
    let (_, products) = terminal.add_mixed_complete_sparse(&sparse_x, &mapped.signed_y);
    for (target, product) in [
        (&mut cols.products.u0, &products[0]),
        (&mut cols.products.u1, &products[1]),
        (&mut cols.products.u3, &products[2]),
        (&mut cols.products.u4, &products[3]),
        (&mut cols.products.u5, &products[4]),
    ] {
        target.copy_from_slice(product.coefficients());
    }
    let h = header(cols);
    cols.quotient.map = map_quotient(&packed_x(cols, &h), &h.signed_y);
    let input_coefficients = [
        *terminal.x.coefficients(),
        *terminal.y.coefficients(),
        *terminal.z.coefficients(),
    ];
    let mixed = mixed_output_with_quotients(
        &input_coefficients,
        &[packed_x(cols, &h), h.signed_y],
        &cols.products,
    );
    cols.quotient.u0.copy_from_slice(&mixed.product_quotients[0][..6]);
    cols.quotient.u1 = mixed.product_quotients[1];
    cols.quotient.u3 = mixed.product_quotients[2];
    cols.quotient.u4.copy_from_slice(&mixed.product_quotients[3][..6]);
    cols.quotient.u5 = mixed.product_quotients[4];
    cols.quotient.output_x.fill(F::zero());
    cols.quotient.output_y.fill(F::zero());
    cols.quotient.output_z.fill(F::zero());
    write_point(&mut cols.cumulative, &terminal);
    debug_assert!(constraint_residuals(cols).iter().all(Field::is_zero));
    row
}

#[must_use]
pub fn global_tile_count(raw_rows: usize) -> usize {
    raw_rows.div_ceil(512)
}

pub fn global_tile_reducer_real_rows(raw_rows: usize) -> usize {
    let tile_count = global_tile_count(raw_rows);
    if tile_count == 0 {
        return 0;
    }
    16 * tile_count.next_power_of_two() + 6
}

pub fn global_tile_reducer_padded_rows(raw_rows: usize) -> Result<usize, GlobalPrepareError> {
    global_padded_rows(global_tile_reducer_real_rows(raw_rows))
}

fn reducer_row<F: Field>(values: &mut [F], row: usize) -> &mut GlobalTileReducerCols<F> {
    let start = row * NUM_GLOBAL_TILE_REDUCER_COLS;
    values[start..start + NUM_GLOBAL_TILE_REDUCER_COLS].borrow_mut()
}

fn write_d11_group<F: Field>(target: &mut [F; 66], group: usize, value: &D11<F>) {
    target[group * 11..(group + 1) * 11].copy_from_slice(value.coefficients());
}

fn write_reducer_point<F: Field>(target: &mut [F; 66], point: &D11ProjectivePointV1<F>) {
    write_d11_group(target, 0, &point.x);
    write_d11_group(target, 1, &point.y);
    write_d11_group(target, 2, &point.z);
}

fn write_reducer_product<F: Field>(
    cols: &mut GlobalTileReducerCols<F>,
    n: usize,
    p: usize,
    node: usize,
    stage: usize,
    product: usize,
    infinity: bool,
    lhs: D11<F>,
    rhs: D11<F>,
) {
    cols.mode_product = F::one();
    cols.payload.control[0] = F::from_canonical_usize(n);
    cols.payload.control[1] = F::from_canonical_usize(p);
    cols.payload.control[2] = F::from_canonical_usize(node);
    cols.payload.control[3] = F::from_bool(stage == 2);
    cols.payload.control[4] = F::from_canonical_usize(product);
    let last = product + if stage == 2 { 2 } else { 0 } == 5;
    cols.payload.control[5] = F::from_bool(last);
    cols.payload.values[REDUCER_PRODUCT_WAVE_VALUE] = F::from_bool(stage == 1);
    cols.payload.values[REDUCER_PRODUCT_INFINITY_VALUE] = F::from_bool(infinity);
    cols.payload.values[REDUCER_PRODUCT_REBASE_VALUE] = F::from_bool(node == p && stage != 2);
    cols.payload.values[REDUCER_PRODUCT_CONTINUE_VALUE] = F::from_bool(!last);
    cols.payload.values[REDUCER_PRODUCT_TO_MIDDLE_VALUE] = F::from_bool(last && stage == 0);
    cols.payload.values[REDUCER_PRODUCT_TO_NODE_VALUE] = F::from_bool(last && stage == 1);
    cols.payload.values[REDUCER_PRODUCT_TO_ROOT_VALUE] = F::from_bool(last && stage == 2);
    write_d11_group(&mut cols.payload.values, 0, &lhs);
    write_d11_group(&mut cols.payload.values, 1, &rhs);
    let witnessed = mul_with_quotient(lhs.coefficients(), rhs.coefficients());
    let reduced = D11::new(witnessed.reduced);
    write_d11_group(&mut cols.payload.values, 2, &reduced);
    cols.payload.values[33..43].copy_from_slice(&witnessed.quotient);
}

fn complete_first_operands<F: Field>(
    left: &D11ProjectivePointV1<F>,
    right: &D11ProjectivePointV1<F>,
) -> [(D11<F>, D11<F>); 6] {
    [
        (left.x, right.x),
        (left.y, right.y),
        (left.z, right.z),
        (left.x + left.y, right.x + right.y),
        (left.y + left.z, right.y + right.z),
        (left.x + left.z, right.x + right.z),
    ]
}

fn complete_second_operands<F: Field>(first: &[D11<F>; 6]) -> [(D11<F>, D11<F>); 6] {
    let [xx, yy, zz, xy_product, yz_product, xz_product] = *first;
    let xy = xy_product - (xx + yy);
    let yz = yz_product - (yy + zz);
    let xz = xz_product - (xx + zz);
    let bzz3 = (xz - zz.mul_by_z_plus_36()) * F::from_canonical_u32(3);
    let yy_minus = yy - bzz3;
    let yy_plus = yy + bzz3;
    let zz3 = zz * F::from_canonical_u32(3);
    let bxz3 = (xz.mul_by_z_plus_36() - (zz3 + xx)) * F::from_canonical_u32(3);
    let xx3_minus_zz3 = xx * F::from_canonical_u32(3) - zz3;
    [
        (yy_plus, xy),
        (yz, bxz3),
        (yy_plus, yy_minus),
        (xx3_minus_zz3, bxz3),
        (yy_minus, yz),
        (xy, xx3_minus_zz3),
    ]
}

pub(super) fn build_tile_reducer_trace<F: PrimeField32>(
    raw_rows: usize,
    terminals: &[D11ProjectivePointV1<F>],
    start: D11ProjectivePointV1<F>,
) -> Result<(CompressedMatrix<F>, D11ProjectivePointV1<F>), GlobalPrepareError> {
    let k = terminals.len();
    let p = k.next_power_of_two();
    let real_rows = global_tile_reducer_real_rows(raw_rows);
    let padded_rows = global_tile_reducer_padded_rows(raw_rows)?;
    let mut values = vec![F::zero(); real_rows * NUM_GLOBAL_TILE_REDUCER_COLS];
    let identity = D11ProjectivePointV1::<F>::identity();
    let mut heap = vec![identity; 2 * p];
    let mut cursor = 0usize;

    for ordinal in 0..p {
        let point = terminals.get(ordinal).copied().unwrap_or(identity);
        heap[p + ordinal] = point;
        let cols = reducer_row(&mut values, cursor);
        cols.mode_leaf = F::one();
        cols.payload.control[0] = F::from_canonical_usize(raw_rows);
        cols.payload.control[1] = F::from_canonical_usize(p);
        cols.payload.control[2] = F::from_canonical_usize(ordinal);
        cols.payload.control[3] = F::from_canonical_usize(((ordinal + 1) * 512).min(raw_rows));
        cols.payload.control[4] = F::from_bool(ordinal < k);
        cols.payload.control[5] = F::from_bool(ordinal + 1 == k);
        let leaf_real = usize::from(ordinal < k);
        let leaf_last = usize::from(ordinal + 1 == k);
        cols.control_rank = F::from_canonical_usize(2 * ordinal + leaf_real);
        cols.control_next_rank =
            F::from_canonical_usize(2 * (ordinal + 1) + leaf_real - leaf_last);
        write_reducer_point(&mut cols.payload.values, &point);
        cols.payload.values[REDUCER_LEAF_K_VALUE] = F::from_canonical_usize(k);
        let leaf_end = ordinal + 1 == p;
        cols.payload.values[REDUCER_LEAF_END_VALUE] = F::from_bool(leaf_end);
        if leaf_end {
            let p_bit = p.trailing_zeros() as usize;
            debug_assert!(p_bit < REDUCER_LEAF_P_BITS);
            cols.payload.values[REDUCER_LEAF_P_BITS_START + p_bit] = F::one();
            let gap = k - (p / 2 + 1);
            for bit in 0..REDUCER_LEAF_GAP_BITS {
                cols.payload.values[REDUCER_LEAF_GAP_BITS_START + bit] =
                    F::from_bool((gap >> bit) & 1 == 1);
            }
            cols.control_next_tag = F::from_canonical_u32(if p == 1 { 8 } else { 4 });
        }
        cursor += 1;
    }

    for node in (1..p).rev() {
        let left = heap[2 * node];
        let right = heap[2 * node + 1];
        let (output, first, second) = left.add_complete_with_products(&right);
        heap[node] = output;
        let cols = reducer_row(&mut values, cursor);
        cols.mode_node_input = F::one();
        cols.payload.control[0] = F::from_canonical_usize(raw_rows);
        cols.payload.control[1] = F::from_canonical_usize(p);
        cols.payload.control[2] = F::from_canonical_usize(node);
        cols.control_rank = F::from_canonical_usize(2 * (2 * p - 1 - node));
        write_reducer_point(&mut cols.payload.values, &left);
        write_d11_group(&mut cols.payload.values, 3, &right.x);
        write_d11_group(&mut cols.payload.values, 4, &right.y);
        write_d11_group(&mut cols.payload.values, 5, &right.z);
        cursor += 1;
        let first_operands = complete_first_operands(&left, &right);
        for product in 0..6 {
            write_reducer_product(
                reducer_row(&mut values, cursor),
                raw_rows,
                p,
                node,
                0,
                product,
                false,
                first_operands[product].0,
                first_operands[product].1,
            );
            cursor += 1;
        }
        let cols = reducer_row(&mut values, cursor);
        cols.mode_node_middle = F::one();
        cols.payload.control[0] = F::from_canonical_usize(raw_rows);
        cols.payload.control[1] = F::from_canonical_usize(p);
        cols.payload.control[2] = F::from_canonical_usize(node);
        for (group, value) in first.iter().enumerate() {
            write_d11_group(&mut cols.payload.values, group, value);
        }
        cursor += 1;
        let second_operands = complete_second_operands(&first);
        for product in 0..6 {
            write_reducer_product(
                reducer_row(&mut values, cursor),
                raw_rows,
                p,
                node,
                1,
                product,
                false,
                second_operands[product].0,
                second_operands[product].1,
            );
            cursor += 1;
        }
        let cols = reducer_row(&mut values, cursor);
        cols.mode_node_output = F::one();
        cols.payload.control[0] = F::from_canonical_usize(raw_rows);
        cols.payload.control[1] = F::from_canonical_usize(p);
        cols.payload.control[2] = F::from_canonical_usize(node);
        cols.payload.control[4] = F::from_bool(node == 1);
        cols.control_next_rank = F::from_canonical_usize(2 * (2 * p - node));
        cols.control_next_tag = F::from_canonical_u32(if node == 1 { 8 } else { 4 });
        for (group, value) in second.iter().enumerate() {
            write_d11_group(&mut cols.payload.values, group, value);
        }
        cursor += 1;
    }

    let raw = heap[1];
    let (rebased, first, second) = start.add_complete_with_products(&raw);
    let cols = reducer_row(&mut values, cursor);
    cols.selector_spare = F::one();
    cols.payload.control[0] = F::from_canonical_usize(raw_rows);
    cols.payload.control[1] = F::from_canonical_usize(p);
    cols.payload.control[2] = F::from_canonical_usize(p);
    cols.control_rank = F::from_canonical_usize(4 * p - 2);
    write_reducer_point(&mut cols.payload.values, &start);
    write_d11_group(&mut cols.payload.values, 3, &raw.x);
    write_d11_group(&mut cols.payload.values, 4, &raw.y);
    write_d11_group(&mut cols.payload.values, 5, &raw.z);
    cursor += 1;
    let first_operands = complete_first_operands(&start, &raw);
    for product in 0..6 {
        write_reducer_product(
            reducer_row(&mut values, cursor),
            raw_rows,
            p,
            p,
            0,
            product,
            false,
            first_operands[product].0,
            first_operands[product].1,
        );
        cursor += 1;
    }
    let cols = reducer_row(&mut values, cursor);
    cols.mode_node_middle = F::one();
    cols.payload.control[0] = F::from_canonical_usize(raw_rows);
    cols.payload.control[1] = F::from_canonical_usize(p);
    cols.payload.control[2] = F::from_canonical_usize(p);
    cols.payload.control[3] = F::one();
    for (group, value) in first.iter().enumerate() {
        write_d11_group(&mut cols.payload.values, group, value);
    }
    cursor += 1;
    let second_operands = complete_second_operands(&first);
    for product in 0..6 {
        write_reducer_product(
            reducer_row(&mut values, cursor),
            raw_rows,
            p,
            p,
            1,
            product,
            false,
            second_operands[product].0,
            second_operands[product].1,
        );
        cursor += 1;
    }
    let cols = reducer_row(&mut values, cursor);
    cols.mode_node_output = F::one();
    cols.payload.control[0] = F::from_canonical_usize(raw_rows);
    cols.payload.control[1] = F::from_canonical_usize(p);
    cols.payload.control[2] = F::from_canonical_usize(p);
    cols.payload.control[3] = F::one();
    cols.control_next_rank = F::from_canonical_usize(4 * p);
    cols.control_next_tag = F::from_canonical_u32(12);
    for (group, value) in second.iter().enumerate() {
        write_d11_group(&mut cols.payload.values, group, value);
    }
    cursor += 1;

    let raw = rebased;
    let infinity = raw.z.is_zero();
    let lambda = if infinity { raw.y } else { raw.z };
    let lambda_inv = lambda.inverse();
    let canonical = raw.rescaled(lambda_inv);
    let cols = reducer_row(&mut values, cursor);
    cols.mode_root_input = F::one();
    cols.payload.control[0] = F::from_canonical_usize(raw_rows);
    cols.payload.control[1] = F::from_canonical_usize(p);
    cols.payload.control[2] = F::from_canonical_usize(p);
    cols.payload.control[3] = F::from_bool(infinity);
    cols.control_rank = F::from_canonical_usize(4 * p);
    cols.control_next_rank = F::from_canonical_usize(4 * p + 2);
    cols.control_next_tag = F::from_canonical_u32(16);
    write_reducer_point(&mut cols.payload.values, &raw);
    write_d11_group(&mut cols.payload.values, 3, &lambda);
    write_d11_group(&mut cols.payload.values, 4, &lambda_inv);
    cursor += 1;
    for (product, lhs) in [lambda, raw.x, raw.y, raw.z].into_iter().enumerate() {
        write_reducer_product(
            reducer_row(&mut values, cursor),
            raw_rows,
            p,
            p,
            2,
            product,
            infinity,
            lhs,
            lambda_inv,
        );
        cursor += 1;
    }
    let cols = reducer_row(&mut values, cursor);
    cols.mode_root_output = F::one();
    cols.payload.control[0] = F::from_canonical_usize(raw_rows);
    cols.payload.control[1] = F::from_canonical_usize(p);
    cols.payload.control[2] = F::from_canonical_usize(p);
    cols.payload.control[3] = F::from_bool(infinity);
    cols.control_rank = F::from_canonical_usize(4 * p + 2);
    cols.control_next_rank = F::one();
    for (group, value) in [lambda * lambda_inv, canonical.x, canonical.y, canonical.z]
        .into_iter()
        .enumerate()
    {
        write_d11_group(&mut cols.payload.values, group, &value);
    }
    cursor += 1;
    debug_assert_eq!(cursor, real_rows);
    Ok((
        CompressedMatrix::new(
            RowMajorMatrix::new(values, NUM_GLOBAL_TILE_REDUCER_COLS),
            PaddingRow::Zero { width: NUM_GLOBAL_TILE_REDUCER_COLS },
            padded_rows,
        ),
        canonical,
    ))
}

struct SourceTask<'a, F> {
    source: GlobalSourceId,
    batch: crate::global::sources::GlobalProducerBatch,
    range: Range<usize>,
    rows: &'a mut [F],
}

#[derive(Default)]
struct TaskAccumulator {
    rows: usize,
    byte_delta: GlobalByteLookupDelta,
}

fn merge_task_accumulators(
    left: TaskAccumulator,
    right: TaskAccumulator,
) -> Result<TaskAccumulator, GlobalPrepareError> {
    Ok(TaskAccumulator {
        rows: left.rows.checked_add(right.rows).ok_or(GlobalPrepareError::SourceCountOverflow)?,
        byte_delta: merge_delta(left.byte_delta, right.byte_delta)?,
    })
}

/// Builds the canonical Global trace without materializing a normalized endpoint list
/// or any full-height point/quotient side buffer.
pub fn prepare_global_trace(
    input: &ExecutionRecord,
) -> Result<PreparedGlobalTrace<KoalaBear>, GlobalPrepareError> {
    prepare_global_trace_for_field(input)
}

pub(crate) fn prepare_global_trace_for_field<F: PrimeField32>(
    input: &ExecutionRecord,
) -> Result<PreparedGlobalTrace<F>, GlobalPrepareError> {
    let prepared_program = input.program.prepared_global_program()?;
    let start = program_global_seed(prepared_program.initial_boundary())?;
    prepare_global_trace_with_options(input, PreparationOptions::default(), start)
}

pub(crate) fn prepare_global_trace_for_field_from_start<F: PrimeField32>(
    input: &ExecutionRecord,
    start: D11ProjectivePointV1<F>,
) -> Result<PreparedGlobalTrace<F>, GlobalPrepareError> {
    prepare_global_trace_with_options(input, PreparationOptions::default(), start)
}

fn prepare_global_trace_with_options<F: PrimeField32>(
    input: &ExecutionRecord,
    options: PreparationOptions,
    start: D11ProjectivePointV1<F>,
) -> Result<PreparedGlobalTrace<F>, GlobalPrepareError> {
    let local_memory_events = input.get_local_mem_events().collect::<Vec<_>>();
    let plan =
        GlobalSourcePlan::build(input, options.memory_chunk_rows, local_memory_events.len())?;
    let raw_rows = plan.raw_rows;
    let padded_rows = global_padded_rows(raw_rows)?;
    let prepared_program = input.program.prepared_global_program()?;
    let value_count = raw_rows
        .checked_mul(NUM_GLOBAL_COLS)
        .ok_or(GlobalPrepareError::TraceElementCountOverflow)?;
    value_count
        .checked_add(NUM_GLOBAL_COLS)
        .and_then(|elements| elements.checked_mul(core::mem::size_of::<F>()))
        .ok_or(GlobalPrepareError::TraceByteCountOverflow)?;
    padded_rows
        .checked_mul(NUM_GLOBAL_COLS)
        .and_then(|elements| elements.checked_mul(core::mem::size_of::<F>()))
        .ok_or(GlobalPrepareError::TraceByteCountOverflow)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(value_count)
        .map_err(|_| GlobalPrepareError::TraceAllocationFailed { elements: value_count })?;
    values.resize(value_count, F::zero());

    let mut tail = values.as_mut_slice();
    let mut tasks = Vec::with_capacity(plan.chunks.len());
    for chunk in plan.chunks {
        let row_count = chunk
            .source_range
            .end
            .checked_sub(chunk.source_range.start)
            .ok_or(GlobalPrepareError::SourceTaskRangeOverflow)?;
        let element_count = row_count
            .checked_mul(NUM_GLOBAL_COLS)
            .ok_or(GlobalPrepareError::SourceTaskRangeOverflow)?;
        if element_count > tail.len() {
            return Err(GlobalPrepareError::SourceTaskRangeOverflow);
        }
        let (rows, next) = tail.split_at_mut(element_count);
        tasks.push(SourceTask {
            source: chunk.source,
            batch: chunk.batch,
            range: chunk.source_range,
            rows,
        });
        tail = next;
    }
    if !tail.is_empty() {
        return Err(GlobalPrepareError::SourceTaskRangeOverflow);
    }
    let accumulated = tasks
        .into_par_iter()
        .try_fold(TaskAccumulator::default, |mut accumulated, task| {
            let mut rows_written = 0usize;
            let range_start = task.range.start;
            let mut failure = None;
            task.batch.visit_endpoint_range_indexed(
                input,
                &local_memory_events,
                task.range,
                |event| {
                    if failure.is_some() {
                        return;
                    }
                    let expected_rows = task.rows.len() / NUM_GLOBAL_COLS;
                    if rows_written >= expected_rows {
                        failure = Some(GlobalPrepareError::SourceTaskRowCountMismatch {
                            expected: expected_rows,
                            actual: rows_written + 1,
                        });
                        return;
                    }
                    let mapped = if let Some(mapped) =
                        cached_program_map(&event, task.source, &prepared_program)
                    {
                        Ok(mapped)
                    } else {
                        construct_map::<F>(
                            GlobalPackInputV1 { message: event.message, kind: event.kind },
                            event.is_receive,
                        )
                    };
                    let mapped = match mapped {
                        Ok(mapped) => mapped,
                        Err(cause) => {
                            failure = Some(GlobalPrepareError::Map {
                                source: task.source,
                                source_ordinal: range_start + rows_written,
                                endpoint: event,
                                cause,
                            });
                            return;
                        }
                    };
                    let start = rows_written * NUM_GLOBAL_COLS;
                    populate_header(
                        &mut task.rows[start..start + NUM_GLOBAL_COLS],
                        &event,
                        &mapped,
                    );
                    record_byte_delta(&event, &mapped, &mut accumulated.byte_delta);
                    rows_written += 1;
                },
            );
            if let Some(failure) = failure {
                return Err(failure);
            }
            let expected_rows = task.rows.len() / NUM_GLOBAL_COLS;
            if rows_written != expected_rows {
                return Err(GlobalPrepareError::SourceTaskRowCountMismatch {
                    expected: expected_rows,
                    actual: rows_written,
                });
            }
            accumulated.rows = accumulated
                .rows
                .checked_add(rows_written)
                .ok_or(GlobalPrepareError::SourceCountOverflow)?;
            Ok(accumulated)
        })
        .try_reduce(TaskAccumulator::default, merge_task_accumulators)?;
    if accumulated.rows != raw_rows {
        return Err(GlobalPrepareError::SourceTaskRowCountMismatch {
            expected: raw_rows,
            actual: accumulated.rows,
        });
    }

    let identity = D11ProjectivePointV1::<F>::identity();
    let mut running = identity;
    let tile_count = global_tile_count(raw_rows);
    let mut terminals = Vec::with_capacity(tile_count);
    for (index, row) in values.chunks_exact_mut(NUM_GLOBAL_COLS).enumerate() {
        if index % 512 == 0 {
            running = identity;
        }
        populate_chain_row(row, index, &mut running);
        let boundary = index + 1;
        if boundary % 512 == 0 || boundary == raw_rows {
            terminals.push(running);
        }
    }

    let padding = padding_row(raw_rows, running);
    let trace = CompressedMatrix::new(
        RowMajorMatrix::new(values, NUM_GLOBAL_COLS),
        PaddingRow::General(padding),
        padded_rows,
    );
    let (reducer_trace, terminal) = build_tile_reducer_trace(raw_rows, &terminals, start)?;
    let log_height = u8::try_from(padded_rows.trailing_zeros()).map_err(|_| {
        GlobalPrepareError::HeightExceeded {
            raw_rows,
            padded_rows,
            maximum_rows: 1usize << MAX_GLOBAL_LOG_HEIGHT,
        }
    })?;
    let claim = GlobalClaim {
        has_global_opening: F::one(),
        count: F::from_canonical_usize(raw_rows),
        interval: GlobalStateInterval {
            start: state_from_point(&start),
            end: state_from_point(&terminal),
        },
    };
    Ok(PreparedGlobalTrace {
        trace,
        reducer_trace,
        byte_delta: accumulated.byte_delta,
        raw_rows,
        start,
        terminal,
        log_height,
        claim,
    })
}

/// Evaluate the exact shared 192-residual relation for one canonical row.
#[cfg(feature = "test-utils")]
#[must_use]
pub fn global_relation_accepts_for_test(row: &[KoalaBear]) -> bool {
    assert_eq!(row.len(), NUM_GLOBAL_COLS);
    // SAFETY: the length check and compile-time layout assertions establish the exact repr(C)
    // extent and field order.
    let cols = unsafe { &*row.as_ptr().cast::<GlobalCols<KoalaBear>>() };
    constraint_residuals(cols).iter().all(Field::is_zero)
}
