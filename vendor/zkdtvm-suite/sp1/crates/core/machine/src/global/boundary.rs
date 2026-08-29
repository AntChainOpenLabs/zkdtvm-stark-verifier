use dt_stark::{
    air::{GlobalClaim, GlobalState, GlobalStateInterval},
    global_d11::{StableChipId, CORE_GLOBAL_OWNER},
};
use p3_field::Field;
use p3_matrix::Matrix;

use super::{
    columns::{
        GlobalTileReducerCols, GLOBAL_TILE_REDUCER_COL_MAP, NUM_GLOBAL_TILE_REDUCER_COLS,
    },
    writer::{PreparedGlobalTrace, RetainedGlobalTrace},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GlobalBoundaryLayout {
    pub owner: StableChipId,
    pub count: usize,
    pub canonical_x: [usize; 11],
    pub canonical_y: [usize; 11],
    pub canonical_z: [usize; 11],
}

pub const GLOBAL_BOUNDARY_LAYOUT: GlobalBoundaryLayout = GlobalBoundaryLayout {
    owner: CORE_GLOBAL_OWNER,
    count: GLOBAL_TILE_REDUCER_COL_MAP.payload.control[0],
    canonical_x: [
        GLOBAL_TILE_REDUCER_COL_MAP.payload.values[11],
        GLOBAL_TILE_REDUCER_COL_MAP.payload.values[12],
        GLOBAL_TILE_REDUCER_COL_MAP.payload.values[13],
        GLOBAL_TILE_REDUCER_COL_MAP.payload.values[14],
        GLOBAL_TILE_REDUCER_COL_MAP.payload.values[15],
        GLOBAL_TILE_REDUCER_COL_MAP.payload.values[16],
        GLOBAL_TILE_REDUCER_COL_MAP.payload.values[17],
        GLOBAL_TILE_REDUCER_COL_MAP.payload.values[18],
        GLOBAL_TILE_REDUCER_COL_MAP.payload.values[19],
        GLOBAL_TILE_REDUCER_COL_MAP.payload.values[20],
        GLOBAL_TILE_REDUCER_COL_MAP.payload.values[21],
    ],
    canonical_y: [
        GLOBAL_TILE_REDUCER_COL_MAP.payload.values[22],
        GLOBAL_TILE_REDUCER_COL_MAP.payload.values[23],
        GLOBAL_TILE_REDUCER_COL_MAP.payload.values[24],
        GLOBAL_TILE_REDUCER_COL_MAP.payload.values[25],
        GLOBAL_TILE_REDUCER_COL_MAP.payload.values[26],
        GLOBAL_TILE_REDUCER_COL_MAP.payload.values[27],
        GLOBAL_TILE_REDUCER_COL_MAP.payload.values[28],
        GLOBAL_TILE_REDUCER_COL_MAP.payload.values[29],
        GLOBAL_TILE_REDUCER_COL_MAP.payload.values[30],
        GLOBAL_TILE_REDUCER_COL_MAP.payload.values[31],
        GLOBAL_TILE_REDUCER_COL_MAP.payload.values[32],
    ],
    canonical_z: [
        GLOBAL_TILE_REDUCER_COL_MAP.payload.values[33],
        GLOBAL_TILE_REDUCER_COL_MAP.payload.values[34],
        GLOBAL_TILE_REDUCER_COL_MAP.payload.values[35],
        GLOBAL_TILE_REDUCER_COL_MAP.payload.values[36],
        GLOBAL_TILE_REDUCER_COL_MAP.payload.values[37],
        GLOBAL_TILE_REDUCER_COL_MAP.payload.values[38],
        GLOBAL_TILE_REDUCER_COL_MAP.payload.values[39],
        GLOBAL_TILE_REDUCER_COL_MAP.payload.values[40],
        GLOBAL_TILE_REDUCER_COL_MAP.payload.values[41],
        GLOBAL_TILE_REDUCER_COL_MAP.payload.values[42],
        GLOBAL_TILE_REDUCER_COL_MAP.payload.values[43],
    ],
};

const GLOBAL_REBASE_START_X: [usize; 11] = [
    GLOBAL_TILE_REDUCER_COL_MAP.payload.values[0],
    GLOBAL_TILE_REDUCER_COL_MAP.payload.values[1],
    GLOBAL_TILE_REDUCER_COL_MAP.payload.values[2],
    GLOBAL_TILE_REDUCER_COL_MAP.payload.values[3],
    GLOBAL_TILE_REDUCER_COL_MAP.payload.values[4],
    GLOBAL_TILE_REDUCER_COL_MAP.payload.values[5],
    GLOBAL_TILE_REDUCER_COL_MAP.payload.values[6],
    GLOBAL_TILE_REDUCER_COL_MAP.payload.values[7],
    GLOBAL_TILE_REDUCER_COL_MAP.payload.values[8],
    GLOBAL_TILE_REDUCER_COL_MAP.payload.values[9],
    GLOBAL_TILE_REDUCER_COL_MAP.payload.values[10],
];
const GLOBAL_REBASE_START_Y: [usize; 11] = [
    GLOBAL_TILE_REDUCER_COL_MAP.payload.values[11],
    GLOBAL_TILE_REDUCER_COL_MAP.payload.values[12],
    GLOBAL_TILE_REDUCER_COL_MAP.payload.values[13],
    GLOBAL_TILE_REDUCER_COL_MAP.payload.values[14],
    GLOBAL_TILE_REDUCER_COL_MAP.payload.values[15],
    GLOBAL_TILE_REDUCER_COL_MAP.payload.values[16],
    GLOBAL_TILE_REDUCER_COL_MAP.payload.values[17],
    GLOBAL_TILE_REDUCER_COL_MAP.payload.values[18],
    GLOBAL_TILE_REDUCER_COL_MAP.payload.values[19],
    GLOBAL_TILE_REDUCER_COL_MAP.payload.values[20],
    GLOBAL_TILE_REDUCER_COL_MAP.payload.values[21],
];
const GLOBAL_REBASE_START_Z: [usize; 11] = [
    GLOBAL_TILE_REDUCER_COL_MAP.payload.values[22],
    GLOBAL_TILE_REDUCER_COL_MAP.payload.values[23],
    GLOBAL_TILE_REDUCER_COL_MAP.payload.values[24],
    GLOBAL_TILE_REDUCER_COL_MAP.payload.values[25],
    GLOBAL_TILE_REDUCER_COL_MAP.payload.values[26],
    GLOBAL_TILE_REDUCER_COL_MAP.payload.values[27],
    GLOBAL_TILE_REDUCER_COL_MAP.payload.values[28],
    GLOBAL_TILE_REDUCER_COL_MAP.payload.values[29],
    GLOBAL_TILE_REDUCER_COL_MAP.payload.values[30],
    GLOBAL_TILE_REDUCER_COL_MAP.payload.values[31],
    GLOBAL_TILE_REDUCER_COL_MAP.payload.values[32],
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GlobalBoundaryLayoutError {
    InvalidRowWidth { expected: usize, actual: usize },
    CountOverflow(usize),
}

fn identity_state<F: Field>() -> GlobalState<F> {
    let mut y = [F::zero(); 11];
    y[0] = F::one();
    GlobalState { x: [F::zero(); 11], y, z: [F::zero(); 11] }
}

fn prepared_claim<F: Field>(
    trace: &[F],
    reducer_rows: usize,
) -> Result<Option<GlobalClaim<F>>, GlobalBoundaryLayoutError> {
    if reducer_rows == 0 {
        return Ok(None);
    }
    let p = ((reducer_rows - 6) / 16).max(1);
    let rebase_row = 16 * p - 15;
    let rebase_start = rebase_row
        .checked_mul(NUM_GLOBAL_TILE_REDUCER_COLS)
        .ok_or(GlobalBoundaryLayoutError::CountOverflow(reducer_rows))?;
    let rebase = trace.get(rebase_start..rebase_start + NUM_GLOBAL_TILE_REDUCER_COLS).ok_or(
        GlobalBoundaryLayoutError::InvalidRowWidth {
            expected: rebase_start + NUM_GLOBAL_TILE_REDUCER_COLS,
            actual: trace.len(),
        },
    )?;
    let start = (reducer_rows - 1)
        .checked_mul(NUM_GLOBAL_TILE_REDUCER_COLS)
        .ok_or(GlobalBoundaryLayoutError::CountOverflow(reducer_rows))?;
    let end = start
        .checked_add(NUM_GLOBAL_TILE_REDUCER_COLS)
        .ok_or(GlobalBoundaryLayoutError::CountOverflow(reducer_rows))?;
    let last = trace
        .get(start..end)
        .ok_or(GlobalBoundaryLayoutError::InvalidRowWidth { expected: end, actual: trace.len() })?;
    Ok(Some(GlobalClaim {
        has_global_opening: F::one(),
        count: last[GLOBAL_BOUNDARY_LAYOUT.count],
        interval: GlobalStateInterval {
            start: GlobalState {
                x: GLOBAL_REBASE_START_X.map(|column| rebase[column]),
                y: GLOBAL_REBASE_START_Y.map(|column| rebase[column]),
                z: GLOBAL_REBASE_START_Z.map(|column| rebase[column]),
            },
            end: GlobalState {
                x: GLOBAL_BOUNDARY_LAYOUT.canonical_x.map(|column| last[column]),
                y: GLOBAL_BOUNDARY_LAYOUT.canonical_y.map(|column| last[column]),
                z: GLOBAL_BOUNDARY_LAYOUT.canonical_z.map(|column| last[column]),
            },
        },
    }))
}

pub fn claim_from_compressed_tile_reducer_trace<F: Field>(
    trace: &dt_stark::sumcheck::trace::CompressedMatrix<F>,
) -> Result<Option<GlobalClaim<F>>, GlobalBoundaryLayoutError> {
    prepared_claim(&trace.main.values, trace.main.height())
}

impl<F: Field> PreparedGlobalTrace<F> {
    pub fn extracted_claim(&self) -> Result<Option<GlobalClaim<F>>, GlobalBoundaryLayoutError> {
        claim_from_compressed_tile_reducer_trace(&self.reducer_trace)
    }
}

impl<F: Field> RetainedGlobalTrace<F> {
    pub fn extracted_claim(&self) -> Result<Option<GlobalClaim<F>>, GlobalBoundaryLayoutError> {
        claim_from_compressed_tile_reducer_trace(&self.reducer_trace)
    }
}

const _: () = {
    assert!(GLOBAL_BOUNDARY_LAYOUT.owner.0 == 43);
    assert!(GLOBAL_BOUNDARY_LAYOUT.count == 8);
    assert!(GLOBAL_BOUNDARY_LAYOUT.canonical_x[0] == 25);
    assert!(GLOBAL_BOUNDARY_LAYOUT.canonical_z[10] == 57);
    assert!(core::mem::size_of::<GlobalTileReducerCols<u8>>() == NUM_GLOBAL_TILE_REDUCER_COLS);
};
