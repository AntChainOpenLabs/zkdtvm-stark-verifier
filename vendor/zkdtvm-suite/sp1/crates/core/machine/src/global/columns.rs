use core::mem::transmute;

use dt_derive::AlignedBorrow;

use crate::utils::indices_arr;

/// A simple-projective D11 point in `(X,Y,Z)` order.
#[derive(AlignedBorrow, Clone, Debug)]
#[repr(C)]
pub struct D11PointCols<T: Clone> {
    pub x: [T; 11],
    pub y: [T; 11],
    pub z: [T; 11],
}

/// The five materialized mixed-RCB products in protocol order.
#[derive(AlignedBorrow, Clone, Debug)]
#[repr(C)]
pub struct D11ProductCols<T: Clone> {
    pub u0: [T; 11],
    pub u1: [T; 11],
    pub u3: [T; 11],
    pub u4: [T; 11],
    pub u5: [T; 11],
}

/// Quotients for the nine committed D11 polynomial identities.
#[derive(AlignedBorrow, Clone, Debug)]
#[repr(C)]
pub struct D11QuotientCols<T: Clone> {
    pub map: [T; 10],
    pub u0: [T; 6],
    pub u1: [T; 10],
    pub u3: [T; 10],
    pub u4: [T; 6],
    pub u5: [T; 10],
    pub output_x: [T; 10],
    pub output_y: [T; 10],
    pub output_z: [T; 10],
}

/// Exact `Projective228QIntervalV4` main-row layout.
#[derive(AlignedBorrow, Clone, Debug)]
#[repr(C)]
pub struct GlobalCols<T: Clone> {
    pub message_rest: [T; 6],
    pub x6: T,
    pub x5: T,
    pub m0_lo16: T,
    pub m0_hi8: T,
    pub y_lower: [T; 10],
    pub w_lo16: T,
    pub w_hi: T,
    pub is_receive: T,
    pub is_real: T,
    pub index: T,
    pub input: D11PointCols<T>,
    pub products: D11ProductCols<T>,
    pub cumulative: D11PointCols<T>,
    pub quotient: D11QuotientCols<T>,
}

pub const NUM_GLOBAL_COLS: usize = size_of::<GlobalCols<u8>>();

/// Fixed seven-mode payload overlay for `GlobalTileReducerV3`.
///
/// Every mode gives the same 66 committed value columns a typed meaning.  The
/// control words remain disjoint, so no Rust union or raw-storage lifetime is
/// involved in the overlay.
#[derive(AlignedBorrow, Clone, Debug)]
#[repr(C)]
pub struct GlobalTileReducerPayloadCols<T: Clone> {
    pub control: [T; 6],
    pub values: [T; 66],
}

/// Exact 83-column `GlobalTileReducerV3` row layout.
#[derive(AlignedBorrow, Clone, Debug)]
#[repr(C)]
pub struct GlobalTileReducerCols<T: Clone> {
    pub mode_leaf: T,
    pub mode_node_input: T,
    pub mode_product: T,
    pub mode_node_middle: T,
    pub mode_node_output: T,
    pub mode_root_input: T,
    pub mode_root_output: T,
    pub selector_spare: T,
    pub payload: GlobalTileReducerPayloadCols<T>,
    pub control_rank: T,
    pub control_next_rank: T,
    pub control_next_tag: T,
}

pub const NUM_GLOBAL_TILE_REDUCER_COLS: usize = size_of::<GlobalTileReducerCols<u8>>();

pub(crate) const REDUCER_LEAF_K_VALUE: usize = 33;
pub(crate) const REDUCER_LEAF_END_VALUE: usize = 34;
pub(crate) const REDUCER_LEAF_P_BITS_START: usize = 35;
pub(crate) const REDUCER_LEAF_P_BITS: usize = 14;
pub(crate) const REDUCER_LEAF_GAP_BITS_START: usize = 49;
pub(crate) const REDUCER_LEAF_GAP_BITS: usize = 13;
pub(crate) const REDUCER_PRODUCT_WAVE_VALUE: usize = 43;
pub(crate) const REDUCER_PRODUCT_INFINITY_VALUE: usize = 44;
pub(crate) const REDUCER_PRODUCT_REBASE_VALUE: usize = 45;
pub(crate) const REDUCER_PRODUCT_CONTINUE_VALUE: usize = 62;
pub(crate) const REDUCER_PRODUCT_TO_MIDDLE_VALUE: usize = 63;
pub(crate) const REDUCER_PRODUCT_TO_NODE_VALUE: usize = 64;
pub(crate) const REDUCER_PRODUCT_TO_ROOT_VALUE: usize = 65;

const fn make_col_map() -> GlobalCols<usize> {
    let indices = indices_arr::<NUM_GLOBAL_COLS>();
    // SAFETY: every field is `usize` or an array/nested repr(C) aggregate of
    // `usize`; the byte-sized mirror proves the exact element count.
    unsafe { transmute::<[usize; NUM_GLOBAL_COLS], GlobalCols<usize>>(indices) }
}

pub const GLOBAL_COL_MAP: GlobalCols<usize> = make_col_map();

const fn make_tile_reducer_col_map() -> GlobalTileReducerCols<usize> {
    let indices = indices_arr::<NUM_GLOBAL_TILE_REDUCER_COLS>();
    // SAFETY: the same repr(C), byte-width argument as `make_col_map` applies.
    unsafe {
        transmute::<[usize; NUM_GLOBAL_TILE_REDUCER_COLS], GlobalTileReducerCols<usize>>(indices)
    }
}

pub const GLOBAL_TILE_REDUCER_COL_MAP: GlobalTileReducerCols<usize> =
    make_tile_reducer_col_map();

/// One contiguous named range in the frozen row layout.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GlobalLayoutFieldDescriptor {
    pub name: &'static str,
    pub offset: usize,
    pub len: usize,
}

/// Complete, ordered partition of all 228 canonical Global main columns.
pub const GLOBAL_LAYOUT_FIELDS: [GlobalLayoutFieldDescriptor; 31] = [
    GlobalLayoutFieldDescriptor {
        name: "message_rest",
        offset: GLOBAL_COL_MAP.message_rest[0],
        len: 6,
    },
    GlobalLayoutFieldDescriptor { name: "x6", offset: GLOBAL_COL_MAP.x6, len: 1 },
    GlobalLayoutFieldDescriptor { name: "x5", offset: GLOBAL_COL_MAP.x5, len: 1 },
    GlobalLayoutFieldDescriptor { name: "m0_lo16", offset: GLOBAL_COL_MAP.m0_lo16, len: 1 },
    GlobalLayoutFieldDescriptor { name: "m0_hi8", offset: GLOBAL_COL_MAP.m0_hi8, len: 1 },
    GlobalLayoutFieldDescriptor { name: "y_lower", offset: GLOBAL_COL_MAP.y_lower[0], len: 10 },
    GlobalLayoutFieldDescriptor { name: "w_lo16", offset: GLOBAL_COL_MAP.w_lo16, len: 1 },
    GlobalLayoutFieldDescriptor { name: "w_hi", offset: GLOBAL_COL_MAP.w_hi, len: 1 },
    GlobalLayoutFieldDescriptor { name: "is_receive", offset: GLOBAL_COL_MAP.is_receive, len: 1 },
    GlobalLayoutFieldDescriptor { name: "is_real", offset: GLOBAL_COL_MAP.is_real, len: 1 },
    GlobalLayoutFieldDescriptor { name: "index", offset: GLOBAL_COL_MAP.index, len: 1 },
    GlobalLayoutFieldDescriptor { name: "input.x", offset: GLOBAL_COL_MAP.input.x[0], len: 11 },
    GlobalLayoutFieldDescriptor { name: "input.y", offset: GLOBAL_COL_MAP.input.y[0], len: 11 },
    GlobalLayoutFieldDescriptor { name: "input.z", offset: GLOBAL_COL_MAP.input.z[0], len: 11 },
    GlobalLayoutFieldDescriptor {
        name: "products.u0",
        offset: GLOBAL_COL_MAP.products.u0[0],
        len: 11,
    },
    GlobalLayoutFieldDescriptor {
        name: "products.u1",
        offset: GLOBAL_COL_MAP.products.u1[0],
        len: 11,
    },
    GlobalLayoutFieldDescriptor {
        name: "products.u3",
        offset: GLOBAL_COL_MAP.products.u3[0],
        len: 11,
    },
    GlobalLayoutFieldDescriptor {
        name: "products.u4",
        offset: GLOBAL_COL_MAP.products.u4[0],
        len: 11,
    },
    GlobalLayoutFieldDescriptor {
        name: "products.u5",
        offset: GLOBAL_COL_MAP.products.u5[0],
        len: 11,
    },
    GlobalLayoutFieldDescriptor {
        name: "cumulative.x",
        offset: GLOBAL_COL_MAP.cumulative.x[0],
        len: 11,
    },
    GlobalLayoutFieldDescriptor {
        name: "cumulative.y",
        offset: GLOBAL_COL_MAP.cumulative.y[0],
        len: 11,
    },
    GlobalLayoutFieldDescriptor {
        name: "cumulative.z",
        offset: GLOBAL_COL_MAP.cumulative.z[0],
        len: 11,
    },
    GlobalLayoutFieldDescriptor {
        name: "quotient.map",
        offset: GLOBAL_COL_MAP.quotient.map[0],
        len: 10,
    },
    GlobalLayoutFieldDescriptor {
        name: "quotient.u0",
        offset: GLOBAL_COL_MAP.quotient.u0[0],
        len: 6,
    },
    GlobalLayoutFieldDescriptor {
        name: "quotient.u1",
        offset: GLOBAL_COL_MAP.quotient.u1[0],
        len: 10,
    },
    GlobalLayoutFieldDescriptor {
        name: "quotient.u3",
        offset: GLOBAL_COL_MAP.quotient.u3[0],
        len: 10,
    },
    GlobalLayoutFieldDescriptor {
        name: "quotient.u4",
        offset: GLOBAL_COL_MAP.quotient.u4[0],
        len: 6,
    },
    GlobalLayoutFieldDescriptor {
        name: "quotient.u5",
        offset: GLOBAL_COL_MAP.quotient.u5[0],
        len: 10,
    },
    GlobalLayoutFieldDescriptor {
        name: "quotient.output_x",
        offset: GLOBAL_COL_MAP.quotient.output_x[0],
        len: 10,
    },
    GlobalLayoutFieldDescriptor {
        name: "quotient.output_y",
        offset: GLOBAL_COL_MAP.quotient.output_y[0],
        len: 10,
    },
    GlobalLayoutFieldDescriptor {
        name: "quotient.output_z",
        offset: GLOBAL_COL_MAP.quotient.output_z[0],
        len: 10,
    },
];

const _: () = {
    assert!(NUM_GLOBAL_COLS == 228);
    assert!(core::mem::offset_of!(GlobalCols<u8>, message_rest) == 0);
    assert!(core::mem::offset_of!(GlobalCols<u8>, x6) == 6);
    assert!(core::mem::offset_of!(GlobalCols<u8>, x5) == 7);
    assert!(core::mem::offset_of!(GlobalCols<u8>, m0_lo16) == 8);
    assert!(core::mem::offset_of!(GlobalCols<u8>, y_lower) == 10);
    assert!(core::mem::offset_of!(GlobalCols<u8>, w_lo16) == 20);
    assert!(core::mem::offset_of!(GlobalCols<u8>, is_receive) == 22);
    assert!(core::mem::offset_of!(GlobalCols<u8>, index) == 24);
    assert!(core::mem::offset_of!(GlobalCols<u8>, input) == 25);
    assert!(core::mem::offset_of!(GlobalCols<u8>, products) == 58);
    assert!(core::mem::offset_of!(GlobalCols<u8>, cumulative) == 113);
    assert!(core::mem::offset_of!(GlobalCols<u8>, quotient) == 146);
    assert!(GLOBAL_COL_MAP.quotient.map[0] == 146);
    assert!(GLOBAL_COL_MAP.quotient.u0[0] == 156);
    assert!(GLOBAL_COL_MAP.quotient.u1[0] == 162);
    assert!(GLOBAL_COL_MAP.quotient.u3[0] == 172);
    assert!(GLOBAL_COL_MAP.quotient.u4[0] == 182);
    assert!(GLOBAL_COL_MAP.quotient.u5[0] == 188);
    assert!(GLOBAL_COL_MAP.quotient.output_x[0] == 198);
    assert!(GLOBAL_COL_MAP.quotient.output_y[0] == 208);
    assert!(GLOBAL_COL_MAP.quotient.output_z[0] == 218);
    let mut next = 0;
    let mut field = 0;
    while field < GLOBAL_LAYOUT_FIELDS.len() {
        assert!(GLOBAL_LAYOUT_FIELDS[field].offset == next);
        next += GLOBAL_LAYOUT_FIELDS[field].len;
        field += 1;
    }
    assert!(next == NUM_GLOBAL_COLS);
    assert!(NUM_GLOBAL_TILE_REDUCER_COLS == 83);
    assert!(REDUCER_LEAF_P_BITS_START + REDUCER_LEAF_P_BITS == REDUCER_LEAF_GAP_BITS_START);
    assert!(REDUCER_LEAF_GAP_BITS_START + REDUCER_LEAF_GAP_BITS == 62);
    assert!(REDUCER_PRODUCT_TO_ROOT_VALUE < 66);
    assert!(GLOBAL_TILE_REDUCER_COL_MAP.mode_leaf == 0);
    assert!(GLOBAL_TILE_REDUCER_COL_MAP.mode_node_input == 1);
    assert!(GLOBAL_TILE_REDUCER_COL_MAP.mode_product == 2);
    assert!(GLOBAL_TILE_REDUCER_COL_MAP.mode_node_middle == 3);
    assert!(GLOBAL_TILE_REDUCER_COL_MAP.mode_node_output == 4);
    assert!(GLOBAL_TILE_REDUCER_COL_MAP.mode_root_input == 5);
    assert!(GLOBAL_TILE_REDUCER_COL_MAP.mode_root_output == 6);
    assert!(GLOBAL_TILE_REDUCER_COL_MAP.selector_spare == 7);
    assert!(GLOBAL_TILE_REDUCER_COL_MAP.payload.control[0] == 8);
    assert!(GLOBAL_TILE_REDUCER_COL_MAP.payload.control[5] == 13);
    assert!(GLOBAL_TILE_REDUCER_COL_MAP.payload.values[0] == 14);
    assert!(GLOBAL_TILE_REDUCER_COL_MAP.payload.values[65] == 79);
    assert!(GLOBAL_TILE_REDUCER_COL_MAP.control_rank == 80);
    assert!(GLOBAL_TILE_REDUCER_COL_MAP.control_next_rank == 81);
    assert!(GLOBAL_TILE_REDUCER_COL_MAP.control_next_tag == 82);
};
