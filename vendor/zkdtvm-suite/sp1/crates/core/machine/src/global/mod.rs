//! Canonical Global row relation and writer authority.

mod boundary;
mod chip;
mod columns;
mod constraints;
mod interaction;
mod link;
mod writer;

pub mod kind;
pub mod sources;

#[cfg(feature = "test-utils")]
#[doc(hidden)]
pub mod p7_kats {
    pub fn quotient_reconstructs_dense_and_sparse_products() {
        super::constraints::p7_kats::quotient_reconstructs_dense_and_sparse_products();
    }

    pub fn map_quotient_matches_high_square_and_x_q2_formula() {
        super::constraints::p7_kats::map_quotient_matches_high_square_and_x_q2_formula();
    }

    pub fn selected_output_quotients_use_wave2_signed_pairs() {
        super::constraints::p7_kats::selected_output_quotients_use_wave2_signed_pairs();
    }

    pub fn tile_reducer_fixed_tree_and_product_beta() {
        super::link::p8_kats::tile_reducer_fixed_tree_and_product_beta();
    }

    pub fn tile_reducer_rejects_malicious_schedule() {
        super::link::p8_kats::tile_reducer_rejects_malicious_schedule();
    }
}

pub use boundary::{
    claim_from_compressed_tile_reducer_trace, GlobalBoundaryLayout, GlobalBoundaryLayoutError,
    GLOBAL_BOUNDARY_LAYOUT,
};
pub use chip::GlobalChip;
pub use columns::{
    D11PointCols, D11ProductCols, D11QuotientCols, GlobalCols, GlobalLayoutFieldDescriptor,
    GlobalTileReducerCols, GlobalTileReducerPayloadCols, GLOBAL_COL_MAP, GLOBAL_LAYOUT_FIELDS,
    GLOBAL_TILE_REDUCER_COL_MAP, NUM_GLOBAL_COLS, NUM_GLOBAL_TILE_REDUCER_COLS,
};
pub use interaction::{
    projective_chain_denominator, projective_chain_payload, GlobalInteractionDescriptor,
    GlobalInteractionSemantic, LookupDirection, GLOBAL_INTERACTION_DESCRIPTORS,
};
pub use link::GlobalTileReducerChip;
#[cfg(feature = "test-utils")]
pub use writer::global_relation_accepts_for_test;
pub(crate) use writer::{
    prepare_global_trace_for_field, prepare_global_trace_for_field_from_start,
};
pub use writer::{
    global_padded_rows, global_tile_count, global_tile_reducer_padded_rows,
    global_tile_reducer_real_rows, prepare_global_trace, GlobalByteLookupDelta, GlobalPrepareError,
    PreparedGlobalTrace, RetainedGlobalTrace,
};
