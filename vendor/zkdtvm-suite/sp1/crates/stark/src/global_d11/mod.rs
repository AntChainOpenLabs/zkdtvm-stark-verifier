//! Protocol primitives for the D11 Global scheme.
//!
//! The coefficient convention is low-to-high in the basis
//! `1, z, ..., z^10`, with `z^11 = z^3 + 2`.  This module deliberately has
//! no executor or machine dependency so that host, AIR, recursion, and native
//! consumers can share one arithmetic authority.

mod boundary;
mod constants;
mod curve;
mod field;
mod identity;
mod interaction;
mod kernels;
mod manifest;
mod map;

pub use constants::*;
pub use curve::{curve_b, D11AffinePointV1, D11ProjectivePointV1, ProjectivePointError};
pub use field::{D11Sparse7, D11};
pub use identity::*;
pub use interaction::{
    flatten_projective_chain_v1, pack_projective_chain_blocks_v1, ProjectiveChainPayloadV1,
    PROJECTIVE_CHAIN_BASE_VALUES, PROJECTIVE_CHAIN_BLOCKS, PROJECTIVE_CHAIN_BLOCK_WIDTH,
};
pub use kernels::{KernelCost, DENSE_SPARSE_7_COST, SCHOOLBOOK_COST, SQUARE_COST};
pub use manifest::{overflow_certificate_digest, parameter_manifest_digest};
pub use map::{
    apply_direction, canonicalize_y, construct_map, construct_map_reference, direct_map_residual,
    fixed_padding_dummy, pack_unsigned, GlobalMapErrorV1, GlobalMapWitnessV1, GlobalPackErrorV1,
    GlobalPackInputV1, GlobalPackWordSemanticsV1, GlobalSignedMapRowV1,
    GLOBAL_PACK_WORD_SEMANTICS_V1,
};

#[cfg(test)]
mod tests;
pub use boundary::*;
