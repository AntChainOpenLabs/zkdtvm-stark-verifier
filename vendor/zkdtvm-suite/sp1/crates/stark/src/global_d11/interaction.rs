use p3_field::{AbstractExtensionField, AbstractField};

use super::{D11ProjectivePointV1, D11_DEGREE};

/// Number of base-field values in `[index, X[11], Y[11], Z[11]]`.
pub const PROJECTIVE_CHAIN_BASE_VALUES: usize = 1 + 3 * D11_DEGREE;
/// Number of base limbs packed into one protocol extension value.
pub const PROJECTIVE_CHAIN_BLOCK_WIDTH: usize = 5;
/// Number of quintic blocks in one projective-chain payload.
pub const PROJECTIVE_CHAIN_BLOCKS: usize =
    PROJECTIVE_CHAIN_BASE_VALUES.div_ceil(PROJECTIVE_CHAIN_BLOCK_WIDTH);

/// Typed, low-to-high base-field representation of one indexed chain state.
#[derive(Clone, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct ProjectiveChainPayloadV1<T>(pub [T; PROJECTIVE_CHAIN_BASE_VALUES]);

/// Flattens in the only protocol order: `index, X, Y, Z`.
#[must_use]
pub fn flatten_projective_chain_v1<T: Clone>(
    index: T,
    point: &D11ProjectivePointV1<T>,
) -> ProjectiveChainPayloadV1<T>
where
    T: p3_field::Field,
{
    ProjectiveChainPayloadV1(core::array::from_fn(|offset| match offset {
        0 => index.clone(),
        1..=11 => point.x.coefficients()[offset - 1].clone(),
        12..=22 => point.y.coefficients()[offset - 12].clone(),
        23..=33 => point.z.coefficients()[offset - 23].clone(),
        _ => unreachable!("fixed projective-chain payload offset"),
    }))
}

/// Packs 34 base values into seven quintic-extension blocks. The sole spare
/// limb, block 6 limb 4, is structurally zero rather than witness supplied.
#[must_use]
pub fn pack_projective_chain_blocks_v1<Base, Ext>(
    payload: &ProjectiveChainPayloadV1<Base>,
) -> [Ext; PROJECTIVE_CHAIN_BLOCKS]
where
    Base: AbstractField + Clone,
    Ext: AbstractExtensionField<Base>,
{
    assert_eq!(
        Ext::D,
        PROJECTIVE_CHAIN_BLOCK_WIDTH,
        "GlobalProjectiveChainV2 requires the frozen quintic basis"
    );
    core::array::from_fn(|block| {
        let start = block * PROJECTIVE_CHAIN_BLOCK_WIDTH;
        Ext::from_base_fn(|limb| payload.0.get(start + limb).cloned().unwrap_or_else(Base::zero))
    })
}

const _: () = {
    assert!(PROJECTIVE_CHAIN_BASE_VALUES == 34);
    assert!(PROJECTIVE_CHAIN_BLOCKS == 7);
};
