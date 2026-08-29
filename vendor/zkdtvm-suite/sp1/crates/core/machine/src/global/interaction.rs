use dt_stark::{
    air::FullAirBuilder,
    global_d11::{PROJECTIVE_CHAIN_BASE_VALUES, PROJECTIVE_CHAIN_BLOCKS},
    InteractionKind, InteractionValueEncoding,
};
use p3_field::AbstractField;

use super::columns::D11PointCols;

/// Direction of one entry in the canonical Global lookup schedule.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LookupDirection {
    Send,
    Receive,
}

/// Proof-visible meaning of one entry in the canonical Global lookup schedule.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum GlobalInteractionSemantic {
    Endpoint = 1,
    Message0U16 = 2,
    Message0AndMessage5U8 = 3,
    Message6AndKindU8 = 4,
    TweakBitRange = 5,
    SignedYLowU16 = 6,
    SignedYHighU16 = 7,
    SignedYHighComplementU16 = 8,
    ProjectiveInput = 9,
    ProjectiveOutput = 10,
}

/// Minimal typed authority shared by the direct and PolyAir lookup adapters.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GlobalInteractionDescriptor {
    pub semantic: GlobalInteractionSemantic,
    pub kind: InteractionKind,
    pub direction: LookupDirection,
    pub value_encoding: InteractionValueEncoding,
    pub base_values: usize,
    pub slots_per_row: u32,
}

const BASE: InteractionValueEncoding = InteractionValueEncoding::Base;
const EXT5: InteractionValueEncoding = InteractionValueEncoding::ExtensionBlocks { degree: 5 };

/// Exact ordered lookup schedule compiled into the canonical Global AIR.
pub const GLOBAL_INTERACTION_DESCRIPTORS: [GlobalInteractionDescriptor; 10] = [
    GlobalInteractionDescriptor {
        semantic: GlobalInteractionSemantic::Endpoint,
        kind: InteractionKind::Global,
        direction: LookupDirection::Receive,
        value_encoding: BASE,
        base_values: 10,
        slots_per_row: 1,
    },
    GlobalInteractionDescriptor {
        semantic: GlobalInteractionSemantic::Message0U16,
        kind: InteractionKind::Byte,
        direction: LookupDirection::Send,
        value_encoding: BASE,
        base_values: 5,
        slots_per_row: 1,
    },
    GlobalInteractionDescriptor {
        semantic: GlobalInteractionSemantic::Message0AndMessage5U8,
        kind: InteractionKind::Byte,
        direction: LookupDirection::Send,
        value_encoding: BASE,
        base_values: 5,
        slots_per_row: 1,
    },
    GlobalInteractionDescriptor {
        semantic: GlobalInteractionSemantic::Message6AndKindU8,
        kind: InteractionKind::Byte,
        direction: LookupDirection::Send,
        value_encoding: BASE,
        base_values: 5,
        slots_per_row: 1,
    },
    GlobalInteractionDescriptor {
        semantic: GlobalInteractionSemantic::TweakBitRange,
        kind: InteractionKind::Byte,
        direction: LookupDirection::Send,
        value_encoding: BASE,
        base_values: 5,
        slots_per_row: 1,
    },
    GlobalInteractionDescriptor {
        semantic: GlobalInteractionSemantic::SignedYLowU16,
        kind: InteractionKind::Byte,
        direction: LookupDirection::Send,
        value_encoding: BASE,
        base_values: 5,
        slots_per_row: 1,
    },
    GlobalInteractionDescriptor {
        semantic: GlobalInteractionSemantic::SignedYHighU16,
        kind: InteractionKind::Byte,
        direction: LookupDirection::Send,
        value_encoding: BASE,
        base_values: 5,
        slots_per_row: 1,
    },
    GlobalInteractionDescriptor {
        semantic: GlobalInteractionSemantic::SignedYHighComplementU16,
        kind: InteractionKind::Byte,
        direction: LookupDirection::Send,
        value_encoding: BASE,
        base_values: 5,
        slots_per_row: 1,
    },
    GlobalInteractionDescriptor {
        semantic: GlobalInteractionSemantic::ProjectiveInput,
        kind: InteractionKind::GlobalProjectiveChainV2,
        direction: LookupDirection::Receive,
        value_encoding: EXT5,
        base_values: PROJECTIVE_CHAIN_BASE_VALUES,
        slots_per_row: 1,
    },
    GlobalInteractionDescriptor {
        semantic: GlobalInteractionSemantic::ProjectiveOutput,
        kind: InteractionKind::GlobalProjectiveChainV2,
        direction: LookupDirection::Send,
        value_encoding: EXT5,
        base_values: PROJECTIVE_CHAIN_BASE_VALUES,
        slots_per_row: 1,
    },
];

/// Flattens the symbolic chain payload in the shared protocol order.
#[must_use]
pub fn projective_chain_payload<T: Clone>(
    index: T,
    point: &D11PointCols<T>,
) -> [T; PROJECTIVE_CHAIN_BASE_VALUES] {
    core::array::from_fn(|offset| match offset {
        0 => index.clone(),
        1..=11 => point.x[offset - 1].clone(),
        12..=22 => point.y[offset - 12].clone(),
        23..=33 => point.z[offset - 23].clone(),
        _ => unreachable!("fixed projective-chain payload offset"),
    })
}

/// Builds the domain-separated denominator from seven packed Ext5 blocks.
pub fn projective_chain_denominator<AB: FullAirBuilder>(
    builder: &AB,
    index: AB::VarMaybeExt,
    point: &D11PointCols<AB::VarMaybeExt>,
) -> AB::VarExt {
    let payload = projective_chain_payload(index, point);
    let blocks: [AB::VarExt; PROJECTIVE_CHAIN_BLOCKS] = core::array::from_fn(|block| {
        let start = block * 5;
        let limbs: [AB::VarMaybeExt; 5] = core::array::from_fn(|limb| {
            payload.get(start + limb).cloned().unwrap_or_else(AB::zero_maybe)
        });
        AB::pack_ext_limbs(&limbs)
    });
    let interaction_kind = AB::VarMaybeExt::from(AB::F::from_canonical_usize(
        InteractionKind::GlobalProjectiveChainV2 as usize,
    ));
    builder.lookup_denominator_ext_blocks(interaction_kind, blocks)
}

const _: () = assert!(PROJECTIVE_CHAIN_BLOCKS == 7);
