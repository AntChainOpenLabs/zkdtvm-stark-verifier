use std::cmp::Reverse;

use p3_challenger::CanObserve;
use p3_field::PrimeField32;

/// Transcript domain for the verifier-derived active AIR inventory ("ASH1" LE).
pub const ACTIVE_SHAPE_TAG_V1: u32 = 0x3148_5341;
pub const ACTIVE_SHAPE_VERSION_V2: u32 = 2;
pub const MAX_ACTIVE_AIRS_V1: usize = 256;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ActiveShapeEntryV1 {
    pub stable_id: u32,
    pub log_height: u32,
    pub main_width: u32,
    pub derived_index: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ActiveShapeErrorV1 {
    TooManyEntries(usize),
    EmptyName,
    DuplicateName(String),
    DuplicateStableId(u32),
    NonCanonicalOrder,
    IntegerOverflow,
    ZeroMainWidth(String),
}

/// Stable AIR identity authority shared by direct and PolyAir adapters.
#[must_use]
pub fn stable_air_id_v1(name: &str) -> u32 {
    match name {
        "Global" => return 43,
        "GlobalTileReducerV3" => return 60,
        _ => {}
    }
    let mut hash = 0x811c_9dc5u32;
    for byte in b"dt-active-air-v1\0".iter().chain(name.as_bytes()) {
        hash ^= u32::from(*byte);
        hash = hash.wrapping_mul(0x0100_0193);
    }
    hash | 0x8000_0000
}

/// Derive the canonical live descriptor from authenticated chip assignments and PCS dimensions.
pub fn derive_active_shape_v1(
    entries: impl IntoIterator<Item = (String, usize, usize)>,
) -> Result<Vec<ActiveShapeEntryV1>, ActiveShapeErrorV1> {
    let raw = entries.into_iter().collect::<Vec<_>>();
    if raw.len() > MAX_ACTIVE_AIRS_V1 {
        return Err(ActiveShapeErrorV1::TooManyEntries(raw.len()));
    }
    let mut names = std::collections::BTreeSet::new();
    let mut ids = std::collections::BTreeSet::new();
    let mut previous: Option<(Reverse<usize>, Vec<u8>)> = None;
    let mut result = Vec::with_capacity(raw.len());
    for (index, (name, main_width, log_height)) in raw.into_iter().enumerate() {
        if name.is_empty() {
            return Err(ActiveShapeErrorV1::EmptyName);
        }
        if main_width == 0 {
            return Err(ActiveShapeErrorV1::ZeroMainWidth(name));
        }
        if !names.insert(name.clone()) {
            return Err(ActiveShapeErrorV1::DuplicateName(name));
        }
        let stable_id = stable_air_id_v1(&name);
        if !ids.insert(stable_id) {
            return Err(ActiveShapeErrorV1::DuplicateStableId(stable_id));
        }
        let key = (Reverse(log_height), name.as_bytes().to_vec());
        if previous.as_ref().is_some_and(|prior| prior > &key) {
            return Err(ActiveShapeErrorV1::NonCanonicalOrder);
        }
        previous = Some(key);
        result.push(ActiveShapeEntryV1 {
            stable_id,
            log_height: u32::try_from(log_height)
                .map_err(|_| ActiveShapeErrorV1::IntegerOverflow)?,
            main_width: u32::try_from(main_width)
                .map_err(|_| ActiveShapeErrorV1::IntegerOverflow)?,
            derived_index: u32::try_from(index).map_err(|_| ActiveShapeErrorV1::IntegerOverflow)?,
        });
    }
    Ok(result)
}

pub fn observe_active_shape_v1<F, Challenger>(
    challenger: &mut Challenger,
    entries: &[ActiveShapeEntryV1],
) where
    F: PrimeField32,
    Challenger: CanObserve<F>,
{
    for word in active_shape_transcript_words_v2(entries) {
        challenger.observe(F::from_canonical_u32(word));
    }
}

/// Canonical field-safe transcript words shared by host and recursive verifiers.
#[must_use]
pub fn active_shape_transcript_words_v2(entries: &[ActiveShapeEntryV1]) -> Vec<u32> {
    let mut words = Vec::with_capacity(3 + 5 * entries.len());
    words.push(ACTIVE_SHAPE_TAG_V1);
    words.push(ACTIVE_SHAPE_VERSION_V2);
    words.push(u32::try_from(entries.len()).expect("active AIR count is bounded by 256"));
    for entry in entries {
        words.push(entry.stable_id & 0xffff);
        words.push(entry.stable_id >> 16);
        words.push(entry.log_height);
        words.push(entry.main_width);
        words.push(entry.derived_index);
    }
    words
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn global_shape_has_frozen_identities_and_transcript_words() {
        let shape = derive_active_shape_v1([
            ("Global".to_string(), 228, 10),
            ("GlobalTileReducerV3".to_string(), 83, 4),
        ])
        .unwrap();
        assert_eq!(shape[0].stable_id, 43);
        assert_eq!(shape[1].stable_id, 60);
        assert_eq!(
            active_shape_transcript_words_v2(&shape),
            vec![
                ACTIVE_SHAPE_TAG_V1,
                ACTIVE_SHAPE_VERSION_V2,
                2,
                43,
                0,
                10,
                228,
                0,
                60,
                0,
                4,
                83,
                1,
            ]
        );
    }

    #[test]
    fn shape_derivation_rejects_noncanonical_order_and_duplicate_names() {
        assert_eq!(
            derive_active_shape_v1([("Cpu".to_string(), 16, 8), ("Global".to_string(), 228, 10),]),
            Err(ActiveShapeErrorV1::NonCanonicalOrder)
        );
        assert_eq!(
            derive_active_shape_v1([
                ("Global".to_string(), 228, 10),
                ("Global".to_string(), 228, 8),
            ]),
            Err(ActiveShapeErrorV1::DuplicateName("Global".to_string()))
        );
    }
}
