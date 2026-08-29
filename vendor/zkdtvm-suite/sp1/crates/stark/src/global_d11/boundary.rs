use p3_challenger::CanObserve;
use p3_field::{ExtensionField, Field, PrimeField32};
use serde::{Deserialize, Serialize};
use tiny_keccak::{Hasher, Keccak};

use super::{
    flatten_projective_chain_v1, pack_projective_chain_blocks_v1, D11AffinePointV1,
    D11ProjectivePointV1, ProjectivePointError, D11, PROJECTIVE_CHAIN_BLOCKS,
};
use crate::{
    air::{GlobalClaim, GlobalState},
    InteractionKind,
};

const OWNER_REGISTRY_DOMAIN: &[u8] = b"dt-global-d11-boundary-owner-registry-v2\0";
pub const GLOBAL_CLAIM_SCHEMA_TAG_V3: u32 = 0x0d31;
pub const PROGRAM_BOUNDARY_TAG_V1: u32 = 0x0d12;
pub const GLOBAL_MAX_LOG_HEIGHT: u8 = 22;
pub const CORE_GLOBAL_OWNER: StableChipId = StableChipId(43);
pub const CORE_GLOBAL_OWNER_REGISTRY_DIGEST: [u8; 32] = [
    0x07, 0x87, 0x14, 0xa7, 0xaf, 0x0f, 0xd0, 0x6c, 0x09, 0xad, 0x9b, 0x49, 0x21, 0xbd, 0x0b, 0xc9,
    0xf1, 0x4a, 0xd0, 0x57, 0x22, 0x95, 0x3a, 0xc4, 0xb9, 0xf1, 0x34, 0x4c, 0x75, 0x49, 0xbb, 0x1d,
];

/// Manifest-bound numeric identity of the sole proof-system Global owner.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[repr(transparent)]
pub struct StableChipId(pub u32);

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[repr(u8)]
pub enum GlobalBoundaryKindV2 {
    Projective = 1,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct BoundaryOwnerV2 {
    pub owner: StableChipId,
    pub kind: GlobalBoundaryKindV2,
}

/// Key/transcript authority for the unique Global claim owner.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct BoundaryOwnerRegistryV2 {
    pub owners: Vec<BoundaryOwnerV2>,
    pub digest: [u8; 32],
}

/// Canonical program-image boundary carried by the verifying key.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ProgramImageBoundaryV1<F> {
    Infinity,
    Affine { x: [F; 11], y: [F; 11] },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GlobalBoundaryError {
    DuplicateRegistryOwner(StableChipId),
    RegistryDigestMismatch,
    HeightExceeded { owner: StableChipId, log_height: u8, maximum: u8 },
    InvalidProgramPoint(ProjectivePointError),
    InvalidRoot(ProjectivePointError),
    RootNotIdentity,
    WireTruncated,
    NonCanonicalField(u32),
    InvalidClaimPresence(u32),
    ClaimOpeningMismatch,
    ClaimCountZero,
    NonCanonicalAbsentClaim,
    FirstSeedMismatch,
    DiscontinuousInterval { left: usize, right: usize },
    NonCanonicalActiveClaim,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProjectiveChainBoundaryBlocksV2<Ext> {
    pub source: [Ext; PROJECTIVE_CHAIN_BLOCKS],
    pub sink: [Ext; PROJECTIVE_CHAIN_BLOCKS],
}

fn point_from_state<F: Field>(state: &GlobalState<F>) -> D11ProjectivePointV1<F> {
    D11ProjectivePointV1 { x: D11::new(state.x), y: D11::new(state.y), z: D11::new(state.z) }
}

#[must_use]
pub fn state_from_projective<F: Field>(point: &D11ProjectivePointV1<F>) -> GlobalState<F> {
    GlobalState {
        x: *point.x.coefficients(),
        y: *point.y.coefficients(),
        z: *point.z.coefficients(),
    }
}

#[must_use]
pub fn empty_global_claim<F: PrimeField32>() -> GlobalClaim<F> {
    let state = state_from_projective(&D11ProjectivePointV1::identity());
    GlobalClaim {
        has_global_opening: F::zero(),
        count: F::zero(),
        interval: crate::air::GlobalStateInterval { start: state, end: state },
    }
}

pub fn validate_global_claim<F: PrimeField32>(
    claim: &GlobalClaim<F>,
    has_authenticated_opening: bool,
) -> Result<(), GlobalBoundaryError> {
    let has = claim.has_global_opening.as_canonical_u32();
    if has > 1 {
        return Err(GlobalBoundaryError::InvalidClaimPresence(has));
    }
    if (has == 1) != has_authenticated_opening {
        return Err(GlobalBoundaryError::ClaimOpeningMismatch);
    }
    let count = claim.count.as_canonical_u32();
    let start = point_from_state(&claim.interval.start);
    let end = point_from_state(&claim.interval.end);
    let canonical = |point: D11ProjectivePointV1<F>| {
        point.validate().is_ok() &&
            ((point.z.is_zero() && point == D11ProjectivePointV1::identity()) ||
                (!point.z.is_zero() && point.z == D11::one()))
    };
    if has_authenticated_opening {
        if count == 0 {
            return Err(GlobalBoundaryError::ClaimCountZero);
        }
        if !canonical(start) || !canonical(end) {
            return Err(GlobalBoundaryError::NonCanonicalActiveClaim);
        }
    } else if count != 0
        || claim.interval.start != claim.interval.end
        || !canonical(start)
    {
        return Err(GlobalBoundaryError::NonCanonicalAbsentClaim);
    }
    Ok(())
}

fn keccak256(input: &[u8]) -> [u8; 32] {
    let mut digest = [0u8; 32];
    let mut hasher = Keccak::v256();
    hasher.update(input);
    hasher.finalize(&mut digest);
    digest
}

fn registry_digest(owners: &[BoundaryOwnerV2]) -> [u8; 32] {
    let mut bytes = Vec::with_capacity(OWNER_REGISTRY_DOMAIN.len() + 8 + 5 * owners.len());
    bytes.extend_from_slice(OWNER_REGISTRY_DOMAIN);
    bytes.extend_from_slice(
        &u64::try_from(owners.len()).expect("owner registry length must fit u64").to_le_bytes(),
    );
    for entry in owners {
        bytes.extend_from_slice(&entry.owner.0.to_le_bytes());
        bytes.push(entry.kind as u8);
    }
    keccak256(&bytes)
}

impl BoundaryOwnerRegistryV2 {
    pub fn new(owners: Vec<BoundaryOwnerV2>) -> Result<Self, GlobalBoundaryError> {
        for (index, entry) in owners.iter().enumerate() {
            if owners[..index].iter().any(|other| other.owner == entry.owner) {
                return Err(GlobalBoundaryError::DuplicateRegistryOwner(entry.owner));
            }
        }
        let digest = registry_digest(&owners);
        Ok(Self { owners, digest })
    }

    pub fn validate(&self) -> Result<(), GlobalBoundaryError> {
        for (index, entry) in self.owners.iter().enumerate() {
            if self.owners[..index].iter().any(|other| other.owner == entry.owner) {
                return Err(GlobalBoundaryError::DuplicateRegistryOwner(entry.owner));
            }
        }
        if registry_digest(&self.owners) != self.digest {
            return Err(GlobalBoundaryError::RegistryDigestMismatch);
        }
        Ok(())
    }

    #[must_use]
    pub fn position(&self, owner: StableChipId) -> Option<usize> {
        self.owners.iter().position(|entry| entry.owner == owner)
    }
}

pub fn core_global_owner_registry() -> BoundaryOwnerRegistryV2 {
    let registry = BoundaryOwnerRegistryV2::new(vec![BoundaryOwnerV2 {
        owner: CORE_GLOBAL_OWNER,
        kind: GlobalBoundaryKindV2::Projective,
    }])
    .expect("frozen core Global owner registry must be valid");
    assert_eq!(registry.digest, CORE_GLOBAL_OWNER_REGISTRY_DIGEST);
    registry
}

pub fn observe_owner_registry_v2<F, Challenger>(
    challenger: &mut Challenger,
    registry: &BoundaryOwnerRegistryV2,
) -> Result<(), GlobalBoundaryError>
where
    F: PrimeField32,
    Challenger: CanObserve<F>,
{
    registry.validate()?;
    challenger.observe(F::from_canonical_usize(registry.owners.len()));
    for entry in &registry.owners {
        challenger.observe(F::from_canonical_u32(entry.owner.0));
        challenger.observe(F::from_canonical_u8(entry.kind as u8));
    }
    challenger.observe_slice(&registry.digest.map(F::from_canonical_u8));
    Ok(())
}

pub fn canonical_program_boundary_transcript_fields_v1<F: PrimeField32>(
    boundary: &ProgramImageBoundaryV1<u32>,
) -> Result<Vec<F>, GlobalBoundaryError> {
    let canonical = canonical_program_boundary_fields_v1::<F>(boundary)?;
    let mut fields = Vec::with_capacity(24);
    fields.push(F::from_canonical_u32(PROGRAM_BOUNDARY_TAG_V1));
    fields.extend(canonical);
    Ok(fields)
}

/// Canonical field encoding of a program boundary without the transcript domain tag.
///
/// The encoding is `kind || x[11] || y[11]`, where `kind=0` denotes infinity and
/// its unused affine payload is all zero, while `kind=1` denotes the supplied affine point.
pub fn canonical_program_boundary_fields_v1<F: PrimeField32>(
    boundary: &ProgramImageBoundaryV1<u32>,
) -> Result<[F; 23], GlobalBoundaryError> {
    let _ = program_global_seed::<F>(boundary)?;
    let mut fields = [F::zero(); 23];
    match boundary {
        ProgramImageBoundaryV1::Infinity => {}
        ProgramImageBoundaryV1::Affine { x, y } => {
            fields[0] = F::one();
            for (dst, src) in fields[1..12].iter_mut().zip(x) {
                *dst = F::from_canonical_u32(*src);
            }
            for (dst, src) in fields[12..23].iter_mut().zip(y) {
                *dst = F::from_canonical_u32(*src);
            }
        }
    }
    Ok(fields)
}

pub fn observe_program_boundary_v1<F, Challenger>(
    challenger: &mut Challenger,
    boundary: &ProgramImageBoundaryV1<u32>,
) -> Result<(), GlobalBoundaryError>
where
    F: PrimeField32,
    Challenger: CanObserve<F>,
{
    challenger.observe_slice(&canonical_program_boundary_transcript_fields_v1::<F>(boundary)?);
    Ok(())
}

/// Observe the role-specific program-Global metadata after the VK commitment.
///
/// The owner registry is a validated construction-time VK authority and is already part of the
/// key identity; it is deliberately not repeated in every child proof transcript/metadata row.
/// Core keys have a Global owner and observe `pc || kind || x[11] || y[11]`. Native keys have an
/// empty registry, for which setup fixes pc to zero and the seed to identity, and observe neither.
pub fn observe_program_global_metadata_v2<F, Challenger>(
    challenger: &mut Challenger,
    pc_start: F,
    boundary: &ProgramImageBoundaryV1<u32>,
    registry: &BoundaryOwnerRegistryV2,
) -> Result<(), GlobalBoundaryError>
where
    F: PrimeField32,
    Challenger: CanObserve<F>,
{
    registry.validate()?;
    if registry.owners.is_empty() {
        return Ok(());
    }
    challenger.observe(pc_start);
    challenger.observe_slice(&canonical_program_boundary_fields_v1::<F>(boundary)?);
    Ok(())
}

pub fn projective_chain_claim_blocks_v2<Base, Ext>(
    claim: &GlobalClaim<Base>,
) -> ProjectiveChainBoundaryBlocksV2<Ext>
where
    Base: PrimeField32,
    Ext: p3_field::AbstractExtensionField<Base>,
{
    let start = point_from_state(&claim.interval.start);
    let end = point_from_state(&claim.interval.end);
    let source = flatten_projective_chain_v1(Base::zero(), &start);
    let sink = flatten_projective_chain_v1(claim.count, &end);
    ProjectiveChainBoundaryBlocksV2 {
        source: pack_projective_chain_blocks_v1(&source),
        sink: pack_projective_chain_blocks_v1(&sink),
    }
}

fn projective_chain_fingerprint_v2<Base, Ext>(
    alpha: Ext,
    beta: Ext,
    blocks: &[Ext; PROJECTIVE_CHAIN_BLOCKS],
) -> Ext
where
    Base: PrimeField32,
    Ext: ExtensionField<Base>,
{
    let mut fingerprint =
        alpha + Ext::from_canonical_usize(InteractionKind::GlobalProjectiveChainV2 as usize);
    let mut beta_powers = beta.powers().skip(1);
    for block in blocks {
        fingerprint += beta_powers.next().expect("unbounded powers iterator") * *block;
    }
    fingerprint
}

pub fn compute_expected_global_claim_imbalance_v2<Base, Ext>(
    alpha: Ext,
    beta: Ext,
    claim: &GlobalClaim<Base>,
) -> Result<Ext, GlobalBoundaryError>
where
    Base: PrimeField32,
    Ext: ExtensionField<Base>,
{
    if claim.has_global_opening.is_zero() {
        return Ok(Ext::zero());
    }
    let blocks = projective_chain_claim_blocks_v2::<Base, Ext>(claim);
    let source = projective_chain_fingerprint_v2::<Base, Ext>(alpha, beta, &blocks.source);
    let sink = projective_chain_fingerprint_v2::<Base, Ext>(alpha, beta, &blocks.sink);
    Ok(sink.inverse() - source.inverse())
}

pub fn program_global_seed<F: PrimeField32>(
    boundary: &ProgramImageBoundaryV1<u32>,
) -> Result<D11ProjectivePointV1<F>, GlobalBoundaryError> {
    match boundary {
        ProgramImageBoundaryV1::Infinity => Ok(D11ProjectivePointV1::identity()),
        ProgramImageBoundaryV1::Affine { x, y } => {
            if let Some(value) = x.iter().chain(y).copied().find(|value| *value >= F::ORDER_U32) {
                return Err(GlobalBoundaryError::NonCanonicalField(value));
            }
            let affine =
                D11AffinePointV1 { x: D11::from_canonical_u32(*x), y: D11::from_canonical_u32(*y) };
            if !affine.is_on_curve() {
                return Err(GlobalBoundaryError::InvalidProgramPoint(
                    ProjectivePointError::OffCurve,
                ));
            }
            Ok(affine.to_projective())
        }
    }
}

pub fn canonicalize_projective_v2<F: PrimeField32>(
    point: D11ProjectivePointV1<F>,
) -> Result<D11ProjectivePointV1<F>, GlobalBoundaryError> {
    point.validate().map_err(GlobalBoundaryError::InvalidRoot)?;
    if point.is_identity() {
        return Ok(D11ProjectivePointV1::identity());
    }
    Ok(point.rescaled(point.z.inverse()))
}

/// Compose authenticated shard intervals in proof order and bind the program root.
pub fn verify_global_interval_root_v4<F: PrimeField32>(
    program: &ProgramImageBoundaryV1<u32>,
    claims: &[GlobalClaim<F>],
) -> Result<(), GlobalBoundaryError> {
    let identity = state_from_projective(&D11ProjectivePointV1::identity());
    let seed = state_from_projective(&program_global_seed::<F>(program)?);
    let mut previous_end = None;
    for (index, claim) in claims.iter().enumerate() {
        let active = !claim.has_global_opening.is_zero();
        validate_global_claim(claim, active)?;
        if let Some(previous) = previous_end {
            if claim.interval.start != previous {
                return Err(GlobalBoundaryError::DiscontinuousInterval {
                    left: index - 1,
                    right: index,
                });
            }
        } else if claim.interval.start != seed {
            return Err(GlobalBoundaryError::FirstSeedMismatch);
        }
        previous_end = Some(claim.interval.end);
    }
    if previous_end != Some(identity) {
        return Err(GlobalBoundaryError::RootNotIdentity);
    }
    Ok(())
}

const _: () = {
    assert!(CORE_GLOBAL_OWNER.0 == 43);
    assert!(GLOBAL_MAX_LOG_HEIGHT == 22);
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::air::GlobalStateInterval;
    use p3_challenger::CanObserve;
    use p3_field::{AbstractField, PrimeField32};
    use p3_koala_bear::KoalaBear;

    type F = KoalaBear;

    fn claim(start: GlobalState<F>, end: GlobalState<F>, count: u32) -> GlobalClaim<F> {
        GlobalClaim {
            has_global_opening: F::one(),
            count: F::from_canonical_u32(count),
            interval: GlobalStateInterval { start, end },
        }
    }

    fn scaled_infinity(scale: u32) -> GlobalState<F> {
        state_from_projective(&D11ProjectivePointV1 {
            x: D11::zero(),
            y: D11::from_base(F::from_canonical_u32(scale)),
            z: D11::zero(),
        })
    }

    #[derive(Default)]
    struct RecordingChallenger(Vec<F>);

    impl CanObserve<F> for RecordingChallenger {
        fn observe(&mut self, value: F) {
            self.0.push(value);
        }
    }

    fn words(values: &[F]) -> Vec<u32> {
        values.iter().map(PrimeField32::as_canonical_u32).collect()
    }

    #[test]
    fn direct_and_sumcheck_metadata_have_role_specific_sequences() {
        let boundary = ProgramImageBoundaryV1::Infinity;
        let core_registry = core_global_owner_registry();

        let mut direct = RecordingChallenger::default();
        observe_program_boundary_v1::<F, _>(&mut direct, &boundary).unwrap();
        observe_owner_registry_v2::<F, _>(&mut direct, &core_registry).unwrap();
        let direct = words(&direct.0);
        assert_eq!(direct.len(), 24 + 1 + 2 + 32);
        assert_eq!(&direct[..4], &[PROGRAM_BOUNDARY_TAG_V1, 0, 0, 0]);
        assert_eq!(&direct[24..27], &[1, CORE_GLOBAL_OWNER.0, 1]);
        assert_eq!(&direct[27..], &CORE_GLOBAL_OWNER_REGISTRY_DIGEST.map(u32::from));

        let pc = F::from_canonical_u32(0x20_0000);
        let mut sumcheck_core = RecordingChallenger::default();
        observe_program_global_metadata_v2(&mut sumcheck_core, pc, &boundary, &core_registry)
            .unwrap();
        let mut expected = vec![pc.as_canonical_u32()];
        expected.extend(
            canonical_program_boundary_fields_v1::<F>(&boundary)
                .unwrap()
                .map(|value| value.as_canonical_u32()),
        );
        assert_eq!(words(&sumcheck_core.0), expected);
        assert_eq!(sumcheck_core.0.len(), 24);

        let native_registry = BoundaryOwnerRegistryV2::new(Vec::new()).unwrap();
        let mut sumcheck_native = RecordingChallenger::default();
        observe_program_global_metadata_v2(
            &mut sumcheck_native,
            F::zero(),
            &boundary,
            &native_registry,
        )
        .unwrap();
        assert!(sumcheck_native.0.is_empty());
    }

    #[test]
    fn claim_admission_rejects_tampered_presence_and_count() {
        let identity = scaled_infinity(1);
        let honest = claim(identity, identity, 4);
        assert_eq!(validate_global_claim(&honest, true), Ok(()));

        let mut bad = honest;
        bad.has_global_opening = F::from_canonical_u8(2);
        assert!(matches!(
            validate_global_claim(&bad, true),
            Err(GlobalBoundaryError::InvalidClaimPresence(2))
        ));
        bad = honest;
        bad.count = F::zero();
        assert_eq!(validate_global_claim(&bad, true), Err(GlobalBoundaryError::ClaimCountZero));
    }

    #[test]
    fn absent_claim_is_exact_empty_interval() {
        let honest = empty_global_claim();
        assert_eq!(validate_global_claim(&honest, false), Ok(()));
        let mut bad = honest;
        bad.interval.end = scaled_infinity(2);
        assert_eq!(
            validate_global_claim(&bad, false),
            Err(GlobalBoundaryError::NonCanonicalAbsentClaim)
        );
        let mapped = crate::global_d11::fixed_padding_dummy::<F>();
        let running = state_from_projective(
            &D11AffinePointV1 { x: mapped.packed_x, y: mapped.signed_y }.to_projective(),
        );
        let carried = GlobalClaim {
            has_global_opening: F::zero(),
            count: F::zero(),
            interval: GlobalStateInterval { start: running, end: running },
        };
        assert_eq!(validate_global_claim(&carried, false), Ok(()));
    }

    #[test]
    fn two_and_three_child_intervals_compose_exactly() {
        let program = ProgramImageBoundaryV1::Infinity;
        let identity = scaled_infinity(1);
        let mapped = crate::global_d11::fixed_padding_dummy::<F>();
        let p_point = D11AffinePointV1 { x: mapped.packed_x, y: mapped.signed_y }.to_projective();
        let p = state_from_projective(&p_point);
        let two = [
            claim(identity, p, 1),
            claim(p, identity, 1),
        ];
        assert_eq!(verify_global_interval_root_v4(&program, &two), Ok(()));

        let q = state_from_projective(&p_point.negated());
        let three = [
            claim(identity, p, 1),
            claim(p, q, 1),
            claim(q, identity, 1),
        ];
        assert_eq!(verify_global_interval_root_v4(&program, &three), Ok(()));
    }

    #[test]
    fn root_binds_program_seed_and_checks_exact_identity() {
        let mapped = crate::global_d11::fixed_padding_dummy::<F>();
        let p = D11AffinePointV1 { x: mapped.packed_x, y: mapped.signed_y };
        let program = ProgramImageBoundaryV1::Affine {
            x: p.x.to_canonical_u32(),
            y: p.y.to_canonical_u32(),
        };
        let identity = scaled_infinity(1);
        let start = state_from_projective(&p.to_projective());
        let interval = claim(start, identity, 1);
        assert_eq!(verify_global_interval_root_v4(&program, &[interval]), Ok(()));
        assert_eq!(
            verify_global_interval_root_v4(&ProgramImageBoundaryV1::Infinity, &[interval]),
            Err(GlobalBoundaryError::FirstSeedMismatch)
        );
    }
}
