use std::{
    fmt,
    io::{Cursor, Read},
    marker::PhantomData,
};

use bincode::Options;
use dt_stark::{
    sumcheck::{
        config::{MlCom, MlPcsOpeningProof, SCStarkGenericConfig},
        keys::SCStarkVerifyingKey,
        proof::{
            SCChipOpenedValues, SCShardCommitment, SCShardOpenedValues, SCShardProof, SumcheckProof,
        },
        types::UniPolyEvals,
    },
    Challenge, SCAirOpenedValues, Val,
};
use p3_field::{AbstractExtensionField, Field};
use p3_matrix::Dimensions;
use serde::{
    de::{DeserializeOwned, DeserializeSeed, Error as DeError, SeqAccess, Visitor},
    Deserialize, Deserializer, Serialize, Serializer,
};
/// An intermediate proof which proves the execution.

// v4: the wire carries the shard's own chip_ordering as a compact blob. v3
// substituted `vk.chip_ordering` on deserialize, but the vk ordering only
// covers chips with preprocessed traces (6 of 26 on the native root proof),
// so round-tripped proofs failed machine verify with
// ChipOpeningLengthMismatch.
// v5: the compressed root proof switched its PCS/transcript hash family to
// SHA256 (`RootSC`, 32-byte commitments). The SC type is not encoded in the
// wire, so v4 files (Poseidon2 field-element commitments) must fail loudly on
// the version check instead of misparsing as byte digests.
// v6: the proof-system-native Global interval replaces the old per-chip septic sidecar.
const DT_REDUCE_PROOF_WIRE_VERSION: u32 =
    dt_stark::global_d11::GLOBAL146_DT_REDUCE_PROOF_WIRE_VERSION;

fn validate_dt_reduce_wire_identity(
    format_version: u32,
    global146_identity: &[u8; 32],
) -> Result<(), String> {
    if format_version != DT_REDUCE_PROOF_WIRE_VERSION {
        return Err(format!("unsupported DTReduceProof wire version {format_version}"));
    }
    dt_stark::global_d11::validate_global146_identity(global146_identity).map_err(str::to_string)
}

/// Resource limits for decoding an untrusted, current-root [`DTReduceProof`].
///
/// The defaults deliberately leave headroom above the current root proof while
/// bounding the outer payload, public values, and compact-proof allocations.
/// Callers may tighten these values for a frozen release artifact.
#[derive(Clone, Copy, Debug)]
pub struct DTReduceProofDecodeLimits {
    /// Maximum size of the complete bincode payload.
    pub max_proof_bytes: usize,
    /// Maximum encoded size of the compact opened-values blob.
    pub max_opened_values_bytes: usize,
    /// Maximum encoded size of the compact sumcheck blob.
    pub max_sumcheck_bytes: usize,
    /// Maximum encoded size of the compact chip-ordering blob.
    pub max_chip_ordering_bytes: usize,
    /// Maximum number of public values in the outer proof.
    pub max_public_values: usize,
    /// Maximum number of chips in opened values and chip ordering.
    pub max_chips: usize,
    /// Maximum length of any one opened-values local vector.
    pub max_opened_values_per_local: usize,
    /// Maximum combined length of all opened-values local vectors.
    pub max_total_opened_values: usize,
    /// Maximum number of sumcheck univariates.
    pub max_unipolys: usize,
    /// Maximum number of evaluations in any one sumcheck univariate.
    pub max_evals_per_unipoly: usize,
    /// Maximum combined number of sumcheck evaluations.
    pub max_total_sumcheck_evals: usize,
    /// Maximum accepted trace-height logarithm.
    pub max_log_height: usize,
}

impl Default for DTReduceProofDecodeLimits {
    fn default() -> Self {
        Self {
            max_proof_bytes: 4 * 1024 * 1024,
            max_opened_values_bytes: 1024 * 1024,
            max_sumcheck_bytes: 1024 * 1024,
            max_chip_ordering_bytes: 64 * 1024,
            max_public_values: 159,
            // The frozen v0.8.0 native-root inventory contains exactly 27 chips. Keep this bound
            // exact: the artifact verifier separately checks the full ordered inventory.
            max_chips: 27,
            max_opened_values_per_local: 4096,
            max_total_opened_values: 64 * 1024,
            max_unipolys: 128,
            max_evals_per_unipoly: 64,
            max_total_sumcheck_evals: 4096,
            // The frozen current-root stacked opening has height 18. Reject larger heights at the
            // untrusted decode boundary before they can amplify dimension arithmetic or reach a
            // downstream verifier assertion.
            max_log_height: 18,
        }
    }
}

#[derive(Clone)]
pub struct DTReduceProof<SC: SCStarkGenericConfig> {
    /// The compress verifying key associated with the proof.
    pub vk: SCStarkVerifyingKey<SC>,
    /// The shard proof representing the compressed proof.
    pub proof: SCShardProof<SC>,
}

#[derive(Serialize)]
#[serde(bound(serialize = "SCStarkVerifyingKey<SC>: Serialize"))]
struct DTReduceProofWire<'a, SC: SCStarkGenericConfig> {
    format_version: u32,
    global146_identity: [u8; 32],
    vk: &'a SCStarkVerifyingKey<SC>,
    proof: CompactShardProofWire<'a, SC>,
}

#[derive(Serialize)]
#[serde(bound(
    serialize = "SCShardCommitment<MlCom<SC>>: Serialize, MlPcsOpeningProof<SC>: Serialize, Val<SC>: Serialize"
))]
struct CompactShardProofWire<'a, SC: SCStarkGenericConfig> {
    commitment: &'a SCShardCommitment<MlCom<SC>>,
    opened_values: Vec<u8>,
    opening_proof: &'a MlPcsOpeningProof<SC>,
    sumcheck_proof: Vec<u8>,
    chip_ordering: Vec<u8>,
    public_values: &'a [Val<SC>],
}

#[derive(Deserialize)]
#[serde(bound(deserialize = "SCStarkVerifyingKey<SC>: Deserialize<'de>"))]
struct DTReduceProofWireOwned<SC: SCStarkGenericConfig> {
    format_version: u32,
    global146_identity: [u8; 32],
    vk: SCStarkVerifyingKey<SC>,
    proof: CompactShardProofWireOwned<SC>,
}

#[derive(Deserialize)]
#[serde(bound(
    deserialize = "SCShardCommitment<MlCom<SC>>: Deserialize<'de>, MlPcsOpeningProof<SC>: Deserialize<'de>, Val<SC>: Deserialize<'de>"
))]
struct CompactShardProofWireOwned<SC: SCStarkGenericConfig> {
    commitment: SCShardCommitment<MlCom<SC>>,
    opened_values: Vec<u8>,
    opening_proof: MlPcsOpeningProof<SC>,
    sumcheck_proof: Vec<u8>,
    chip_ordering: Vec<u8>,
    public_values: Vec<Val<SC>>,
}

struct DTReduceProofWireBounded<'de, SC: SCStarkGenericConfig> {
    format_version: u32,
    global146_identity: [u8; 32],
    vk: SCStarkVerifyingKey<SC>,
    proof: CompactShardProofWireBounded<'de, SC>,
}

struct CompactShardProofWireBounded<'de, SC: SCStarkGenericConfig> {
    commitment: SCShardCommitment<MlCom<SC>>,
    opened_values: &'de [u8],
    opening_proof: MlPcsOpeningProof<SC>,
    sumcheck_proof: &'de [u8],
    chip_ordering: &'de [u8],
    public_values: Vec<Val<SC>>,
}

struct DTReduceProofWireSeed<SC> {
    limits: DTReduceProofDecodeLimits,
    _marker: PhantomData<SC>,
}

impl<'de, SC> DeserializeSeed<'de> for DTReduceProofWireSeed<SC>
where
    SC: SCStarkGenericConfig,
    SCStarkVerifyingKey<SC>: Deserialize<'de>,
    SCShardCommitment<MlCom<SC>>: Deserialize<'de>,
    MlPcsOpeningProof<SC>: Deserialize<'de>,
    Val<SC>: Deserialize<'de>,
{
    type Value = DTReduceProofWireBounded<'de, SC>;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        const FIELDS: &[&str] = &["format_version", "global146_identity", "vk", "proof"];
        deserializer.deserialize_struct(
            "DTReduceProofWire",
            FIELDS,
            DTReduceProofWireVisitor { limits: self.limits, _marker: PhantomData },
        )
    }
}

struct DTReduceProofWireVisitor<SC> {
    limits: DTReduceProofDecodeLimits,
    _marker: PhantomData<SC>,
}

impl<'de, SC> Visitor<'de> for DTReduceProofWireVisitor<SC>
where
    SC: SCStarkGenericConfig,
    SCStarkVerifyingKey<SC>: Deserialize<'de>,
    SCShardCommitment<MlCom<SC>>: Deserialize<'de>,
    MlPcsOpeningProof<SC>: Deserialize<'de>,
    Val<SC>: Deserialize<'de>,
{
    type Value = DTReduceProofWireBounded<'de, SC>;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a current DTReduceProof wire")
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let format_version =
            seq.next_element()?.ok_or_else(|| A::Error::invalid_length(0, &self))?;
        let global146_identity =
            seq.next_element()?.ok_or_else(|| A::Error::invalid_length(1, &self))?;
        validate_dt_reduce_wire_identity(format_version, &global146_identity)
            .map_err(A::Error::custom)?;

        let vk = seq.next_element()?.ok_or_else(|| A::Error::invalid_length(2, &self))?;
        let proof = seq
            .next_element_seed(CompactShardProofWireSeed::<SC> {
                limits: self.limits,
                _marker: PhantomData,
            })?
            .ok_or_else(|| A::Error::invalid_length(3, &self))?;

        Ok(DTReduceProofWireBounded { format_version, global146_identity, vk, proof })
    }
}

struct CompactShardProofWireSeed<SC> {
    limits: DTReduceProofDecodeLimits,
    _marker: PhantomData<SC>,
}

impl<'de, SC> DeserializeSeed<'de> for CompactShardProofWireSeed<SC>
where
    SC: SCStarkGenericConfig,
    SCShardCommitment<MlCom<SC>>: Deserialize<'de>,
    MlPcsOpeningProof<SC>: Deserialize<'de>,
    Val<SC>: Deserialize<'de>,
{
    type Value = CompactShardProofWireBounded<'de, SC>;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "commitment",
            "opened_values",
            "opening_proof",
            "sumcheck_proof",
            "chip_ordering",
            "public_values",
        ];
        deserializer.deserialize_struct(
            "CompactShardProofWire",
            FIELDS,
            CompactShardProofWireVisitor { limits: self.limits, _marker: PhantomData },
        )
    }
}

struct CompactShardProofWireVisitor<SC> {
    limits: DTReduceProofDecodeLimits,
    _marker: PhantomData<SC>,
}

impl<'de, SC> Visitor<'de> for CompactShardProofWireVisitor<SC>
where
    SC: SCStarkGenericConfig,
    SCShardCommitment<MlCom<SC>>: Deserialize<'de>,
    MlPcsOpeningProof<SC>: Deserialize<'de>,
    Val<SC>: Deserialize<'de>,
{
    type Value = CompactShardProofWireBounded<'de, SC>;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a compact current-root shard proof")
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let commitment = seq.next_element()?.ok_or_else(|| A::Error::invalid_length(0, &self))?;
        let opened_values: &'de [u8] =
            seq.next_element()?.ok_or_else(|| A::Error::invalid_length(1, &self))?;
        ensure_at_most::<A::Error>(
            "opened-values blob bytes",
            opened_values.len(),
            self.limits.max_opened_values_bytes,
        )?;

        let opening_proof =
            seq.next_element()?.ok_or_else(|| A::Error::invalid_length(2, &self))?;
        let sumcheck_proof: &'de [u8] =
            seq.next_element()?.ok_or_else(|| A::Error::invalid_length(3, &self))?;
        ensure_at_most::<A::Error>(
            "sumcheck blob bytes",
            sumcheck_proof.len(),
            self.limits.max_sumcheck_bytes,
        )?;

        let chip_ordering: &'de [u8] =
            seq.next_element()?.ok_or_else(|| A::Error::invalid_length(4, &self))?;
        ensure_at_most::<A::Error>(
            "chip-ordering blob bytes",
            chip_ordering.len(),
            self.limits.max_chip_ordering_bytes,
        )?;

        let public_values = seq
            .next_element_seed(BoundedVecSeed::<Val<SC>> {
                label: "public values",
                max_len: self.limits.max_public_values,
                _marker: PhantomData,
            })?
            .ok_or_else(|| A::Error::invalid_length(5, &self))?;

        Ok(CompactShardProofWireBounded {
            commitment,
            opened_values,
            opening_proof,
            sumcheck_proof,
            chip_ordering,
            public_values,
        })
    }
}

struct BoundedVecSeed<T> {
    label: &'static str,
    max_len: usize,
    _marker: PhantomData<T>,
}

impl<'de, T> DeserializeSeed<'de> for BoundedVecSeed<T>
where
    T: Deserialize<'de>,
{
    type Value = Vec<T>;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_seq(BoundedVecVisitor::<T> {
            label: self.label,
            max_len: self.max_len,
            _marker: PhantomData,
        })
    }
}

struct BoundedVecVisitor<T> {
    label: &'static str,
    max_len: usize,
    _marker: PhantomData<T>,
}

impl<'de, T> Visitor<'de> for BoundedVecVisitor<T>
where
    T: Deserialize<'de>,
{
    type Value = Vec<T>;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "at most {} {}", self.max_len, self.label)
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let initial_len = seq.size_hint().unwrap_or(0);
        ensure_at_most::<A::Error>(self.label, initial_len, self.max_len)?;

        let mut values = Vec::new();
        values.try_reserve_exact(initial_len).map_err(|err| {
            A::Error::custom(format!("failed to reserve {} {}: {err}", initial_len, self.label))
        })?;
        while let Some(value) = seq.next_element()? {
            if values.len() == self.max_len {
                return Err(A::Error::custom(format!(
                    "{} exceeds limit {}",
                    self.label, self.max_len
                )));
            }
            if values.len() == values.capacity() {
                values.try_reserve_exact(1).map_err(|err| {
                    A::Error::custom(format!("failed to grow {}: {err}", self.label))
                })?;
            }
            values.push(value);
        }
        Ok(values)
    }
}

fn ensure_at_most<E: DeError>(label: &str, actual: usize, maximum: usize) -> Result<(), E> {
    if actual <= maximum {
        Ok(())
    } else {
        Err(E::custom(format!("{label} {actual} exceeds limit {maximum}")))
    }
}

impl<SC> Serialize for DTReduceProof<SC>
where
    SC: SCStarkGenericConfig,
    SCStarkVerifyingKey<SC>: Serialize,
    SCShardCommitment<MlCom<SC>>: Serialize,
    MlPcsOpeningProof<SC>: Serialize,
    Val<SC>: Field + Serialize,
    Challenge<SC>: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let wire = DTReduceProofWire {
            format_version: DT_REDUCE_PROOF_WIRE_VERSION,
            global146_identity: dt_stark::global_d11::GLOBAL146_COMPOSITE_IDENTITY,
            vk: &self.vk,
            proof: CompactShardProofWire {
                commitment: &self.proof.commitment,
                opened_values: encode_opened_values::<SC>(&self.proof.opened_values)
                    .map_err(serde::ser::Error::custom)?,
                opening_proof: &self.proof.opening_proof,
                sumcheck_proof: encode_sumcheck_proof::<SC>(&self.proof.sumcheck_proof)
                    .map_err(serde::ser::Error::custom)?,
                chip_ordering: encode_chip_ordering(&self.proof.chip_ordering)
                    .map_err(serde::ser::Error::custom)?,
                public_values: &self.proof.public_values,
            },
        };
        wire.serialize(serializer)
    }
}

impl<'de, SC> Deserialize<'de> for DTReduceProof<SC>
where
    SC: SCStarkGenericConfig,
    SCStarkVerifyingKey<SC>: Deserialize<'de>,
    SCShardCommitment<MlCom<SC>>: Deserialize<'de>,
    MlPcsOpeningProof<SC>: Deserialize<'de>,
    Val<SC>: Field + DeserializeOwned,
    Challenge<SC>: AbstractExtensionField<Val<SC>> + DeserializeOwned,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = DTReduceProofWireOwned::<SC>::deserialize(deserializer)?;
        validate_dt_reduce_wire_identity(wire.format_version, &wire.global146_identity)
            .map_err(D::Error::custom)?;

        let opened_values =
            decode_opened_values::<Val<SC>, Challenge<SC>>(&wire.proof.opened_values)
                .map_err(D::Error::custom)?;
        let sumcheck_proof =
            decode_sumcheck_proof::<SC>(&wire.proof.sumcheck_proof).map_err(D::Error::custom)?;
        let dimensions = reconstruct_dimensions::<SC>(&wire.proof.commitment, &opened_values);
        let chip_ordering =
            decode_chip_ordering(&wire.proof.chip_ordering).map_err(D::Error::custom)?;

        let proof = SCShardProof {
            commitment: wire.proof.commitment,
            opened_values,
            opening_proof: wire.proof.opening_proof,
            sumcheck_proof,
            dimensions,
            chip_ordering,
            public_values: wire.proof.public_values,
        };

        Ok(Self { vk: wire.vk, proof })
    }
}

/// Deserializes the legacy bincode-1.3 proof wire with current-root resource limits.
///
/// Unlike the general-purpose [`Deserialize`] implementation, this entry point is
/// suitable for untrusted byte input: it gates the fixed wire header before Serde,
/// rejects trailing bytes, borrows the compact blobs instead of copying them, and
/// bounds every compact allocation before reserving it.
///
/// `MlPcsOpeningProof<SC>` remains an opaque associated type and therefore uses
/// its existing Serde implementation. Its decode is contained by
/// [`DTReduceProofDecodeLimits::max_proof_bytes`] and Serde's cautious initial
/// collection allocation, but callers must still apply the frozen release's
/// post-decode opening-shape checks and keep fuzz/memory regression gates.
pub fn deserialize_reduce_proof_bounded<SC>(bytes: &[u8]) -> bincode::Result<DTReduceProof<SC>>
where
    SC: SCStarkGenericConfig,
    SCStarkVerifyingKey<SC>: DeserializeOwned,
    SCShardCommitment<MlCom<SC>>: DeserializeOwned,
    MlPcsOpeningProof<SC>: DeserializeOwned,
    Val<SC>: Field + DeserializeOwned,
    Challenge<SC>: AbstractExtensionField<Val<SC>> + DeserializeOwned,
{
    deserialize_reduce_proof_bounded_with_limits(bytes, DTReduceProofDecodeLimits::default())
}

/// Equivalent to [`deserialize_reduce_proof_bounded`], with caller-supplied limits.
pub fn deserialize_reduce_proof_bounded_with_limits<SC>(
    bytes: &[u8],
    limits: DTReduceProofDecodeLimits,
) -> bincode::Result<DTReduceProof<SC>>
where
    SC: SCStarkGenericConfig,
    SCStarkVerifyingKey<SC>: DeserializeOwned,
    SCShardCommitment<MlCom<SC>>: DeserializeOwned,
    MlPcsOpeningProof<SC>: DeserializeOwned,
    Val<SC>: Field + DeserializeOwned,
    Challenge<SC>: AbstractExtensionField<Val<SC>> + DeserializeOwned,
{
    validate_bounded_reduce_proof_header(bytes, limits.max_proof_bytes)?;

    // `bincode::serialize` uses this legacy little-endian/fixint wire. A slice
    // length is checked above because bincode 1.3 intentionally replaces a
    // configured limit with `Infinite` in its slice-deserialization path.
    let wire = bincode::DefaultOptions::new()
        .with_little_endian()
        .with_fixint_encoding()
        .reject_trailing_bytes()
        .deserialize_seed(DTReduceProofWireSeed::<SC> { limits, _marker: PhantomData }, bytes)?;

    let DTReduceProofWireBounded { format_version, global146_identity, vk, proof } = wire;
    validate_dt_reduce_wire_identity(format_version, &global146_identity)
        .map_err(bounded_decode_error)?;

    let opened_values =
        decode_opened_values_bounded::<Val<SC>, Challenge<SC>>(proof.opened_values, limits)?;
    let sumcheck_proof = decode_sumcheck_proof_bounded::<SC>(proof.sumcheck_proof, limits)?;
    let chip_ordering = decode_chip_ordering_bounded(proof.chip_ordering, limits)?;
    if chip_ordering.len() != opened_values.chips.len() {
        return Err(bounded_decode_error(format!(
            "chip ordering has {} entries but opened values has {} chips",
            chip_ordering.len(),
            opened_values.chips.len()
        )));
    }
    let dimensions =
        reconstruct_dimensions_bounded::<SC>(&proof.commitment, &opened_values, limits)?;

    Ok(DTReduceProof {
        vk,
        proof: SCShardProof {
            commitment: proof.commitment,
            opened_values,
            opening_proof: proof.opening_proof,
            sumcheck_proof,
            dimensions,
            chip_ordering,
            public_values: proof.public_values,
        },
    })
}

const DT_REDUCE_PROOF_HEADER_BYTES: usize = 4 + 32;

fn validate_bounded_reduce_proof_header(
    bytes: &[u8],
    max_proof_bytes: usize,
) -> bincode::Result<()> {
    if bytes.len() > max_proof_bytes {
        return Err(bounded_decode_error(format!(
            "DTReduceProof bytes {} exceeds limit {max_proof_bytes}",
            bytes.len()
        )));
    }
    if bytes.len() < DT_REDUCE_PROOF_HEADER_BYTES {
        return Err(bounded_decode_error(format!(
            "truncated DTReduceProof header: got {} bytes, need {DT_REDUCE_PROOF_HEADER_BYTES}",
            bytes.len()
        )));
    }

    let mut version_bytes = [0u8; 4];
    version_bytes.copy_from_slice(&bytes[..4]);
    let format_version = u32::from_le_bytes(version_bytes);
    let mut identity = [0u8; 32];
    identity.copy_from_slice(&bytes[4..DT_REDUCE_PROOF_HEADER_BYTES]);
    validate_dt_reduce_wire_identity(format_version, &identity).map_err(bounded_decode_error)
}

fn bounded_decode_error(message: impl Into<String>) -> Box<bincode::ErrorKind> {
    Box::new(bincode::ErrorKind::Custom(message.into()))
}

fn encode_opened_values<SC>(
    opened_values: &SCShardOpenedValues<Val<SC>, Challenge<SC>>,
) -> bincode::Result<Vec<u8>>
where
    SC: SCStarkGenericConfig,
    Val<SC>: Field + Serialize,
    Challenge<SC>: Serialize,
{
    let mut out = Vec::new();
    put_len(&mut out, opened_values.chips.len())?;

    for chip in &opened_values.chips {
        put_slice(&mut out, &chip.preprocessed.local)?;
        put_slice(&mut out, &chip.main.local)?;
        put_slice(&mut out, &chip.permutation.local)?;
        put_value(&mut out, &chip.local_cumulative_sum)?;
        put_len(&mut out, chip.log_height)?;
    }

    Ok(out)
}

fn decode_opened_values<F, EF>(bytes: &[u8]) -> bincode::Result<SCShardOpenedValues<F, EF>>
where
    F: Field + DeserializeOwned,
    EF: DeserializeOwned,
{
    let mut cursor = Cursor::new(bytes);
    let chip_count = read_len(&mut cursor)?;
    let mut chips = Vec::with_capacity(chip_count);

    for _ in 0..chip_count {
        let preprocessed = SCAirOpenedValues { local: read_vec(&mut cursor)? };
        let main = SCAirOpenedValues { local: read_vec(&mut cursor)? };
        let permutation = SCAirOpenedValues { local: read_vec(&mut cursor)? };
        let local_cumulative_sum = read_value(&mut cursor)?;
        let log_height = read_len(&mut cursor)?;
        chips.push(SCChipOpenedValues {
            preprocessed,
            main,
            permutation,
            local_cumulative_sum,
            log_height,
            _field: core::marker::PhantomData,
        });
    }
    ensure_finished(&cursor)?;

    Ok(SCShardOpenedValues { chips, _field: core::marker::PhantomData })
}

fn decode_opened_values_bounded<F, EF>(
    bytes: &[u8],
    limits: DTReduceProofDecodeLimits,
) -> bincode::Result<SCShardOpenedValues<F, EF>>
where
    F: Field + DeserializeOwned,
    EF: DeserializeOwned,
{
    ensure_bincode_at_most(
        "opened-values blob bytes",
        bytes.len(),
        limits.max_opened_values_bytes,
    )?;
    let mut cursor = Cursor::new(bytes);
    let chip_count = read_len_bounded(&mut cursor, limits.max_chips, "opened-values chips")?;
    let mut chips = Vec::new();
    chips.try_reserve_exact(chip_count).map_err(|err| {
        bounded_decode_error(format!("failed to reserve {chip_count} opened-values chips: {err}"))
    })?;
    let mut total_opened_values = 0usize;

    for _ in 0..chip_count {
        let preprocessed = SCAirOpenedValues {
            local: read_opened_local_bounded::<EF>(
                &mut cursor,
                limits,
                &mut total_opened_values,
                "preprocessed local values",
            )?,
        };
        let main = SCAirOpenedValues {
            local: read_opened_local_bounded::<EF>(
                &mut cursor,
                limits,
                &mut total_opened_values,
                "main local values",
            )?,
        };
        let permutation = SCAirOpenedValues {
            local: read_opened_local_bounded::<EF>(
                &mut cursor,
                limits,
                &mut total_opened_values,
                "permutation local values",
            )?,
        };
        let local_cumulative_sum = read_value(&mut cursor)?;
        let log_height =
            read_len_bounded(&mut cursor, limits.max_log_height, "opened-values log_height")?;
        checked_height_from_log(log_height, limits.max_log_height)?;
        chips.push(SCChipOpenedValues {
            preprocessed,
            main,
            permutation,
            local_cumulative_sum,
            log_height,
            _field: core::marker::PhantomData,
        });
    }
    ensure_finished(&cursor)?;

    Ok(SCShardOpenedValues { chips, _field: core::marker::PhantomData })
}

fn read_opened_local_bounded<T: DeserializeOwned>(
    cursor: &mut Cursor<&[u8]>,
    limits: DTReduceProofDecodeLimits,
    total: &mut usize,
    label: &'static str,
) -> bincode::Result<Vec<T>> {
    let remaining = limits
        .max_total_opened_values
        .checked_sub(*total)
        .ok_or_else(|| bounded_decode_error("opened-values total length overflow"))?;
    let values =
        read_vec_bounded(cursor, limits.max_opened_values_per_local.min(remaining), label)?;
    *total = (*total)
        .checked_add(values.len())
        .ok_or_else(|| bounded_decode_error("opened-values total length overflow"))?;
    Ok(values)
}

/// Compact chip-ordering blob: u32 count, then per chip (in index order) a
/// u8 name length + the name bytes. The shard's ordering is a bijection
/// name → 0..count, so index-ordered names reconstruct it exactly.
fn encode_chip_ordering(
    chip_ordering: &hashbrown::HashMap<String, usize>,
) -> bincode::Result<Vec<u8>> {
    let mut names: Vec<Option<&String>> = vec![None; chip_ordering.len()];
    for (name, &idx) in chip_ordering {
        let slot = names.get_mut(idx).ok_or_else(|| {
            Box::new(bincode::ErrorKind::Custom(format!(
                "chip_ordering index {idx} out of range for {} chips",
                chip_ordering.len()
            )))
        })?;
        if slot.is_some() {
            return Err(Box::new(bincode::ErrorKind::Custom(format!(
                "duplicate chip_ordering index {idx}"
            ))));
        }
        *slot = Some(name);
    }

    let mut out = Vec::new();
    put_len(&mut out, names.len())?;
    for (idx, name) in names.into_iter().enumerate() {
        let name = name.ok_or_else(|| {
            Box::new(bincode::ErrorKind::Custom(format!("chip_ordering index {idx} unassigned")))
        })?;
        let len = u8::try_from(name.len()).map_err(|_| {
            Box::new(bincode::ErrorKind::Custom(format!("chip name longer than 255 bytes: {name}")))
        })?;
        out.push(len);
        out.extend_from_slice(name.as_bytes());
    }
    Ok(out)
}

fn decode_chip_ordering(bytes: &[u8]) -> bincode::Result<hashbrown::HashMap<String, usize>> {
    let mut cursor = Cursor::new(bytes);
    let count = read_len(&mut cursor)?;
    let mut chip_ordering = hashbrown::HashMap::with_capacity(count);
    for idx in 0..count {
        let len = read_u8(&mut cursor)? as usize;
        let mut name = vec![0u8; len];
        cursor.read_exact(&mut name).map_err(|err| {
            Box::new(bincode::ErrorKind::Custom(format!("read chip name: {err}")))
        })?;
        let name = String::from_utf8(name).map_err(|err| {
            Box::new(bincode::ErrorKind::Custom(format!("chip name not utf-8: {err}")))
        })?;
        if chip_ordering.insert(name, idx).is_some() {
            return Err(Box::new(bincode::ErrorKind::Custom(
                "duplicate chip name in chip_ordering".to_string(),
            )));
        }
    }
    ensure_finished(&cursor)?;
    Ok(chip_ordering)
}

fn decode_chip_ordering_bounded(
    bytes: &[u8],
    limits: DTReduceProofDecodeLimits,
) -> bincode::Result<hashbrown::HashMap<String, usize>> {
    ensure_bincode_at_most(
        "chip-ordering blob bytes",
        bytes.len(),
        limits.max_chip_ordering_bytes,
    )?;
    let mut cursor = Cursor::new(bytes);
    let count = read_len_bounded(&mut cursor, limits.max_chips, "chip-ordering entries")?;
    let mut chip_ordering = hashbrown::HashMap::new();
    chip_ordering.try_reserve(count).map_err(|err| {
        bounded_decode_error(format!("failed to reserve {count} chip-ordering entries: {err:?}"))
    })?;
    for idx in 0..count {
        let len = read_u8(&mut cursor)? as usize;
        let mut name = Vec::new();
        name.try_reserve_exact(len).map_err(|err| {
            bounded_decode_error(format!("failed to reserve {len} chip-name bytes: {err}"))
        })?;
        name.resize(len, 0);
        cursor
            .read_exact(&mut name)
            .map_err(|err| bounded_decode_error(format!("read chip name: {err}")))?;
        let name = String::from_utf8(name)
            .map_err(|err| bounded_decode_error(format!("chip name not utf-8: {err}")))?;
        if chip_ordering.insert(name, idx).is_some() {
            return Err(bounded_decode_error("duplicate chip name in chip_ordering"));
        }
    }
    ensure_finished(&cursor)?;
    Ok(chip_ordering)
}

fn encode_sumcheck_proof<SC>(proof: &SumcheckProof<SC>) -> bincode::Result<Vec<u8>>
where
    SC: SCStarkGenericConfig,
    Challenge<SC>: Serialize,
{
    let mut out = Vec::new();
    put_len(&mut out, proof.unipolys.len())?;
    for unipoly in &proof.unipolys {
        put_slice(&mut out, &unipoly.evals)?;
    }
    Ok(out)
}

fn decode_sumcheck_proof<SC>(bytes: &[u8]) -> bincode::Result<SumcheckProof<SC>>
where
    SC: SCStarkGenericConfig,
    Challenge<SC>: DeserializeOwned,
{
    let mut cursor = Cursor::new(bytes);
    let count = read_len(&mut cursor)?;
    let mut unipolys = Vec::with_capacity(count);
    for _ in 0..count {
        unipolys.push(UniPolyEvals { evals: read_vec(&mut cursor)? });
    }
    ensure_finished(&cursor)?;
    Ok(SumcheckProof { unipolys })
}

fn decode_sumcheck_proof_bounded<SC>(
    bytes: &[u8],
    limits: DTReduceProofDecodeLimits,
) -> bincode::Result<SumcheckProof<SC>>
where
    SC: SCStarkGenericConfig,
    Challenge<SC>: DeserializeOwned,
{
    ensure_bincode_at_most("sumcheck blob bytes", bytes.len(), limits.max_sumcheck_bytes)?;
    let mut cursor = Cursor::new(bytes);
    let count = read_len_bounded(&mut cursor, limits.max_unipolys, "sumcheck unipolys")?;
    let mut unipolys = Vec::new();
    unipolys.try_reserve_exact(count).map_err(|err| {
        bounded_decode_error(format!("failed to reserve {count} sumcheck unipolys: {err}"))
    })?;
    let mut total_evals = 0usize;
    for _ in 0..count {
        let remaining = limits
            .max_total_sumcheck_evals
            .checked_sub(total_evals)
            .ok_or_else(|| bounded_decode_error("sumcheck evaluation total length overflow"))?;
        let evals = read_vec_bounded(
            &mut cursor,
            limits.max_evals_per_unipoly.min(remaining),
            "sumcheck unipoly evaluations",
        )?;
        total_evals = total_evals
            .checked_add(evals.len())
            .ok_or_else(|| bounded_decode_error("sumcheck evaluation total length overflow"))?;
        unipolys.push(UniPolyEvals { evals });
    }
    ensure_finished(&cursor)?;
    Ok(SumcheckProof { unipolys })
}

fn reconstruct_dimensions<SC>(
    commitment: &SCShardCommitment<MlCom<SC>>,
    opened_values: &SCShardOpenedValues<Val<SC>, Challenge<SC>>,
) -> Vec<Vec<Dimensions>>
where
    SC: SCStarkGenericConfig,
    Challenge<SC>: AbstractExtensionField<Val<SC>>,
{
    let prep_dims = opened_values
        .chips
        .iter()
        .filter(|chip| !chip.preprocessed.local.is_empty())
        .map(|chip| Dimensions {
            width: chip.preprocessed.local.len(),
            height: 1usize << chip.log_height,
        })
        .collect::<Vec<_>>();
    let main_dims = opened_values
        .chips
        .iter()
        .map(|chip| Dimensions { width: chip.main.local.len(), height: 1usize << chip.log_height })
        .collect::<Vec<_>>();

    let mut dimensions =
        Vec::with_capacity(if commitment.permutation_commit.is_some() { 3 } else { 2 });
    dimensions.push(prep_dims);
    dimensions.push(main_dims);

    if commitment.permutation_commit.is_some() {
        dimensions.push(
            opened_values
                .chips
                .iter()
                .map(|chip| Dimensions {
                    width: chip.permutation.local.len() *
                        <Challenge<SC> as AbstractExtensionField<Val<SC>>>::D,
                    height: 1usize << chip.log_height,
                })
                .collect(),
        );
    }

    dimensions
}

fn reconstruct_dimensions_bounded<SC>(
    commitment: &SCShardCommitment<MlCom<SC>>,
    opened_values: &SCShardOpenedValues<Val<SC>, Challenge<SC>>,
    limits: DTReduceProofDecodeLimits,
) -> bincode::Result<Vec<Vec<Dimensions>>>
where
    SC: SCStarkGenericConfig,
    Challenge<SC>: AbstractExtensionField<Val<SC>>,
{
    ensure_bincode_at_most("opened-values chips", opened_values.chips.len(), limits.max_chips)?;

    let mut prep_dims = Vec::new();
    prep_dims.try_reserve_exact(opened_values.chips.len()).map_err(|err| {
        bounded_decode_error(format!("failed to reserve preprocessed dimensions: {err}"))
    })?;
    let mut main_dims = Vec::new();
    main_dims
        .try_reserve_exact(opened_values.chips.len())
        .map_err(|err| bounded_decode_error(format!("failed to reserve main dimensions: {err}")))?;
    for chip in &opened_values.chips {
        let height = checked_height_from_log(chip.log_height, limits.max_log_height)?;
        if !chip.preprocessed.local.is_empty() {
            prep_dims.push(Dimensions { width: chip.preprocessed.local.len(), height });
        }
        main_dims.push(Dimensions { width: chip.main.local.len(), height });
    }

    let mut dimensions = Vec::new();
    dimensions.try_reserve_exact(3).map_err(|err| {
        bounded_decode_error(format!("failed to reserve dimension groups: {err}"))
    })?;
    dimensions.push(prep_dims);
    dimensions.push(main_dims);

    if commitment.permutation_commit.is_some() {
        let mut permutation_dims = Vec::new();
        permutation_dims.try_reserve_exact(opened_values.chips.len()).map_err(|err| {
            bounded_decode_error(format!("failed to reserve permutation dimensions: {err}"))
        })?;
        for chip in &opened_values.chips {
            let width = chip
                .permutation
                .local
                .len()
                .checked_mul(<Challenge<SC> as AbstractExtensionField<Val<SC>>>::D)
                .ok_or_else(|| bounded_decode_error("permutation dimension width overflow"))?;
            let height = checked_height_from_log(chip.log_height, limits.max_log_height)?;
            permutation_dims.push(Dimensions { width, height });
        }
        dimensions.push(permutation_dims);
    }

    Ok(dimensions)
}

fn put_len(out: &mut Vec<u8>, value: usize) -> bincode::Result<()> {
    let value = u32::try_from(value)
        .map_err(|_| Box::new(bincode::ErrorKind::Custom(format!("length {value} exceeds u32"))))?;
    out.extend_from_slice(&value.to_le_bytes());
    Ok(())
}

fn put_slice<T: Serialize>(out: &mut Vec<u8>, values: &[T]) -> bincode::Result<()> {
    put_len(out, values.len())?;
    for value in values {
        put_value(out, value)?;
    }
    Ok(())
}

fn put_value<T: Serialize>(out: &mut Vec<u8>, value: &T) -> bincode::Result<()> {
    bincode::serialize_into(&mut *out, value)
}

fn read_len(cursor: &mut Cursor<&[u8]>) -> bincode::Result<usize> {
    let mut bytes = [0u8; 4];
    cursor
        .read_exact(&mut bytes)
        .map_err(|err| Box::new(bincode::ErrorKind::Custom(format!("read u32 length: {err}"))))?;
    Ok(u32::from_le_bytes(bytes) as usize)
}

fn read_len_bounded(
    cursor: &mut Cursor<&[u8]>,
    maximum: usize,
    label: &str,
) -> bincode::Result<usize> {
    let len = read_len(cursor)?;
    ensure_bincode_at_most(label, len, maximum)?;
    Ok(len)
}

fn read_u8(cursor: &mut Cursor<&[u8]>) -> bincode::Result<u8> {
    let mut byte = [0u8; 1];
    cursor
        .read_exact(&mut byte)
        .map_err(|err| Box::new(bincode::ErrorKind::Custom(format!("read u8: {err}"))))?;
    Ok(byte[0])
}

fn read_vec<T: DeserializeOwned>(cursor: &mut Cursor<&[u8]>) -> bincode::Result<Vec<T>> {
    let len = read_len(cursor)?;
    let mut values = Vec::with_capacity(len);
    for _ in 0..len {
        values.push(read_value(cursor)?);
    }
    Ok(values)
}

fn read_vec_bounded<T: DeserializeOwned>(
    cursor: &mut Cursor<&[u8]>,
    maximum: usize,
    label: &str,
) -> bincode::Result<Vec<T>> {
    let len = read_len_bounded(cursor, maximum, label)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|err| bounded_decode_error(format!("failed to reserve {len} {label}: {err}")))?;
    for _ in 0..len {
        values.push(read_value(cursor)?);
    }
    Ok(values)
}

fn read_value<T: DeserializeOwned>(cursor: &mut Cursor<&[u8]>) -> bincode::Result<T> {
    bincode::deserialize_from(&mut *cursor)
}

fn ensure_finished(cursor: &Cursor<&[u8]>) -> bincode::Result<()> {
    if cursor.position() == cursor.get_ref().len() as u64 {
        Ok(())
    } else {
        Err(bincode::ErrorKind::Custom(format!(
            "trailing compact proof bytes: {} of {} consumed",
            cursor.position(),
            cursor.get_ref().len()
        ))
        .into())
    }
}

fn ensure_bincode_at_most(label: &str, actual: usize, maximum: usize) -> bincode::Result<()> {
    if actual <= maximum {
        Ok(())
    } else {
        Err(bounded_decode_error(format!("{label} {actual} exceeds limit {maximum}")))
    }
}

fn checked_height_from_log(log_height: usize, maximum: usize) -> bincode::Result<usize> {
    ensure_bincode_at_most("log_height", log_height, maximum)?;
    let shift = u32::try_from(log_height)
        .map_err(|_| bounded_decode_error(format!("log_height {log_height} does not fit u32")))?;
    1usize.checked_shl(shift).ok_or_else(|| {
        bounded_decode_error(format!("log_height {log_height} exceeds usize width {}", usize::BITS))
    })
}

#[allow(clippy::missing_fields_in_debug)]
impl<SC: SCStarkGenericConfig> std::fmt::Debug for DTReduceProof<SC> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug_struct = f.debug_struct("DTReduceProof");
        debug_struct.field("vk", &self.vk);
        debug_struct.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dt_stark::baby_bear_poseidon2::SCBabyBearPoseidon2;
    use p3_baby_bear::BabyBear;

    fn current_wire_header() -> Vec<u8> {
        let mut bytes = Vec::with_capacity(DT_REDUCE_PROOF_HEADER_BYTES);
        bytes.extend_from_slice(&DT_REDUCE_PROOF_WIRE_VERSION.to_le_bytes());
        bytes.extend_from_slice(&dt_stark::global_d11::GLOBAL146_COMPOSITE_IDENTITY);
        bytes
    }

    fn assert_error_contains<T>(result: bincode::Result<T>, needle: &str) {
        let error = result.err().expect("expected bounded decode to fail");
        assert!(error.to_string().contains(needle), "unexpected error: {error}");
    }

    #[test]
    fn reduce_wire_accepts_only_current_global146_identity() {
        assert!(validate_dt_reduce_wire_identity(
            DT_REDUCE_PROOF_WIRE_VERSION,
            &dt_stark::global_d11::GLOBAL146_COMPOSITE_IDENTITY,
        )
        .is_ok());

        let mut wrong = dt_stark::global_d11::GLOBAL146_COMPOSITE_IDENTITY;
        wrong[31] ^= 1;
        assert!(validate_dt_reduce_wire_identity(DT_REDUCE_PROOF_WIRE_VERSION, &wrong)
            .unwrap_err()
            .contains("identity mismatch"));
        assert!(validate_dt_reduce_wire_identity(
            DT_REDUCE_PROOF_WIRE_VERSION - 1,
            &dt_stark::global_d11::GLOBAL146_COMPOSITE_IDENTITY,
        )
        .unwrap_err()
        .contains("wire version"));
    }

    #[test]
    fn bounded_header_gates_size_version_and_identity_before_serde() {
        let header = current_wire_header();
        assert!(validate_bounded_reduce_proof_header(&header, header.len()).is_ok());

        assert_error_contains(
            validate_bounded_reduce_proof_header(&header[..header.len() - 1], header.len()),
            "truncated",
        );
        assert_error_contains(
            validate_bounded_reduce_proof_header(&header, header.len() - 1),
            "exceeds limit",
        );

        let mut wrong_version = header.clone();
        wrong_version[..4]
            .copy_from_slice(&DT_REDUCE_PROOF_WIRE_VERSION.wrapping_add(1).to_le_bytes());
        assert_error_contains(
            validate_bounded_reduce_proof_header(&wrong_version, wrong_version.len()),
            "wire version",
        );

        let mut wrong_identity = header;
        wrong_identity[DT_REDUCE_PROOF_HEADER_BYTES - 1] ^= 1;
        assert_error_contains(
            validate_bounded_reduce_proof_header(&wrong_identity, wrong_identity.len()),
            "identity mismatch",
        );
    }

    #[test]
    fn bounded_compact_counts_reject_u32_max_before_reserve() {
        let limits = DTReduceProofDecodeLimits::default();
        let max = u32::MAX.to_le_bytes();

        assert_error_contains(
            decode_opened_values_bounded::<BabyBear, BabyBear>(&max, limits),
            "opened-values chips",
        );
        assert_error_contains(decode_chip_ordering_bounded(&max, limits), "chip-ordering entries");
        assert_error_contains(
            decode_sumcheck_proof_bounded::<SCBabyBearPoseidon2>(&max, limits),
            "sumcheck unipolys",
        );

        let mut opened_local = Vec::new();
        opened_local.extend_from_slice(&1u32.to_le_bytes());
        opened_local.extend_from_slice(&u32::MAX.to_le_bytes());
        assert_error_contains(
            decode_opened_values_bounded::<BabyBear, BabyBear>(&opened_local, limits),
            "preprocessed local values",
        );

        let mut unipoly_evals = Vec::new();
        unipoly_evals.extend_from_slice(&1u32.to_le_bytes());
        unipoly_evals.extend_from_slice(&u32::MAX.to_le_bytes());
        assert_error_contains(
            decode_sumcheck_proof_bounded::<SCBabyBearPoseidon2>(&unipoly_evals, limits),
            "sumcheck unipoly evaluations",
        );
    }

    #[test]
    fn bounded_chip_count_matches_frozen_v080_inventory() {
        let limits = DTReduceProofDecodeLimits::default();
        assert_eq!(limits.max_chips, 27);

        let at_limit = 27u32.to_le_bytes();
        assert_eq!(
            read_len_bounded(&mut Cursor::new(at_limit.as_slice()), limits.max_chips, "chips")
                .expect("the frozen inventory must fit"),
            27,
        );

        let above_limit = 28u32.to_le_bytes();
        assert_error_contains(
            read_len_bounded(&mut Cursor::new(above_limit.as_slice()), limits.max_chips, "chips"),
            "chips 28 exceeds limit 27",
        );
    }

    #[test]
    fn bounded_outer_vector_rejects_declared_length_before_reserve() {
        let declared_len = u64::MAX.to_le_bytes();
        let result = bincode::DefaultOptions::new()
            .with_little_endian()
            .with_fixint_encoding()
            .reject_trailing_bytes()
            .deserialize_seed(
                BoundedVecSeed::<u8> {
                    label: "public values",
                    max_len: DTReduceProofDecodeLimits::default().max_public_values,
                    _marker: PhantomData,
                },
                &declared_len,
            );
        assert_error_contains(result, "public values");
    }

    #[test]
    fn bounded_legacy_bincode_rejects_trailing_bytes() {
        let mut encoded = bincode::serialize(&7u32).expect("serialize test value");
        encoded.push(0xff);
        let result = bincode::DefaultOptions::new()
            .with_little_endian()
            .with_fixint_encoding()
            .reject_trailing_bytes()
            .deserialize::<u32>(&encoded);
        assert_error_contains(result, "remaining after deserialization");

        let compact_with_trailing = [0u8, 0, 0, 0, 0xff];
        assert_error_contains(
            decode_chip_ordering_bounded(
                &compact_with_trailing,
                DTReduceProofDecodeLimits::default(),
            ),
            "trailing compact proof bytes",
        );
    }

    #[test]
    fn bounded_log_height_uses_checked_shift() {
        assert_eq!(checked_height_from_log(3, usize::MAX).expect("small shift"), 8);
        assert_error_contains(
            checked_height_from_log(usize::BITS as usize, usize::MAX),
            "usize width",
        );
        assert_error_contains(
            checked_height_from_log(32, DTReduceProofDecodeLimits::default().max_log_height),
            "exceeds limit",
        );
    }
}
