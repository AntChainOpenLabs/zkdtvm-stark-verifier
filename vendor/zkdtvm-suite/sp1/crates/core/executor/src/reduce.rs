use std::io::{Cursor, Read};

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
    de::{DeserializeOwned, Error as DeError},
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
}
