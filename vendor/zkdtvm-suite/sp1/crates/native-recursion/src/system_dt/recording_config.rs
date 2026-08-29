use dt_stark::{
    koalabear_poseidon2::{
        self,
        koala_bear_poseidon2::{
            compress_mlpcs_config, compressed_fri_config, core_mlpcs_config, default_fri_config,
            my_perm, shrink_fri_config, shrink_mlpcs_config, Challenge, ChallengeMmcs, Challenger,
            Dft, DigestHash, MyCompress, MyHash, SCKoalaBearPoseidon2, Val, ValMmcs,
        },
    },
    sumcheck::config::SCStarkGenericConfig,
    StarkGenericConfig, ZeroCommitment, DIGEST_SIZE,
};
use p3_challenger::{CanObserve, CanSample, CanSampleBits, FieldChallenger, GrindingChallenger};
use p3_field::{AbstractExtensionField, AbstractField, PrimeField64};
use p3_fri::TwoAdicFriPcs;
use pcs::basefold::{
    basefold_pcs::BaseFoldPcs,
    mlpcs::{MlCommitOptions, MlPCS},
};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::{
    config::POSEIDON2_WIDTH,
    system_dt::{RecursionRecord, RecursionRecordingChallenger, SpecSpongeBlock},
};

type RecordingPcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;
type RecordingMlpcs = BaseFoldPcs<Val, ValMmcs, ChallengeMmcs, Challenge, CoreRecordingChallenger>;

mod replay_compatible_sealed {
    pub trait Sealed {}
}

/// Proof configurations whose Fiat--Shamir and MLPCS wire types are replayed by
/// [`RecordingSC`]. The sealed marker prevents another nominal configuration from being accepted
/// merely because its associated Rust types happen to match.
pub trait ReplayCompatibleProofConfig:
    replay_compatible_sealed::Sealed
    + SCStarkGenericConfig<Val = Val, Challenge = Challenge, MlChallenge = Challenge>
{
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecordingStage {
    Core,
    Compress,
    Shrink,
}

impl RecordingStage {
    pub const fn whir_stage(self) -> &'static str {
        match self {
            Self::Core => "core",
            Self::Compress => "compress",
            Self::Shrink => "shrink",
        }
    }

    /// Child-recording stages by whir stage name. root_shrink is EXPLICITLY rejected:
    /// an L4 proof is the ladder's terminal artifact — no native machine ever
    /// verifies a root_shrink child, so recording one is a caller bug, not a config.
    pub fn from_whir_stage(stage: &str) -> Result<Self, String> {
        match stage {
            "core" => Ok(Self::Core),
            "compress" => Ok(Self::Compress),
            "shrink" => Ok(Self::Shrink),
            "root_shrink" => {
                Err("root_shrink is not a child-recording stage: L4 proofs are terminal artifacts"
                    .to_string())
            }
            other => Err(format!("unknown whir stage {other:?}")),
        }
    }

    fn fri_config(self) -> p3_fri::FriConfig<ChallengeMmcs> {
        match self {
            Self::Core => default_fri_config(),
            Self::Compress => compressed_fri_config(),
            Self::Shrink => shrink_fri_config(),
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct CoreRecordingChallenger {
    inner: RecursionRecordingChallenger<Challenger>,
}

struct PendingDuplexCapture {
    tidx: usize,
    prev_rate: [Val; 8],
    input16: [Val; POSEIDON2_WIDTH],
    absorb_count: usize,
}

impl CoreRecordingChallenger {
    pub fn from_inner(proof_idx: usize, inner: Challenger) -> Self {
        Self { inner: RecursionRecordingChallenger::from_inner(proof_idx, inner) }
    }

    pub fn fork_for_proof(&self, proof_idx: usize) -> Self {
        Self { inner: self.inner.fork_for_proof(proof_idx) }
    }

    pub fn into_for_proof(self, proof_idx: usize) -> Self {
        Self { inner: self.inner.into_for_proof(proof_idx) }
    }

    pub fn proof_idx(&self) -> usize {
        self.inner.proof_idx()
    }

    pub fn next_tidx(&self) -> usize {
        self.inner.next_tidx()
    }

    pub fn record(&self) -> &RecursionRecord {
        self.inner.record()
    }

    pub(crate) fn record_mut(&mut self) -> &mut RecursionRecord {
        self.inner.record_mut()
    }

    pub fn take_record(self) -> RecursionRecord {
        self.inner.take_record()
    }

    fn pending_observe_duplex(&self, value: Val) -> Option<PendingDuplexCapture> {
        let challenger = self.inner.inner();
        if challenger.input_buffer.len() + 1 != 8 {
            return None;
        }
        let tidx = self.next_tidx().checked_sub(challenger.input_buffer.len())?;
        let mut input16 = challenger.sponge_state;
        for (idx, buffered) in challenger.input_buffer.iter().copied().enumerate() {
            input16[idx] = buffered;
        }
        input16[challenger.input_buffer.len()] = value;
        Some(PendingDuplexCapture {
            tidx,
            prev_rate: core::array::from_fn(|idx| challenger.sponge_state[idx]),
            input16,
            absorb_count: 8,
        })
    }

    fn pending_sample_duplex(&self) -> Option<PendingDuplexCapture> {
        let challenger = self.inner.inner();
        if challenger.input_buffer.is_empty() && !challenger.output_buffer.is_empty() {
            return None;
        }
        let tidx = self.next_tidx().checked_sub(challenger.input_buffer.len())?;
        let mut input16 = challenger.sponge_state;
        for (idx, buffered) in challenger.input_buffer.iter().copied().enumerate() {
            input16[idx] = buffered;
        }
        Some(PendingDuplexCapture {
            tidx,
            prev_rate: core::array::from_fn(|idx| challenger.sponge_state[idx]),
            input16,
            absorb_count: challenger.input_buffer.len(),
        })
    }

    fn publish_duplex(&mut self, pending: PendingDuplexCapture) {
        let proof_idx = self.proof_idx();
        let output16 = self.inner.inner().sponge_state;
        let proof = self.inner.active_proof_record_mut();
        let prev_s_count =
            proof.transcript.sponge_blocks.last().map_or(0, |block| block.squeeze_count);
        let is_proof_start = proof.transcript.sponge_blocks.is_empty();
        let mut absorb_mask = [false; 8];
        absorb_mask[..pending.absorb_count].fill(true);
        proof.transcript.sponge_blocks.push(SpecSpongeBlock {
            proof_idx,
            is_proof_start,
            is_proof_last: false,
            tidx: pending.tidx,
            prev_rate: pending.prev_rate,
            input16: pending.input16,
            output16,
            absorb_mask,
            squeeze_mask: [false; POSEIDON2_WIDTH],
            prev_s_count,
            absorb_count: pending.absorb_count,
            squeeze_count: 0,
        });
        self.inner.record_mut().poseidon2.record_poseidon2(pending.input16);
    }

    fn mark_sampled_lane(&mut self) {
        let lane = self.inner.inner().output_buffer.len();
        let block = self
            .inner
            .active_proof_record_mut()
            .transcript
            .sponge_blocks
            .last_mut()
            .expect("a real challenger sample must follow a captured duplex");
        assert!(lane < POSEIDON2_WIDTH, "captured challenger output lane out of range");
        assert!(!block.squeeze_mask[lane], "captured challenger output lane sampled twice");
        block.squeeze_mask[lane] = true;
        block.squeeze_count += 1;
    }

    pub fn finish_transcript_capture(&mut self) -> Result<(), String> {
        if !self.inner.inner().input_buffer.is_empty() {
            return Err(format!(
                "proof {} transcript ended with {} unflushed absorbs",
                self.proof_idx(),
                self.inner.inner().input_buffer.len()
            ));
        }
        let proof_idx = self.proof_idx();
        let blocks = &mut self.inner.active_proof_record_mut().transcript.sponge_blocks;
        let last = blocks
            .last_mut()
            .ok_or_else(|| format!("proof {proof_idx} transcript captured no duplex"))?;
        last.is_proof_last = true;
        Ok(())
    }
}

impl CanObserve<Val> for CoreRecordingChallenger {
    fn observe(&mut self, value: Val) {
        let pending = self.pending_observe_duplex(value);
        self.inner.observe(value);
        if let Some(pending) = pending {
            self.publish_duplex(pending);
        }
    }

    fn observe_slice(&mut self, values: &[Val]) {
        for value in values {
            self.observe(*value);
        }
    }
}

impl<const N: usize> CanObserve<[Val; N]> for CoreRecordingChallenger {
    fn observe(&mut self, values: [Val; N]) {
        for value in values {
            self.observe(value);
        }
    }
}

impl CanObserve<DigestHash> for CoreRecordingChallenger {
    fn observe(&mut self, values: DigestHash) {
        for value in values {
            self.observe(value);
        }
    }
}

impl CanSample<Val> for CoreRecordingChallenger {
    fn sample(&mut self) -> Val {
        let pending = self.pending_sample_duplex();
        let value = self.inner.sample();
        if let Some(pending) = pending {
            self.publish_duplex(pending);
        }
        self.mark_sampled_lane();
        value
    }

    fn sample_array<const N: usize>(&mut self) -> [Val; N] {
        core::array::from_fn(|_| <Self as CanSample<Val>>::sample(self))
    }

    fn sample_vec(&mut self, n: usize) -> Vec<Val> {
        (0..n).map(|_| <Self as CanSample<Val>>::sample(self)).collect()
    }
}

impl CanSample<Challenge> for CoreRecordingChallenger {
    fn sample(&mut self) -> Challenge {
        let values = (0..<Challenge as AbstractExtensionField<Val>>::D)
            .map(|_| <Self as CanSample<Val>>::sample(self))
            .collect::<Vec<_>>();
        Challenge::from_base_slice(&values)
    }

    fn sample_array<const N: usize>(&mut self) -> [Challenge; N] {
        core::array::from_fn(|_| <Self as CanSample<Challenge>>::sample(self))
    }

    fn sample_vec(&mut self, n: usize) -> Vec<Challenge> {
        (0..n).map(|_| <Self as CanSample<Challenge>>::sample(self)).collect()
    }
}

impl CanSampleBits<usize> for CoreRecordingChallenger {
    fn sample_bits(&mut self, bits: usize) -> usize {
        debug_assert!(bits < (usize::BITS as usize));
        debug_assert!((1usize << bits) < Val::ORDER_U64 as usize);
        let sample_tidx = self.next_tidx();
        let sample = <Self as CanSample<Val>>::sample(self);
        let value = sample.as_canonical_u64() as usize & ((1usize << bits) - 1);
        self.inner.record_sample_bits_value(sample_tidx, bits, value);
        value
    }
}

impl FieldChallenger<Val> for CoreRecordingChallenger {
    fn observe_ext_element<Ext: AbstractExtensionField<Val>>(&mut self, ext: Ext) {
        self.observe_slice(ext.as_base_slice());
    }

    fn sample_ext_element<Ext: AbstractExtensionField<Val>>(&mut self) -> Ext {
        let values = <Self as CanSample<Val>>::sample_vec(self, Ext::D);
        Ext::from_base_slice(&values)
    }
}

impl GrindingChallenger for CoreRecordingChallenger {
    type Witness = Val;

    fn grind(&mut self, bits: usize) -> Self::Witness {
        let witness = (0..Val::ORDER_U64)
            .map(Val::from_canonical_u64)
            .find(|witness| self.clone().check_witness(bits, *witness))
            .expect("failed to find witness");
        assert!(self.check_witness(bits, witness));
        witness
    }
}

pub struct RecordingSC {
    stage: RecordingStage,
    perm: koalabear_poseidon2::koala_bear_poseidon2::Perm,
    pcs: RecordingPcs,
    mlpcs: RecordingMlpcs,
}

impl replay_compatible_sealed::Sealed for RecordingSC {}
impl ReplayCompatibleProofConfig for RecordingSC {}

impl replay_compatible_sealed::Sealed for SCKoalaBearPoseidon2 {}
impl ReplayCompatibleProofConfig for SCKoalaBearPoseidon2 {}

impl RecordingSC {
    #[must_use]
    pub fn new() -> Self {
        Self::for_stage(RecordingStage::Core)
    }

    #[must_use]
    pub fn for_stage(stage: RecordingStage) -> Self {
        let perm = my_perm();
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());
        let val_mmcs = ValMmcs::new(hash, compress);
        let dft = Dft {};
        let fri_config = stage.fri_config();
        let pcs = RecordingPcs::new(27, dft, val_mmcs.clone(), fri_config);
        let mlpcs_config = match stage {
            RecordingStage::Core => core_mlpcs_config(),
            RecordingStage::Compress => compress_mlpcs_config(),
            RecordingStage::Shrink => shrink_mlpcs_config(),
        };
        let mlpcs = RecordingMlpcs::from_config(val_mmcs, mlpcs_config);
        Self { stage, perm, pcs, mlpcs }
    }

    /// Query grinding strength for the active child-proof transcript. This is
    /// transcript metadata, independent of the tracegen execution backend.
    pub fn whir_grinding_bits_query(&self) -> usize {
        self.stage.fri_config().grinding_bits_query
    }

    pub const fn stage(&self) -> RecordingStage {
        self.stage
    }
}

impl Clone for RecordingSC {
    fn clone(&self) -> Self {
        Self::for_stage(self.stage)
    }
}

impl Default for RecordingSC {
    fn default() -> Self {
        Self::new()
    }
}

impl Serialize for RecordingSC {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.stage.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for RecordingSC {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(Self::for_stage(RecordingStage::deserialize(deserializer)?))
    }
}

impl StarkGenericConfig for RecordingSC {
    type Val = Val;
    type Domain = <RecordingPcs as p3_commit::Pcs<Challenge, Challenger>>::Domain;
    type Pcs = RecordingPcs;
    type Challenge = Challenge;
    type Challenger = Challenger;

    fn pcs(&self) -> &Self::Pcs {
        &self.pcs
    }

    fn challenger(&self) -> Self::Challenger {
        Challenger::new(self.perm.clone())
    }
}

impl ZeroCommitment<RecordingSC> for RecordingPcs {
    fn zero_commitment(&self) -> dt_stark::Com<RecordingSC> {
        DigestHash::from([Val::zero(); DIGEST_SIZE])
    }
}

impl SCStarkGenericConfig for RecordingSC {
    type Mlpcs = RecordingMlpcs;
    type MlChallenge = Challenge;
    type MlPcsProverData = <RecordingMlpcs as MlPCS>::ProverData;
    type MlChallenger = CoreRecordingChallenger;

    fn mlpcs(&self) -> &Self::Mlpcs {
        &self.mlpcs
    }

    fn mlpcs_commit_options(&self) -> MlCommitOptions {
        if koalabear_poseidon2::whir_config().stacking_enabled(self.stage.whir_stage()) {
            MlCommitOptions::auto_stacking()
        } else {
            MlCommitOptions::no_stacking()
        }
    }

    fn mlchallenger(&self) -> Self::MlChallenger {
        CoreRecordingChallenger::from_inner(0, Challenger::new(self.perm.clone()))
    }
}

impl ZeroCommitment<RecordingSC> for RecordingMlpcs {
    fn zero_commitment(&self) -> dt_stark::sumcheck::config::MlCom<RecordingSC> {
        DigestHash::from([Val::zero(); DIGEST_SIZE])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system_dt::SpecSponge;

    #[test]
    fn recording_config_serde_preserves_stage() {
        for stage in [RecordingStage::Core, RecordingStage::Compress, RecordingStage::Shrink] {
            let encoded = bincode::serialize(&RecordingSC::for_stage(stage)).expect("serialize");
            let decoded: RecordingSC = bincode::deserialize(&encoded).expect("deserialize");
            assert_eq!(decoded.stage(), stage);
        }
    }

    #[test]
    fn first_pass_duplex_capture_matches_spec_replay_oracle() {
        let mut challenger = CoreRecordingChallenger::from_inner(0, Challenger::new(my_perm()));
        challenger.observe_slice(&[
            Val::from_canonical_u32(1),
            Val::from_canonical_u32(2),
            Val::from_canonical_u32(3),
        ]);
        let _: Val = challenger.sample();
        let _: Val = challenger.sample();
        challenger.observe_slice(&(10..18).map(Val::from_canonical_u32).collect::<Vec<_>>());
        let _ = challenger.sample_bits(5);
        challenger.finish_transcript_capture().expect("complete transcript");

        let record = challenger.take_record();
        let proof = record
            .proof_records
            .iter()
            .find(|proof| proof.proof_idx == 0)
            .expect("active proof record");
        let replayed = SpecSponge::replay(0, &proof.transcript.events, &record.poseidon2_memo)
            .expect("oracle replay");
        assert_eq!(proof.transcript.sponge_blocks, replayed);
        assert_eq!(record.poseidon2.total_count_usize(), replayed.len());
    }

    #[test]
    fn first_pass_duplex_capture_rejects_tail_absorbs() {
        let mut challenger = CoreRecordingChallenger::from_inner(4, Challenger::new(my_perm()));
        challenger.observe(Val::from_canonical_u32(9));
        let err =
            challenger.finish_transcript_capture().expect_err("unflushed absorb must fail closed");
        assert!(err.contains("unflushed absorbs"));
    }
}
