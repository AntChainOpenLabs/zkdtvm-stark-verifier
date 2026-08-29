use p3_challenger::{CanObserve, CanSample, CanSampleBits, FieldChallenger, GrindingChallenger};
use p3_field::{AbstractExtensionField, AbstractField, PrimeField64};
use serde::{Deserialize, Serialize};

use crate::{
    config::{Digest, F},
    system_dt::{
        RecursionRecord, RecursionTranscriptBitsEvent, RecursionTranscriptEvent,
        RecursionTranscriptEventKind,
    },
};
use dt_stark::koalabear_poseidon2::koala_bear_poseidon2::Val as KoalaBearVal;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecursionRecordingChallenger<C = ()> {
    proof_idx: usize,
    next_tidx: usize,
    record: RecursionRecord,
    inner: C,
}

impl RecursionRecordingChallenger<()> {
    pub fn new(proof_idx: usize) -> Self {
        Self::from_inner(proof_idx, ())
    }
}

impl<C> RecursionRecordingChallenger<C> {
    pub fn from_inner(proof_idx: usize, inner: C) -> Self {
        Self { proof_idx, next_tidx: 0, record: RecursionRecord::default(), inner }
    }

    pub fn proof_idx(&self) -> usize {
        self.proof_idx
    }

    pub fn next_tidx(&self) -> usize {
        self.next_tidx
    }

    pub fn record(&self) -> &RecursionRecord {
        &self.record
    }

    pub fn record_mut(&mut self) -> &mut RecursionRecord {
        &mut self.record
    }

    pub fn inner(&self) -> &C {
        &self.inner
    }

    pub fn inner_mut(&mut self) -> &mut C {
        &mut self.inner
    }

    pub fn into_inner(self) -> C {
        self.inner
    }

    pub fn into_record(self) -> RecursionRecord {
        self.record
    }

    pub fn take_record(self) -> RecursionRecord {
        self.into_record()
    }

    pub fn into_parts(self) -> (C, RecursionRecord) {
        (self.inner, self.record)
    }

    pub fn record_observe(&mut self, value: F) -> usize {
        self.push_transcript_event(RecursionTranscriptEventKind::Observe, value)
    }

    pub fn record_sample(&mut self, value: F) -> usize {
        self.push_transcript_event(RecursionTranscriptEventKind::Sample, value)
    }

    fn sample_base_with_tidx(&mut self) -> (usize, F)
    where
        C: CanSample<F>,
    {
        let value = self.inner.sample();
        let tidx = self.record_sample(value);
        (tidx, value)
    }

    fn push_transcript_event(&mut self, kind: RecursionTranscriptEventKind, value: F) -> usize {
        let tidx = self.next_tidx;
        self.next_tidx = self.next_tidx.checked_add(1).expect("transcript tidx overflow");
        self.active_proof_record_mut().transcript.events.push(RecursionTranscriptEvent {
            tidx,
            kind,
            value,
        });
        tidx
    }

    fn record_sample_bits(&mut self, sample_tidx: usize, bits: usize, value: usize) {
        self.active_proof_record_mut().transcript.bits_events.push(RecursionTranscriptBitsEvent {
            sample_tidx,
            bits,
            value,
        });
    }

    /// Slot-local recorder lease. A challenger owns exactly one proof segment,
    /// so scalar transcript events never search the record's proof vector.
    pub(crate) fn active_proof_record_mut(
        &mut self,
    ) -> &mut crate::system_dt::RecursionProofRecord {
        if self.record.proof_records.is_empty() {
            self.record.proof_records.push(crate::system_dt::RecursionProofRecord {
                proof_idx: self.proof_idx,
                ..Default::default()
            });
        }
        assert_eq!(
            self.record.proof_records.len(),
            1,
            "one recording challenger must own exactly one proof slot"
        );
        assert_eq!(
            self.record.proof_records[0].proof_idx, self.proof_idx,
            "recording challenger proof-slot authority mismatch"
        );
        &mut self.record.proof_records[0]
    }

    pub(crate) fn record_sample_bits_value(
        &mut self,
        sample_tidx: usize,
        bits: usize,
        value: usize,
    ) {
        self.record_sample_bits(sample_tidx, bits, value);
    }
}

impl<C> RecursionRecordingChallenger<C>
where
    C: Clone,
{
    pub fn fork_for_proof(&self, proof_idx: usize) -> Self {
        let mut fork = self.clone();
        // `RecursionRecord::clone` deliberately starts with an independent,
        // empty memo. Preserve the legacy fork helper's shared request table
        // for tests and diagnostic callers; production uses `into_for_proof`
        // and moves the seed without cloning it.
        fork.record.poseidon2_memo = self.record.poseidon2_memo.fork();
        fork.into_for_proof(proof_idx)
    }
}

impl<C> RecursionRecordingChallenger<C> {
    /// Consume a prepared VK/transcript prefix and retag it for its one proof
    /// slot. Production recording uses this move-only path, so no proof clones
    /// the same seed record, prefix vectors, provider segment, or challenger
    /// state before verification begins.
    pub fn into_for_proof(mut self, proof_idx: usize) -> Self {
        assert!(
            self.record.proof_records.len() <= 1,
            "one recording seed may contain at most one proof prefix"
        );
        if let Some(seed_record) = self.record.proof_records.first_mut() {
            assert_eq!(
                seed_record.proof_idx, self.proof_idx,
                "recording seed prefix authority mismatch"
            );
            seed_record.proof_idx = proof_idx;
            for block in &mut seed_record.transcript.sponge_blocks {
                block.proof_idx = proof_idx;
            }
        }
        self.proof_idx = proof_idx;
        self
    }
}

impl<C> CanObserve<KoalaBearVal> for RecursionRecordingChallenger<C>
where
    C: CanObserve<KoalaBearVal>,
{
    fn observe(&mut self, value: KoalaBearVal) {
        self.inner.observe(value);
        self.record_observe(value);
    }

    fn observe_slice(&mut self, values: &[KoalaBearVal]) {
        self.inner.observe_slice(values);
        for value in values {
            self.record_observe(*value);
        }
    }
}

impl<C, const N: usize> CanObserve<[KoalaBearVal; N]> for RecursionRecordingChallenger<C>
where
    C: CanObserve<[KoalaBearVal; N]>,
{
    fn observe(&mut self, values: [KoalaBearVal; N]) {
        self.inner.observe(values);
        for value in values {
            self.record_observe(value);
        }
    }
}

impl<C> CanObserve<Digest> for RecursionRecordingChallenger<C>
where
    C: CanObserve<Digest>,
{
    fn observe(&mut self, values: Digest) {
        self.inner.observe(values);
        for value in values {
            self.record_observe(value);
        }
    }
}

impl<C> CanSample<F> for RecursionRecordingChallenger<C>
where
    C: CanSample<F>,
{
    fn sample(&mut self) -> F {
        self.sample_base_with_tidx().1
    }

    fn sample_array<const N: usize>(&mut self) -> [F; N] {
        core::array::from_fn(|_| self.sample())
    }

    fn sample_vec(&mut self, n: usize) -> Vec<F> {
        (0..n).map(|_| self.sample()).collect()
    }
}

impl<C> CanSampleBits<usize> for RecursionRecordingChallenger<C>
where
    C: CanSample<F>,
{
    fn sample_bits(&mut self, bits: usize) -> usize {
        debug_assert!(bits < (usize::BITS as usize));
        debug_assert!((1usize << bits) < F::ORDER_U64 as usize);
        let (sample_tidx, sample) = self.sample_base_with_tidx();
        let value = sample.as_canonical_u64() as usize & ((1usize << bits) - 1);
        self.record_sample_bits(sample_tidx, bits, value);
        value
    }
}

impl<C> FieldChallenger<F> for RecursionRecordingChallenger<C>
where
    C: CanObserve<F> + CanSample<F> + Sync,
{
    fn observe_ext_element<Ext: AbstractExtensionField<F>>(&mut self, ext: Ext) {
        self.observe_slice(ext.as_base_slice());
    }

    fn sample_ext_element<Ext: AbstractExtensionField<F>>(&mut self) -> Ext {
        let values = self.sample_vec(Ext::D);
        Ext::from_base_slice(&values)
    }
}

impl<C> GrindingChallenger for RecursionRecordingChallenger<C>
where
    C: CanObserve<F> + CanSample<F> + Clone + Sync,
{
    type Witness = F;

    fn grind(&mut self, bits: usize) -> Self::Witness {
        let witness = (0..F::ORDER_U64)
            .map(F::from_canonical_u64)
            .find(|witness| self.clone().check_witness(bits, *witness))
            .expect("failed to find witness");
        assert!(self.check_witness(bits, witness));
        witness
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{D_EF, EF, SC};
    use dt_stark::sumcheck::config::SCStarkGenericConfig;
    use p3_challenger::{
        CanObserve, CanSample, CanSampleBits, FieldChallenger, GrindingChallenger,
    };
    use p3_field::{AbstractExtensionField, AbstractField};

    #[test]
    fn recording_challenger_matches_inner_transcript() {
        let config = SC::default();
        let inner = config.mlchallenger();
        let mut plain = inner.clone();
        let mut recording = RecursionRecordingChallenger::from_inner(3, inner);

        let observed = F::from_canonical_u32(17);
        plain.observe(observed);
        recording.observe(observed);

        let plain_sample: F = plain.sample();
        let recorded_sample: F = recording.sample();
        assert_eq!(recorded_sample, plain_sample);

        let plain_bits = plain.sample_bits(5);
        let recorded_bits = recording.sample_bits(5);
        assert_eq!(recorded_bits, plain_bits);

        let plain_ext: EF = plain.sample_ext_element();
        let recorded_ext: EF = recording.sample_ext_element();
        assert_eq!(recorded_ext, plain_ext);

        let record = recording.into_record();
        let proof = &record.proof_records[0];
        assert_eq!(proof.proof_idx, 3);
        assert_eq!(proof.transcript.bits_events.len(), 1);
        assert_eq!(proof.transcript.bits_events[0].sample_tidx, 2);
        assert_eq!(proof.transcript.bits_events[0].bits, 5);
        assert_eq!(proof.transcript.bits_events[0].value, recorded_bits);

        let events = &proof.transcript.events;
        assert_eq!(events.len(), 3 + D_EF);
        for (expected_tidx, event) in events.iter().enumerate() {
            assert_eq!(event.tidx, expected_tidx);
        }
        assert_eq!(events[0].kind, RecursionTranscriptEventKind::Observe);
        assert_eq!(events[0].value, observed);
        assert_eq!(events[1].kind, RecursionTranscriptEventKind::Sample);
        assert_eq!(events[1].value, plain_sample);
        assert_eq!(events[2].kind, RecursionTranscriptEventKind::Sample);

        for (event, value) in events[3..].iter().zip(plain_ext.as_base_slice()) {
            assert_eq!(event.kind, RecursionTranscriptEventKind::Sample);
            assert_eq!(event.value, *value);
        }
    }

    #[test]
    fn recording_challenger_observes_arrays_hashes_and_pow() {
        let config = SC::default();
        let inner = config.mlchallenger();
        let mut plain = inner.clone();
        let mut recording = RecursionRecordingChallenger::from_inner(4, inner);

        let array: [F; 4] = core::array::from_fn(|i| F::from_canonical_u32(10 + i as u32));
        plain.observe(array);
        recording.observe(array);

        let hash_values = core::array::from_fn(|i| F::from_canonical_u32(30 + i as u32));
        let hash = Digest::from(hash_values);
        plain.observe(hash);
        recording.observe(hash);

        let witness = F::from_canonical_u32(0);
        let plain_ok = plain.check_witness(0, witness);
        let recording_ok = recording.check_witness(0, witness);
        assert_eq!(recording_ok, plain_ok);

        let record = recording.into_record();
        let proof = &record.proof_records[0];
        assert_eq!(proof.proof_idx, 4);
        assert_eq!(proof.transcript.bits_events.len(), 1);

        let events = &proof.transcript.events;
        assert_eq!(events.len(), 4 + 8 + 2);
        for (event, value) in events[..4].iter().zip(array) {
            assert_eq!(event.kind, RecursionTranscriptEventKind::Observe);
            assert_eq!(event.value, value);
        }
        for (event, value) in events[4..12].iter().zip(hash_values) {
            assert_eq!(event.kind, RecursionTranscriptEventKind::Observe);
            assert_eq!(event.value, value);
        }
        assert_eq!(events[12].kind, RecursionTranscriptEventKind::Observe);
        assert_eq!(events[12].value, witness);
        assert_eq!(events[13].kind, RecursionTranscriptEventKind::Sample);
    }

    #[test]
    fn fork_for_proof_carries_seed_record_with_new_proof_idx() {
        let config = SC::default();
        let inner = config.mlchallenger();
        let mut seed = RecursionRecordingChallenger::from_inner(0, inner);

        let seed_value = F::from_canonical_u32(11);
        seed.observe(seed_value);
        let memo_input = core::array::from_fn(|idx| F::from_canonical_usize(idx + 1));
        seed.record_mut().poseidon2_memo.permute(memo_input);

        let mut fork = seed.fork_for_proof(7);
        assert_eq!(fork.proof_idx(), 7);
        assert_eq!(fork.next_tidx(), 1);
        assert_eq!(fork.record().poseidon2_memo.snapshot().hits, 0);
        assert_eq!(fork.record().poseidon2_memo.snapshot().misses, 0);
        fork.record_mut().poseidon2_memo.permute(memo_input);
        assert_eq!(fork.record().poseidon2_memo.snapshot().hits, 1);
        assert_eq!(fork.record().poseidon2_memo.snapshot().misses, 0);

        let shard_value = F::from_canonical_u32(19);
        fork.observe(shard_value);

        let record = fork.into_record();
        assert_eq!(record.proof_records.len(), 1);
        let proof = &record.proof_records[0];
        assert_eq!(proof.proof_idx, 7);

        let events = &proof.transcript.events;
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].tidx, 0);
        assert_eq!(events[0].kind, RecursionTranscriptEventKind::Observe);
        assert_eq!(events[0].value, seed_value);
        assert_eq!(events[1].tidx, 1);
        assert_eq!(events[1].kind, RecursionTranscriptEventKind::Observe);
        assert_eq!(events[1].value, shard_value);
    }

    #[test]
    fn recursion_record_serde_roundtrip_rebuilds_pool_indexes() {
        let mut record = RecursionRecord::default();
        let proof = record.proof_record_mut(2);
        proof.transcript.events.push(RecursionTranscriptEvent {
            tidx: 0,
            kind: RecursionTranscriptEventKind::Observe,
            value: F::from_canonical_u32(3),
        });
        proof.transcript.bits_events.push(RecursionTranscriptBitsEvent {
            sample_tidx: 0,
            bits: 3,
            value: 5,
        });

        let poseidon_input = core::array::from_fn(|i| F::from_canonical_usize(i + 1));
        record.poseidon2.record_poseidon2_count(poseidon_input, 2);
        record.poseidon2_memo.permute(poseidon_input);
        record.range.record_range_count(3, 4, 2);
        record.pow.record_pow_count::<2, 4>(3, 2);

        let encoded = bincode::serialize(&record).expect("serialize record");
        let encoded_with_empty_memo =
            bincode::serialize(&record.clone()).expect("serialize cloned record");
        assert_eq!(encoded, encoded_with_empty_memo);
        let mut decoded: RecursionRecord =
            bincode::deserialize(&encoded).expect("deserialize record");
        assert_eq!(decoded, record);
        assert_eq!(decoded.poseidon2_memo.snapshot().hits, 0);
        assert_eq!(decoded.poseidon2_memo.snapshot().misses, 0);

        decoded.poseidon2.record_poseidon2(poseidon_input);
        assert_eq!(decoded.poseidon2.unique_count(), 2);
        assert_eq!(decoded.poseidon2.total_count_usize(), 3);

        decoded.range.record_range_count(3, 4, 1);
        assert_eq!(decoded.range.unique_count(), 2);
        assert_eq!(decoded.range.total_count_usize(), 3);

        decoded.pow.record_pow_count::<2, 4>(3, 1);
        decoded.pow.record_range_count::<2, 4>(3, 4);
        assert_eq!(decoded.pow.unique_count(), 3);
        assert_eq!(decoded.pow.total_count_usize(), 7);

        let stats = decoded.reduce_provider_inputs().expect("reduce deserialized pools once");
        assert_eq!(stats.raw_entries, 7);
        assert_eq!(stats.unique_entries, 3);
        assert_eq!(stats.duplicate_entries, 4);
        assert_eq!(decoded.poseidon2.unique_count(), 1);
        assert_eq!(decoded.poseidon2.total_count_usize(), 3);
        assert_eq!(decoded.range.unique_count(), 1);
        assert_eq!(decoded.range.total_count_usize(), 3);
        assert_eq!(decoded.pow.unique_count(), 1);
        assert_eq!(decoded.pow.total_count_usize(), 7);
    }
}
