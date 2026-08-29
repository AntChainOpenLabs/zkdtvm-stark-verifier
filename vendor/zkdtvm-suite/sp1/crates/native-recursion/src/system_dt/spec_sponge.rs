use core::{array, fmt};

use p3_field::AbstractField;
use serde::{Deserialize, Serialize};

use crate::{
    config::{F, POSEIDON2_WIDTH},
    system_dt::{RecursionTranscriptEvent, RecursionTranscriptEventKind},
    transcript_dt::poseidon2::RecursionPoseidon2Memo,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpecSpongeBlock {
    pub proof_idx: usize,
    pub is_proof_start: bool,
    pub is_proof_last: bool,
    pub tidx: usize,
    pub prev_rate: [F; 8],
    pub input16: [F; POSEIDON2_WIDTH],
    pub output16: [F; POSEIDON2_WIDTH],
    pub absorb_mask: [bool; 8],
    pub squeeze_mask: [bool; POSEIDON2_WIDTH],
    pub prev_s_count: usize,
    pub absorb_count: usize,
    pub squeeze_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpecSpongeError {
    EmptyTranscript { proof_idx: usize },
    NonContiguousTidx { expected: usize, actual: usize },
    NonContiguousAbsorbTidx { expected: usize, actual: usize },
    EmptyOutputAfterDuplex { tidx: usize },
    SampleMismatch { tidx: usize, expected: F, actual: F },
    SqueezeOverflow { block_tidx: usize },
    TailAbsorb { proof_idx: usize, first_tidx: usize, count: usize },
    TidxOverflow,
}

impl fmt::Display for SpecSpongeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyTranscript { proof_idx } => {
                write!(f, "proof {proof_idx} transcript has no events")
            }
            Self::NonContiguousTidx { expected, actual } => {
                write!(f, "non-contiguous transcript tidx: expected {expected}, got {actual}")
            }
            Self::NonContiguousAbsorbTidx { expected, actual } => {
                write!(f, "non-contiguous absorb tidx: expected {expected}, got {actual}")
            }
            Self::EmptyOutputAfterDuplex { tidx } => {
                write!(f, "duplex at tidx {tidx} did not produce sample output")
            }
            Self::SampleMismatch { tidx, expected, actual } => {
                write!(f, "sample mismatch at tidx {tidx}: expected {expected:?}, got {actual:?}")
            }
            Self::SqueezeOverflow { block_tidx } => {
                write!(f, "more than 16 squeeze events in block starting at tidx {block_tidx}")
            }
            Self::TailAbsorb { proof_idx, first_tidx, count } => write!(
                f,
                "proof {proof_idx} transcript ended with {count} unflushed absorb events starting at tidx {first_tidx}"
            ),
            Self::TidxOverflow => write!(f, "transcript tidx overflow"),
        }
    }
}

impl std::error::Error for SpecSpongeError {}

#[derive(Debug, Clone)]
pub struct SpecSponge<'a> {
    memo: &'a RecursionPoseidon2Memo,
    proof_idx: usize,
    state: [F; POSEIDON2_WIDTH],
    pending_absorb: Vec<(usize, F)>,
    output_buffer: Vec<F>,
    blocks: Vec<SpecSpongeBlock>,
}

impl<'a> SpecSponge<'a> {
    pub fn replay(
        proof_idx: usize,
        events: &[RecursionTranscriptEvent],
        memo: &'a RecursionPoseidon2Memo,
    ) -> Result<Vec<SpecSpongeBlock>, SpecSpongeError> {
        let mut sponge = Self::new(proof_idx, memo);
        sponge.absorb_events(events)?;
        sponge.finish()
    }

    pub fn new(proof_idx: usize, memo: &'a RecursionPoseidon2Memo) -> Self {
        Self {
            memo,
            proof_idx,
            state: [F::zero(); POSEIDON2_WIDTH],
            pending_absorb: Vec::new(),
            output_buffer: Vec::new(),
            blocks: Vec::new(),
        }
    }

    fn absorb_events(
        &mut self,
        events: &[RecursionTranscriptEvent],
    ) -> Result<(), SpecSpongeError> {
        for (expected_tidx, event) in events.iter().enumerate() {
            if event.tidx != expected_tidx {
                return Err(SpecSpongeError::NonContiguousTidx {
                    expected: expected_tidx,
                    actual: event.tidx,
                });
            }

            match event.kind {
                RecursionTranscriptEventKind::Observe => self.observe(event.tidx, event.value)?,
                RecursionTranscriptEventKind::Sample => self.sample(event.tidx, event.value)?,
            }
        }
        Ok(())
    }

    fn finish(mut self) -> Result<Vec<SpecSpongeBlock>, SpecSpongeError> {
        if let Some(&(first_tidx, _)) = self.pending_absorb.first() {
            return Err(SpecSpongeError::TailAbsorb {
                proof_idx: self.proof_idx,
                first_tidx,
                count: self.pending_absorb.len(),
            });
        }
        if self.blocks.is_empty() {
            return Err(SpecSpongeError::EmptyTranscript { proof_idx: self.proof_idx });
        }

        if let Some(last) = self.blocks.last_mut() {
            last.is_proof_last = true;
        }

        Ok(self.blocks)
    }

    fn observe(&mut self, tidx: usize, value: F) -> Result<(), SpecSpongeError> {
        self.output_buffer.clear();
        self.pending_absorb.push((tidx, value));
        if self.pending_absorb.len() == 8 {
            let start_tidx = self.pending_absorb[0].0;
            self.duplex(start_tidx)?;
        }
        Ok(())
    }

    fn sample(&mut self, tidx: usize, value: F) -> Result<(), SpecSpongeError> {
        if !self.pending_absorb.is_empty() || self.output_buffer.is_empty() {
            let start_tidx = self.pending_absorb.first().map(|(tidx, _)| *tidx).unwrap_or(tidx);
            self.duplex(start_tidx)?;
        }

        let output_idx = self
            .output_buffer
            .len()
            .checked_sub(1)
            .ok_or(SpecSpongeError::EmptyOutputAfterDuplex { tidx })?;
        let expected = self.output_buffer.pop().expect("checked non-empty output buffer");
        if expected != value {
            return Err(SpecSpongeError::SampleMismatch { tidx, expected, actual: value });
        }

        let block = self.blocks.last_mut().expect("sample always creates or uses a block");
        let expected_tidx = block
            .tidx
            .checked_add(block.absorb_count)
            .and_then(|tidx| tidx.checked_add(block.squeeze_count))
            .ok_or(SpecSpongeError::TidxOverflow)?;
        if tidx != expected_tidx {
            return Err(SpecSpongeError::NonContiguousTidx {
                expected: expected_tidx,
                actual: tidx,
            });
        }
        if block.squeeze_count >= POSEIDON2_WIDTH {
            return Err(SpecSpongeError::SqueezeOverflow { block_tidx: block.tidx });
        }
        block.squeeze_mask[output_idx] = true;
        block.squeeze_count += 1;
        Ok(())
    }

    fn duplex(&mut self, tidx: usize) -> Result<(), SpecSpongeError> {
        let absorb_count = self.pending_absorb.len();
        for (offset, (actual_tidx, _)) in self.pending_absorb.iter().enumerate() {
            let expected = tidx.checked_add(offset).ok_or(SpecSpongeError::TidxOverflow)?;
            if *actual_tidx != expected {
                return Err(SpecSpongeError::NonContiguousAbsorbTidx {
                    expected,
                    actual: *actual_tidx,
                });
            }
        }

        let prev_rate = array::from_fn(|i| self.state[i]);
        let mut input16 = self.state;
        let mut absorb_mask = [false; 8];
        for (i, (_, value)) in self.pending_absorb.iter().enumerate() {
            input16[i] = *value;
            absorb_mask[i] = true;
        }
        let output16 = self.memo.permute(input16);
        let prev_s_count = self.blocks.last().map(|block| block.squeeze_count).unwrap_or(0);
        let is_proof_start = self.blocks.is_empty();

        self.blocks.push(SpecSpongeBlock {
            proof_idx: self.proof_idx,
            is_proof_start,
            is_proof_last: false,
            tidx,
            prev_rate,
            input16,
            output16,
            absorb_mask,
            squeeze_mask: [false; POSEIDON2_WIDTH],
            prev_s_count,
            absorb_count,
            squeeze_count: 0,
        });

        self.pending_absorb.clear();
        self.state = output16;
        self.output_buffer = output16.to_vec();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observe(tidx: usize, value: usize) -> RecursionTranscriptEvent {
        RecursionTranscriptEvent {
            tidx,
            kind: RecursionTranscriptEventKind::Observe,
            value: F::from_canonical_usize(value),
        }
    }

    fn sample_event(tidx: usize, value: F) -> RecursionTranscriptEvent {
        RecursionTranscriptEvent { tidx, kind: RecursionTranscriptEventKind::Sample, value }
    }

    #[test]
    fn full_absorb_block_can_have_no_squeezes() {
        let memo = RecursionPoseidon2Memo::default();
        let events = (0..16).map(|i| observe(i, i + 1)).collect::<Vec<_>>();
        let blocks = SpecSponge::replay(0, &events, &memo).expect("valid replay");
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].absorb_count, 8);
        assert_eq!(blocks[0].squeeze_count, 0);
        assert!(blocks[0].absorb_mask.iter().all(|mask| *mask));
        assert_eq!(blocks[1].tidx, 8);
    }

    #[test]
    fn partial_absorb_sample_flushes_t2_block() {
        let memo = RecursionPoseidon2Memo::default();
        let mut seed = SpecSponge::new(0, &memo);
        seed.observe(0, F::from_canonical_usize(11)).unwrap();
        let start = seed.pending_absorb[0].0;
        seed.duplex(start).unwrap();
        let expected = seed.output_buffer.last().copied().unwrap();

        let events = vec![observe(0, 11), sample_event(1, expected)];
        let blocks = SpecSponge::replay(0, &events, &memo).expect("valid replay");
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].absorb_count, 1);
        assert_eq!(blocks[0].squeeze_count, 1);
        assert!(blocks[0].squeeze_mask[15]);
    }

    #[test]
    fn pure_squeeze_t3_requires_exhausted_previous_output() {
        let memo = RecursionPoseidon2Memo::default();
        let mut events = (0..8).map(|i| observe(i, i + 1)).collect::<Vec<_>>();
        let mut seed = SpecSponge::new(0, &memo);
        for event in &events {
            seed.observe(event.tidx, event.value).unwrap();
        }
        let mut tidx = 8;
        while let Some(value) = seed.output_buffer.pop() {
            events.push(sample_event(tidx, value));
            tidx += 1;
        }
        let start = tidx;
        seed.duplex(start).unwrap();
        events.push(sample_event(tidx, seed.output_buffer.last().copied().unwrap()));

        let blocks = SpecSponge::replay(0, &events, &memo).expect("valid replay");
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].squeeze_count, 16);
        assert_eq!(blocks[1].absorb_count, 0);
        assert_eq!(blocks[1].prev_s_count, 16);
        assert!(blocks[1].squeeze_mask[15]);
    }

    #[test]
    fn observe_discards_unused_output_before_next_block() {
        let memo = RecursionPoseidon2Memo::default();
        let first = (0..8).map(|i| observe(i, i + 1)).collect::<Vec<_>>();
        let mut seed = SpecSponge::new(0, &memo);
        for event in &first {
            seed.observe(event.tidx, event.value).unwrap();
        }
        let sample = seed.output_buffer.pop().unwrap();

        let mut events = first;
        events.push(sample_event(8, sample));
        for i in 0..8 {
            events.push(observe(9 + i, 100 + i));
        }

        let blocks = SpecSponge::replay(0, &events, &memo).expect("valid replay");
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].squeeze_count, 1);
        assert_eq!(blocks[1].tidx, 9);
        assert_eq!(blocks[1].prev_s_count, 1);
    }

    #[test]
    fn tail_absorb_is_rejected() {
        let memo = RecursionPoseidon2Memo::default();
        let err = SpecSponge::replay(0, &[observe(0, 1)], &memo).expect_err("tail absorb rejected");
        assert!(matches!(err, SpecSpongeError::TailAbsorb { .. }));
    }

    #[test]
    fn empty_per_proof_transcript_is_rejected() {
        let memo = RecursionPoseidon2Memo::default();
        let err = SpecSponge::replay(9, &[], &memo).expect_err("empty proof transcript rejected");
        assert!(matches!(err, SpecSpongeError::EmptyTranscript { proof_idx: 9 }));
    }
}

/// Locates transcript limbs inside sponge rows for the window
/// buses. Kind: 0 = absorb (1045, lanes = input16[0..8]); 1 = squeeze-lo
/// (1046, lanes = output16[0..8]); 2 = squeeze-hi (1047, output16[8..16]).
/// Squeeze offset o (transcript order) maps to output16[15 - o].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpongeWindowSlot {
    pub row_tidx: usize,
    pub kind: u8,
    pub lane: usize,
}

#[derive(Debug, Default, Clone)]
pub struct SpongeWindowLocator {
    /// (tidx, absorb_count, squeeze_count) per block, ascending tidx.
    blocks: Vec<(usize, usize, usize)>,
}

impl SpongeWindowLocator {
    pub fn from_blocks(blocks: &[SpecSpongeBlock]) -> Self {
        let mut spans =
            blocks.iter().map(|b| (b.tidx, b.absorb_count, b.squeeze_count)).collect::<Vec<_>>();
        spans.sort_unstable();
        Self { blocks: spans }
    }

    pub fn locate(&self, t: usize) -> Option<SpongeWindowSlot> {
        let idx = self.blocks.partition_point(|&(tidx, _, _)| tidx <= t);
        if idx == 0 {
            return None;
        }
        let (tidx, absorb, squeeze) = self.blocks[idx - 1];
        if t < tidx + absorb {
            return Some(SpongeWindowSlot { row_tidx: tidx, kind: 0, lane: t - tidx });
        }
        if t < tidx + absorb + squeeze {
            let lane16 = 15 - (t - tidx - absorb);
            return Some(if lane16 >= 8 {
                SpongeWindowSlot { row_tidx: tidx, kind: 2, lane: lane16 - 8 }
            } else {
                SpongeWindowSlot { row_tidx: tidx, kind: 1, lane: lane16 }
            });
        }
        None
    }
}

/// Balance-forced demand counts per (proof, sponge-row tidx, window kind).
#[derive(Debug, Default, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SpongeWindowDemands {
    counts: std::collections::BTreeMap<(usize, usize, u8), u32>,
}

impl SpongeWindowDemands {
    pub fn demand(&mut self, proof_idx: usize, slot: SpongeWindowSlot) {
        *self.counts.entry((proof_idx, slot.row_tidx, slot.kind)).or_default() += 1;
    }

    pub fn count(&self, proof_idx: usize, row_tidx: usize, kind: u8) -> u32 {
        self.counts.get(&(proof_idx, row_tidx, kind)).copied().unwrap_or(0)
    }
}

/// Alignment diagnostic (env `DT_NATIVE_D10_ALIGN=1`): for each consumer
/// item class, print the distinct (window kind, lane) slots its limbs occupy
/// across rows/proofs. Uniform sets => baked-lane window recvs are legal for
/// that class; otherwise the consumer keeps per-limb recvs via ScalarTap.
pub fn d10_alignment_census(
    label: &str,
    locator: &SpongeWindowLocator,
    items: impl Iterator<Item = (usize, usize)>, // (class, tidx)
) {
    use std::collections::BTreeMap;
    if !crate::debug_prints_enabled() {
        return;
    }
    let mut slots: BTreeMap<usize, std::collections::BTreeSet<(u8, usize)>> = BTreeMap::new();
    let mut missed = 0usize;
    for (class, t) in items {
        match locator.locate(t) {
            Some(slot) => {
                slots.entry(class).or_default().insert((slot.kind, slot.lane));
            }
            None => missed += 1,
        }
    }
    for (class, set) in &slots {
        let uniform = set.len() == 1;
        println!(
            "native_d10_align consumer={} class={} distinct_slots={} uniform={} slots={:?}",
            label,
            class,
            set.len(),
            uniform,
            set.iter().take(8).collect::<Vec<_>>(),
        );
    }
    if missed > 0 {
        println!("native_d10_align consumer={} missed_tidx={}", label, missed);
    }
}
