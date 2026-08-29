use core::{array, borrow::Borrow, ops::Deref};

use dt_stark::{
    air::{FullAir, FullAirBuilder, MachineAir, PairCol},
    sumcheck::trace::CompressedMatrix,
};
use p3_air::BaseAir;
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use crate::{
    config::{F, POSEIDON2_WIDTH},
    system_dt::{RecursionNativeProgram, RecursionRecord},
    transcript_dt::{
        bus::Poseidon2PermuteBus,
        sponge::{
            bus::{TranscriptEventBus, TranscriptSpongeChainBus},
            columns::{TranscriptSpongeCols, NUM_TRANSCRIPT_SPONGE_COLS},
            trace::{transcript_sponge_rows_cached, TranscriptSpongeTraceGenerator},
        },
    },
};

#[derive(Debug, Clone, Copy)]
pub struct TranscriptSpongeAir {
    pub poseidon2_bus: Poseidon2PermuteBus,
    pub chain_bus: TranscriptSpongeChainBus,
    pub event_bus: TranscriptEventBus,
}

impl TranscriptSpongeAir {
    pub const fn new(
        poseidon2_bus: Poseidon2PermuteBus,
        chain_bus: TranscriptSpongeChainBus,
        event_bus: TranscriptEventBus,
    ) -> Self {
        Self { poseidon2_bus, chain_bus, event_bus }
    }
}

impl Default for TranscriptSpongeAir {
    fn default() -> Self {
        Self::new(
            Poseidon2PermuteBus::new(),
            TranscriptSpongeChainBus::new(),
            TranscriptEventBus::new(),
        )
    }
}

impl<Fld: Field> BaseAir<Fld> for TranscriptSpongeAir {
    fn width(&self) -> usize {
        NUM_TRANSCRIPT_SPONGE_COLS
    }
}

impl<AB: FullAirBuilder> FullAir<AB> for TranscriptSpongeAir {
    fn width(&self) -> usize {
        NUM_TRANSCRIPT_SPONGE_COLS
    }

    fn required_max_beta_power(&self) -> usize {
        [
            self.poseidon2_bus.required_max_beta_power_floor(),
            self.chain_bus.required_max_beta_power_floor(),
            self.event_bus.required_max_beta_power_floor(),
        ]
        .into_iter()
        .max()
        .expect("non-empty bus list")
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        (0..NUM_TRANSCRIPT_SPONGE_COLS).map(PairCol::Main).collect()
    }

    fn precompute_lc(&self, builder: &mut AB) {
        let denominators = {
            let main = builder.main();
            let local: &TranscriptSpongeCols<AB::VarMaybeExt> = main.borrow();
            let proof_idx = local.proof_idx.clone();
            let absorb_count = sum_masks::<AB, 8>(&local.absorb_mask);
            let squeeze_count = sum_masks::<AB, POSEIDON2_WIDTH>(&local.squeeze_mask);

            let mut denominators = Vec::with_capacity(27);
            denominators.push(self.poseidon2_bus.denominator(
                builder,
                local.input16.clone(),
                local.output16.clone(),
            ));
            for i in 0..8 {
                denominators.push(self.event_bus.denominator(
                    builder,
                    proof_idx.clone(),
                    local.tidx.clone() + const_maybe::<AB>(i),
                    AB::zero_maybe(),
                    local.input16[i].clone(),
                ));
            }
            for j in (0..POSEIDON2_WIDTH).rev() {
                let offset = POSEIDON2_WIDTH - 1 - j;
                denominators.push(self.event_bus.denominator(
                    builder,
                    proof_idx.clone(),
                    local.tidx.clone() + absorb_count.clone() + const_maybe::<AB>(offset),
                    AB::one_maybe(),
                    local.output16[j].clone(),
                ));
            }
            denominators.push(self.chain_bus.denominator(
                builder,
                proof_idx.clone(),
                local.tidx.clone(),
                chain_recv_state(local),
                local.prev_s_count.clone(),
            ));
            denominators.push(self.chain_bus.denominator(
                builder,
                proof_idx,
                local.tidx.clone() + absorb_count + squeeze_count.clone(),
                local.output16.clone(),
                squeeze_count,
            ));
            denominators
        };

        for denominator in denominators {
            builder.retain_precomputed(denominator);
        }
    }

    fn eval(&self, builder: &mut AB) {
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        let local: &TranscriptSpongeCols<AB::VarMaybeExt> = local_binding.deref().borrow();

        assert_bool(builder, local.is_valid.clone());
        assert_bool(builder, local.is_proof_start.clone());
        assert_bool(builder, local.is_proof_last.clone());
        assert_flag_implies(builder, local.is_proof_start.clone(), local.is_valid.clone());
        assert_flag_implies(builder, local.is_proof_last.clone(), local.is_valid.clone());

        for mask in local.absorb_mask.iter().chain(local.squeeze_mask.iter()) {
            assert_bool(builder, mask.clone());
        }
        // The 24 per-mask `mask => is_valid` edges are transitive
        // under the monotonicity gates below (absorb prefix-monotone anchors at
        // mask[0]; squeeze suffix-monotone anchors at mask[15]) - keep only the
        // two anchor edges.
        assert_flag_implies(builder, local.absorb_mask[0].clone(), local.is_valid.clone());
        assert_flag_implies(
            builder,
            local.squeeze_mask[POSEIDON2_WIDTH - 1].clone(),
            local.is_valid.clone(),
        );

        let one = AB::one_maybe();
        for i in 0..7 {
            builder.assert_zero(
                local.absorb_mask[i + 1].clone() * (one.clone() - local.absorb_mask[i].clone()),
            );
        }
        for j in 1..POSEIDON2_WIDTH {
            builder.assert_zero(
                local.squeeze_mask[j - 1].clone() * (one.clone() - local.squeeze_mask[j].clone()),
            );
        }

        for i in 0..8 {
            builder.assert_zero(
                (one.clone() - local.absorb_mask[i].clone()) *
                    (local.input16[i].clone() - local.prev_rate[i].clone()),
            );
        }

        builder.assert_zero(
            local.is_valid.clone() *
                (one.clone() - local.absorb_mask[7].clone()) *
                (one.clone() - local.squeeze_mask[POSEIDON2_WIDTH - 1].clone()),
        );
        builder.assert_zero(
            local.is_valid.clone() *
                (one.clone() - local.absorb_mask[0].clone()) *
                (const_maybe::<AB>(POSEIDON2_WIDTH) - local.prev_s_count.clone()),
        );

        builder.assert_zero(local.is_proof_start.clone() * local.tidx.clone());
        builder.assert_zero(local.is_proof_start.clone() * local.prev_s_count.clone());
        for mask in &local.absorb_mask[..2] {
            builder.assert_zero(local.is_proof_start.clone() * (one.clone() - mask.clone()));
        }
        for value in local.prev_rate.iter() {
            builder.assert_zero(local.is_proof_start.clone() * value.clone());
        }
        for value in &local.input16[8..] {
            builder.assert_zero(local.is_proof_start.clone() * value.clone());
        }
    }

    fn lookup(&self, builder: &mut AB) {
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        let local: &TranscriptSpongeCols<AB::VarMaybeExt> = local_binding.deref().borrow();

        // Order matches precompute_lc: Poseidon2 recv, absorb event sends,
        // squeeze event sends, chain recv, chain send.
        builder.recv(local.is_valid.clone());
        for i in 0..8 {
            // The GKV1 tag/version prefix is authenticated independently by
            // both the batch transcript and proof-shape consumers.  Fan out
            // those two observed events without changing the sponge transcript.
            let multiplicity = if i < 2 {
                local.absorb_mask[i].clone() + local.is_proof_start.clone()
            } else {
                local.absorb_mask[i].clone()
            };
            builder.send(multiplicity);
        }
        for j in (0..POSEIDON2_WIDTH).rev() {
            builder.send(local.squeeze_mask[j].clone());
        }
        builder.recv(local.is_valid.clone() - local.is_proof_start.clone());
        builder.send(local.is_valid.clone() - local.is_proof_last.clone());
    }
}

impl MachineAir<F> for TranscriptSpongeAir {
    type Record = RecursionRecord;
    type Program = RecursionNativeProgram<F>;

    fn name(&self) -> String {
        "NativeTranscriptSponge".to_string()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        Some(TranscriptSpongeTraceGenerator::trace_height(input))
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        _output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        TranscriptSpongeTraceGenerator::generate_trace_compressed(input)
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        for row in transcript_sponge_rows_cached(input).iter() {
            output.poseidon2.record_poseidon2(row.input16);
        }
    }

    fn included(&self, _record: &Self::Record) -> bool {
        true
    }

    fn local_only(&self) -> bool {
        true
    }
}

fn chain_recv_state<T: Clone>(local: &TranscriptSpongeCols<T>) -> [T; POSEIDON2_WIDTH] {
    array::from_fn(|i| if i < 8 { local.prev_rate[i].clone() } else { local.input16[i].clone() })
}

fn sum_masks<AB: FullAirBuilder, const N: usize>(masks: &[AB::VarMaybeExt; N]) -> AB::VarMaybeExt {
    let mut sum = AB::zero_maybe();
    for mask in masks {
        sum = sum + mask.clone();
    }
    sum
}

fn const_maybe<AB: FullAirBuilder>(value: usize) -> AB::VarMaybeExt {
    AB::VarMaybeExt::from(AB::F::from_canonical_usize(value))
}

fn assert_bool<AB: FullAirBuilder>(builder: &mut AB, value: AB::VarMaybeExt) {
    builder.assert_zero(value.clone() * (value - AB::one_maybe()));
}

fn assert_flag_implies<AB: FullAirBuilder>(
    builder: &mut AB,
    flag: AB::VarMaybeExt,
    condition: AB::VarMaybeExt,
) {
    builder.assert_zero(flag * (AB::one_maybe() - condition));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::{D_EF, F},
        system_dt::{
            RecursionProofRecord, RecursionTranscriptEvent, RecursionTranscriptEventKind,
            SpecSponge,
        },
        transcript_dt::sponge::trace::{
            trace_row, transcript_sponge_row_count, transcript_sponge_rows,
        },
    };
    use p3_matrix::Matrix;

    fn observe(tidx: usize, value: usize) -> RecursionTranscriptEvent {
        RecursionTranscriptEvent {
            tidx,
            kind: RecursionTranscriptEventKind::Observe,
            value: F::from_canonical_usize(value),
        }
    }

    fn sample(tidx: usize, value: F) -> RecursionTranscriptEvent {
        RecursionTranscriptEvent { tidx, kind: RecursionTranscriptEventKind::Sample, value }
    }

    fn two_block_record() -> RecursionRecord {
        let mut record = RecursionRecord::default();
        let mut input = [F::zero(); POSEIDON2_WIDTH];
        for i in 0..8 {
            input[i] = F::from_canonical_usize(i + 1);
        }
        let first_output = record.poseidon2_memo.permute(input);

        let mut proof = RecursionProofRecord { proof_idx: 7, ..RecursionProofRecord::default() };
        for i in 0..8 {
            proof.transcript.events.push(observe(i, i + 1));
        }
        proof.transcript.events.push(sample(8, first_output[POSEIDON2_WIDTH - 1]));
        for i in 0..8 {
            proof.transcript.events.push(observe(9 + i, 100 + i));
        }
        proof.transcript.sponge_blocks =
            SpecSponge::replay(7, &proof.transcript.events, &record.poseidon2_memo)
                .expect("synthetic transcript must finalize at its source");

        record.proof_records.push(proof);
        record
    }

    #[test]
    fn symbolic_analysis() {
        let air = TranscriptSpongeAir::default();
        let chip = polyair::Chip::<TranscriptSpongeAir, F, D_EF>::new(air);
        assert_eq!(chip.num_lookup(), 27);
        assert_eq!(chip.required_max_beta_power(), 34);
        assert_eq!(chip.degree, 3);
    }

    #[test]
    fn trace_and_dependencies_for_two_blocks() {
        let record = two_block_record();
        let rows = transcript_sponge_rows(&record);
        assert_eq!(rows.len(), 2);
        assert!(rows[0].is_proof_start);
        assert!(!rows[0].is_proof_last);
        assert_eq!(rows[0].tidx, 0);
        assert_eq!(rows[0].absorb_count, 8);
        assert_eq!(rows[0].squeeze_count, 1);
        assert_eq!(rows[1].tidx, 9);
        assert_eq!(rows[1].prev_s_count, 1);
        assert!(rows[1].is_proof_last);

        let trace = TranscriptSpongeTraceGenerator::generate_trace_row_major(&record);
        assert_eq!(trace.width(), NUM_TRANSCRIPT_SPONGE_COLS);
        assert_eq!(trace.height(), 2);
        let row = trace_row(&rows[0]);
        let cols: &TranscriptSpongeCols<F> = row.as_slice().borrow();
        assert_eq!(cols.proof_idx, F::from_canonical_usize(7));
        assert_eq!(cols.is_valid, F::one());
        assert_eq!(cols.squeeze_mask[POSEIDON2_WIDTH - 1], F::one());

        let mut deps = RecursionRecord::default();
        TranscriptSpongeAir::default().generate_dependencies(&record, &mut deps);
        assert_eq!(deps.poseidon2.total_count(), 2);
    }

    #[test]
    fn finalized_blocks_are_the_row_count_authority() {
        let mut record = two_block_record();
        let sponge_blocks = {
            let proof = &record.proof_records[0];
            SpecSponge::replay(proof.proof_idx, &proof.transcript.events, &record.poseidon2_memo)
                .unwrap()
        };
        let proof = &mut record.proof_records[0];
        proof.transcript.sponge_blocks = sponge_blocks;
        proof.transcript.events.clear();

        assert_eq!(transcript_sponge_row_count(&record), 2);
        assert_eq!(TranscriptSpongeTraceGenerator::trace_height(&record), 2);
        assert_eq!(transcript_sponge_rows(&record).len(), 2);
    }

    #[test]
    fn empty_record_trace_is_padding_only() {
        let record = RecursionRecord::default();
        let trace = TranscriptSpongeTraceGenerator::generate_trace_row_major(&record);
        assert_eq!(trace.width(), NUM_TRANSCRIPT_SPONGE_COLS);
        assert_eq!(trace.height(), 1);
        assert!(trace.row_slice(0).iter().all(|value| *value == F::zero()));
    }
}
