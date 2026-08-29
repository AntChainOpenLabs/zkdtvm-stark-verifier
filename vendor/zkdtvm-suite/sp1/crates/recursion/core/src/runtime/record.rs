use std::{array, ops::Add, sync::Arc};

use super::{
    machine::RecursionAirEventCount, BaseAluEvent, CommitPublicValuesEvent, ExtAluEvent,
    ExtExpReverseBitsEvent, MemEvent, PolyEvalEvent, Poseidon2Event, PrefixSumChecksEvent,
    RecursionProgram, RecursionPublicValues, SelectEvent, SumcheckRoundEvent,
};

use dt_stark::{air::MachineAir, DTCoreOpts, MachineRecord, PROOF_MAX_NUM_PVS};
use p3_field::{AbstractField, Field};

#[derive(Clone, Default, Debug)]
pub struct ExecutionRecord<F> {
    pub program: Arc<RecursionProgram<F>>,
    /// The index of the shard.
    pub index: u32,

    pub base_alu_events: Vec<BaseAluEvent<F>>,
    pub ext_alu_events: Vec<ExtAluEvent<F>>,
    pub mem_const_count: usize,
    pub mem_var_events: Vec<MemEvent<F>>,
    /// The public values.
    pub public_values: RecursionPublicValues<F>,

    pub poseidon2_events: Vec<Poseidon2Event<F>>,
    /// Events for the skinny Poseidon2 chip (one-round-per-row layout).
    /// Shared by both BabyBear `Poseidon2SkinnyChip` and KoalaBear `Poseidon2SkinnyKbChip`;
    /// only one of them is registered at a time depending on the active cargo feature.
    pub poseidon2_skinny_events: Vec<Poseidon2Event<F>>,
    pub select_events: Vec<SelectEvent<F>>,
    pub commit_pv_hash_events: Vec<CommitPublicValuesEvent<F>>,
    pub poly_eval_events: Vec<PolyEvalEvent<F>>,
    pub ext_exp_reverse_bits_events: Vec<ExtExpReverseBitsEvent<F>>,
    pub sumcheck_round_events: Vec<SumcheckRoundEvent<F>>,
    pub prefix_sum_checks_events: Vec<PrefixSumChecksEvent<F>>,
}

impl<F: Field> MachineRecord for ExecutionRecord<F> {
    type Config = DTCoreOpts;

    fn stats(&self) -> hashbrown::HashMap<String, usize> {
        [
            ("base_alu_events", self.base_alu_events.len()),
            ("ext_alu_events", self.ext_alu_events.len()),
            ("mem_const_count", self.mem_const_count),
            ("mem_var_events", self.mem_var_events.len()),
            ("poseidon2_events", self.poseidon2_events.len()),
            ("poseidon2_skinny_events", self.poseidon2_skinny_events.len()),
            ("select_events", self.select_events.len()),
            ("commit_pv_hash_events", self.commit_pv_hash_events.len()),
            ("poly_eval_events", self.poly_eval_events.len()),
            ("ext_exp_reverse_bits_events", self.ext_exp_reverse_bits_events.len()),
            ("sumcheck_round_events", self.sumcheck_round_events.len()),
            ("prefix_sum_checks_events", self.prefix_sum_checks_events.len()),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_owned(), v))
        .collect()
    }

    fn append(&mut self, other: &mut Self) {
        // Exhaustive destructuring for refactoring purposes.
        let Self {
            program: _,
            index: _,
            base_alu_events,
            ext_alu_events,
            mem_const_count,
            mem_var_events,
            public_values: _,
            poseidon2_events,
            poseidon2_skinny_events,
            select_events,
            commit_pv_hash_events,
            poly_eval_events,
            ext_exp_reverse_bits_events,
            sumcheck_round_events,
            prefix_sum_checks_events,
        } = self;
        base_alu_events.append(&mut other.base_alu_events);
        ext_alu_events.append(&mut other.ext_alu_events);
        *mem_const_count += other.mem_const_count;
        mem_var_events.append(&mut other.mem_var_events);
        poseidon2_events.append(&mut other.poseidon2_events);
        poseidon2_skinny_events.append(&mut other.poseidon2_skinny_events);
        select_events.append(&mut other.select_events);
        commit_pv_hash_events.append(&mut other.commit_pv_hash_events);
        poly_eval_events.append(&mut other.poly_eval_events);
        ext_exp_reverse_bits_events.append(&mut other.ext_exp_reverse_bits_events);
        sumcheck_round_events.append(&mut other.sumcheck_round_events);
        prefix_sum_checks_events.append(&mut other.prefix_sum_checks_events);
    }

    fn public_values<T: AbstractField>(&self) -> Vec<T> {
        let pv_elms = self.public_values.as_array();

        let ret: [T; PROOF_MAX_NUM_PVS] = array::from_fn(|i| {
            if i < pv_elms.len() {
                T::from_canonical_u32(pv_elms[i].as_u32())
            } else {
                T::zero()
            }
        });

        ret.to_vec()
    }
}

impl<F: Field> ExecutionRecord<F> {
    #[inline]
    pub fn fixed_log2_rows<A: MachineAir<F>>(&self, air: &A) -> Option<usize> {
        self.program.fixed_log2_rows(air)
    }

    pub fn preallocate(&mut self) {
        let event_counts =
            self.program.inner.iter().fold(RecursionAirEventCount::default(), Add::add);
        self.poseidon2_events.reserve(event_counts.poseidon2_wide_events);
        self.poseidon2_skinny_events.reserve(event_counts.poseidon2_skinny_events);
        self.mem_var_events.reserve(event_counts.mem_var_events);
        self.base_alu_events.reserve(event_counts.base_alu_events);
        self.ext_alu_events.reserve(event_counts.ext_alu_events);
        self.select_events.reserve(event_counts.select_events);
        self.poly_eval_events.reserve(event_counts.poly_eval_events);
        self.ext_exp_reverse_bits_events.reserve(event_counts.ext_exp_reverse_bits_events);
        self.sumcheck_round_events.reserve(event_counts.sumcheck_round_events);
        self.prefix_sum_checks_events.reserve(event_counts.prefix_sum_checks_events);
    }
}
