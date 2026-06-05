use std::{array, cell::UnsafeCell, mem::MaybeUninit, ops::Add, sync::Arc};

use super::{
    machine::RecursionAirEventCount, BaseAluEvent, CommitPublicValuesEvent, ExtAluEvent,
    ExtExpReverseBitsEvent, MemEvent, PolyEvalEvent, Poseidon2Event, PrefixSumChecksEvent,
    RecursionProgram, RecursionPublicValues, SelectEvent,
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
        self.prefix_sum_checks_events.reserve(event_counts.prefix_sum_checks_events);
    }
}

/// Pre-allocated, fixed-size event record allowing lock-free parallel writes.
///
/// Safety invariant: each slot is written to exactly once by the instruction with the
/// corresponding offset (guaranteed by `AnalyzedInstruction`).
pub struct UnsafeRecord<F> {
    pub base_alu_events: Vec<UnsafeCell<MaybeUninit<BaseAluEvent<F>>>>,
    pub ext_alu_events: Vec<UnsafeCell<MaybeUninit<ExtAluEvent<F>>>>,
    pub mem_const_count: usize,
    pub mem_var_events: Vec<UnsafeCell<MaybeUninit<MemEvent<F>>>>,
    pub poseidon2_events: Vec<UnsafeCell<MaybeUninit<Poseidon2Event<F>>>>,
    pub poseidon2_skinny_events: Vec<UnsafeCell<MaybeUninit<Poseidon2Event<F>>>>,
    pub select_events: Vec<UnsafeCell<MaybeUninit<SelectEvent<F>>>>,
    pub commit_pv_hash_events: Vec<UnsafeCell<MaybeUninit<CommitPublicValuesEvent<F>>>>,
}

unsafe impl<F> Sync for UnsafeRecord<F> {}

impl<F> UnsafeRecord<F> {
    pub fn new(event_counts: &RecursionAirEventCount) -> Self {
        #[inline]
        fn create_uninit_vec<T>(len: usize) -> Vec<UnsafeCell<MaybeUninit<T>>> {
            let mut vec = Vec::with_capacity(len);
            unsafe { vec.set_len(len) };
            vec
        }

        Self {
            base_alu_events: create_uninit_vec(event_counts.base_alu_events),
            ext_alu_events: create_uninit_vec(event_counts.ext_alu_events),
            mem_const_count: event_counts.mem_const_events,
            mem_var_events: create_uninit_vec(event_counts.mem_var_events),
            poseidon2_events: create_uninit_vec(event_counts.poseidon2_wide_events),
            poseidon2_skinny_events: create_uninit_vec(event_counts.poseidon2_skinny_events),
            select_events: create_uninit_vec(event_counts.select_events),
            commit_pv_hash_events: create_uninit_vec(event_counts.commit_pv_hash_events),
        }
    }

    /// Convert the fully-initialized UnsafeRecord into an ExecutionRecord.
    ///
    /// # Safety
    /// All slots must have been written to exactly once by the executor.
    #[allow(clippy::missing_transmute_annotations)]
    pub unsafe fn into_record(
        self,
        program: Arc<RecursionProgram<F>>,
        public_values: RecursionPublicValues<F>,
        poly_eval_events: Vec<PolyEvalEvent<F>>,
        ext_exp_reverse_bits_events: Vec<ExtExpReverseBitsEvent<F>>,
        prefix_sum_checks_events: Vec<PrefixSumChecksEvent<F>>,
    ) -> ExecutionRecord<F> {
        ExecutionRecord {
            program,
            index: 0,
            base_alu_events: std::mem::transmute(self.base_alu_events),
            ext_alu_events: std::mem::transmute(self.ext_alu_events),
            mem_const_count: self.mem_const_count,
            mem_var_events: std::mem::transmute(self.mem_var_events),
            public_values,
            poseidon2_events: std::mem::transmute(self.poseidon2_events),
            poseidon2_skinny_events: std::mem::transmute(self.poseidon2_skinny_events),
            select_events: std::mem::transmute(self.select_events),
            commit_pv_hash_events: std::mem::transmute(self.commit_pv_hash_events),
            poly_eval_events,
            ext_exp_reverse_bits_events,
            prefix_sum_checks_events,
        }
    }
}
