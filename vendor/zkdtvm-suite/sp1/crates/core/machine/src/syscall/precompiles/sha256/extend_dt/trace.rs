use dt_core_executor::{
    events::{ByteLookupEvent, ByteRecord, PrecompileEvent, ShaExtendEvent},
    syscalls::SyscallCode,
    ExecutionRecord, Program,
};
use dt_stark::{
    air::MachineAir,
    sumcheck::trace::{CompressedMatrix, PaddingRow},
};
use hashbrown::HashMap;
use itertools::Itertools;
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::{ParallelIterator, ParallelSlice};
use std::borrow::BorrowMut;
use typenum::U4;

use crate::{
    operations_dt::AddNOperationWithoutResult,
    syscall::precompiles::sha256::extend_dt::{ShaExtendChip, ShaExtendCols, NUM_SHA_EXTEND_COLS},
    utils::{next_power_of_two, padded_rows_threshold},
};

impl<F: Field> MachineAir<F> for ShaExtendChip {
    type Record = ExecutionRecord;

    type Program = Program;

    fn name(&self) -> String {
        "ShaExtend".to_string()
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _: &mut ExecutionRecord,
    ) -> CompressedMatrix<F> {
        let events = input.get_precompile_events(SyscallCode::SHA_EXTEND);
        let real_nb_rows = events.len() * 48;

        let mut rows = vec![[F::zero(); NUM_SHA_EXTEND_COLS]; real_nb_rows];

        for (i, (_, event)) in events.iter().enumerate() {
            let event =
                if let PrecompileEvent::ShaExtend(event) = event { event } else { unreachable!() };
            self.event_to_rows(
                event,
                unsafe { &mut *(rows.as_mut_ptr() as *mut [[F; NUM_SHA_EXTEND_COLS]; 48]).add(i) },
                &mut HashMap::new(),
            );
        }

        let size_log2 = input.fixed_log2_rows::<F, _>(self);
        let mut padded_nb_rows = next_power_of_two(real_nb_rows, size_log2);
        padded_nb_rows = padded_rows_threshold(padded_nb_rows);

        let main = RowMajorMatrix::new(
            rows.into_iter().flatten().collect::<Vec<_>>(),
            NUM_SHA_EXTEND_COLS,
        );
        CompressedMatrix::new(main, PaddingRow::Zero { width: NUM_SHA_EXTEND_COLS }, padded_nb_rows)
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        let events = input.get_precompile_events(SyscallCode::SHA_EXTEND);
        let chunk_size = std::cmp::max(events.len() / num_cpus::get(), 1);

        let blu_batches = events
            .par_chunks(chunk_size)
            .map(|events| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                events.iter().for_each(|(_, event)| {
                    let event = if let PrecompileEvent::ShaExtend(event) = event {
                        event
                    } else {
                        unreachable!()
                    };
                    self.event_to_rows::<F>(
                        event,
                        &mut [[F::zero(); NUM_SHA_EXTEND_COLS]; 48],
                        &mut blu,
                    );
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_batches.iter().collect_vec());
    }

    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            !shard.get_precompile_events(SyscallCode::SHA_EXTEND).is_empty()
        }
    }
}

impl ShaExtendChip {
    fn event_to_rows<F: Field>(
        &self,
        event: &ShaExtendEvent,
        rows: &mut [[F; NUM_SHA_EXTEND_COLS]; 48],
        blu: &mut impl ByteRecord,
    ) {
        for i in 0..48usize {
            let cols: &mut ShaExtendCols<F> = rows[i].as_mut_slice().borrow_mut();
            cols.is_real = F::one();
            cols.shard = F::from_canonical_u32(event.shard);
            cols.clk = F::from_canonical_u32(event.clk);
            cols.w_ptr = F::from_canonical_u32(event.w_ptr);
            cols.i = F::from_canonical_u32((16 + i) as u32);

            cols.w_i_minus_15.populate(event.w_i_minus_15_reads[i], blu);
            cols.w_i_minus_2.populate(event.w_i_minus_2_reads[i], blu);
            cols.w_i_minus_16.populate(event.w_i_minus_16_reads[i], blu);
            cols.w_i_minus_7.populate(event.w_i_minus_7_reads[i], blu);

            // `s0 := (w[i-15] rightrotate 7) xor (w[i-15] rightrotate 18) xor (w[i-15] rightshift
            // 3)`.
            let w_i_minus_15 = event.w_i_minus_15_reads[i].value;
            let w_i_minus_15_rr_7 = cols.w_i_minus_15_rr_7.populate(blu, w_i_minus_15, 7);
            let w_i_minus_15_rr_18 = cols.w_i_minus_15_rr_18.populate(blu, w_i_minus_15, 18);
            let w_i_minus_15_rs_3 = cols.w_i_minus_15_rs_3.populate(blu, w_i_minus_15, 3);

            cols.w_i_minus_15_rr_7_witness = w_i_minus_15_rr_7.into();
            cols.w_i_minus_15_rr_18_witness = w_i_minus_15_rr_18.into();
            cols.w_i_minus_15_rs_3_witness = w_i_minus_15_rs_3.into();

            let s0 =
                cols.s0.populate(blu, [w_i_minus_15_rr_7, w_i_minus_15_rr_18, w_i_minus_15_rs_3]);

            // `s1 := (w[i-2] rightrotate 17) xor (w[i-2] rightrotate 19) xor (w[i-2] rightshift
            // 10)`.
            let w_i_minus_2 = event.w_i_minus_2_reads[i].value;
            let w_i_minus_2_rr_17 = cols.w_i_minus_2_rr_17.populate(blu, w_i_minus_2, 17);
            let w_i_minus_2_rr_19 = cols.w_i_minus_2_rr_19.populate(blu, w_i_minus_2, 19);
            let w_i_minus_2_rs_10 = cols.w_i_minus_2_rs_10.populate(blu, w_i_minus_2, 10);

            cols.w_i_minus_2_rr_17_witness = w_i_minus_2_rr_17.into();
            cols.w_i_minus_2_rr_19_witness = w_i_minus_2_rr_19.into();
            cols.w_i_minus_2_rs_10_witness = w_i_minus_2_rs_10.into();

            let s1 =
                cols.s1.populate(blu, [w_i_minus_2_rr_17, w_i_minus_2_rr_19, w_i_minus_2_rs_10]);

            // Compute `s2`.
            let w_i_minus_7 = event.w_i_minus_7_reads[i].value;
            let w_i_minus_16 = event.w_i_minus_16_reads[i].value;
            AddNOperationWithoutResult::<F, U4>::populate(blu, [w_i_minus_16, s0, w_i_minus_7, s1]);

            cols.w_i.populate(event.w_i_writes[i], blu);
        }
    }
}
