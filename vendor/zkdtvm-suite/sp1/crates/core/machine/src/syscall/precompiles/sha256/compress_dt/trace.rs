use std::borrow::BorrowMut;

use dt_core_executor::{
    events::{ByteLookupEvent, ByteRecord, PrecompileEvent, ShaCompressEvent},
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

use crate::{
    operations_dt::NotOperation,
    syscall::precompiles::sha256::{
        compress_dt::SHA_COMPRESS_K, ShaCompressCols, NUM_SHA_COMPRESS_COLS,
    },
    utils::{next_power_of_two, padded_rows_threshold},
};

use super::ShaCompressChip;

impl<F: Field> MachineAir<F> for ShaCompressChip {
    type Record = ExecutionRecord;

    type Program = Program;

    fn name(&self) -> String {
        "ShaCompress".to_string()
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _: &mut ExecutionRecord,
    ) -> CompressedMatrix<F> {
        let events = input.get_precompile_events(SyscallCode::SHA_COMPRESS);
        let real_nb_rows = events.len() * 64;

        let mut rows = vec![[F::zero(); NUM_SHA_COMPRESS_COLS]; real_nb_rows];

        for (i, (_, event)) in events.iter().enumerate() {
            let event = if let PrecompileEvent::ShaCompress(event) = event {
                event
            } else {
                unreachable!()
            };
            self.event_to_rows(
                event,
                unsafe {
                    &mut *(rows.as_mut_ptr() as *mut [[F; NUM_SHA_COMPRESS_COLS]; 64]).add(i)
                },
                &mut HashMap::new(),
            );
        }

        let size_log2 = input.fixed_log2_rows::<F, _>(self);
        let mut padded_nb_rows = next_power_of_two(real_nb_rows, size_log2);
        padded_nb_rows = padded_rows_threshold(padded_nb_rows);

        let main = RowMajorMatrix::new(
            rows.into_iter().flatten().collect::<Vec<_>>(),
            NUM_SHA_COMPRESS_COLS,
        );
        CompressedMatrix::new(
            main,
            PaddingRow::Zero { width: NUM_SHA_COMPRESS_COLS },
            padded_nb_rows,
        )
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        let events = input.get_precompile_events(SyscallCode::SHA_COMPRESS);
        let chunk_size = std::cmp::max(events.len() / num_cpus::get(), 1);

        let blu_batches = events
            .par_chunks(chunk_size)
            .map(|events| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                events.iter().for_each(|(_, event)| {
                    let event = if let PrecompileEvent::ShaCompress(event) = event {
                        event
                    } else {
                        unreachable!()
                    };
                    self.event_to_rows::<F>(
                        event,
                        &mut [[F::zero(); NUM_SHA_COMPRESS_COLS]; 64],
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
            !shard.get_precompile_events(SyscallCode::SHA_COMPRESS).is_empty()
        }
    }
}

impl ShaCompressChip {
    fn event_to_rows<F: Field>(
        &self,
        event: &ShaCompressEvent,
        rows: &mut [[F; NUM_SHA_COMPRESS_COLS]; 64],
        blu: &mut impl ByteRecord,
    ) {
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = event.h;

        for i in 0..64usize {
            let cols: &mut ShaCompressCols<F> = rows[i].as_mut_slice().borrow_mut();

            cols.shard = F::from_canonical_u32(event.shard);
            cols.clk = F::from_canonical_u32(event.clk);
            cols.w_ptr = F::from_canonical_u32(event.w_ptr);

            cols.i = F::from_canonical_u32(i as u32);
            cols.i_low_one_hot[i & 0x7] = F::one();
            cols.i_high_one_hot[i >> 3] = F::one();

            cols.w_access.populate(event.w_i_read_records[i], blu);

            cols.a = a.into();
            cols.b = b.into();
            cols.c = c.into();
            cols.d = d.into();
            cols.e = e.into();
            cols.f = f.into();
            cols.g = g.into();
            cols.h = h.into();

            cols.a_witness = a.into();
            cols.b_witness = b.into();
            cols.c_witness = c.into();
            cols.e_witness = e.into();
            cols.f_witness = f.into();
            cols.g_witness = g.into();

            cols.k = SHA_COMPRESS_K[i].into();

            let e_rr_6 = cols.e_rr_6.populate(blu, e, 6);
            let e_rr_11 = cols.e_rr_11.populate(blu, e, 11);
            let e_rr_25 = cols.e_rr_25.populate(blu, e, 25);

            cols.e_rr_6_witness = e_rr_6.into();
            cols.e_rr_11_witness = e_rr_11.into();
            cols.e_rr_25_witness = e_rr_25.into();

            let s1 = cols.s1.populate(blu, [e_rr_6, e_rr_11, e_rr_25]);

            let e_and_f = cols.e_and_f.populate(blu, [e, f]);
            let e_not = NotOperation::<F>::populate(e);
            let e_not_and_g = cols.e_not_and_g.populate(blu, [e_not, g]);

            let ch = cols.ch.populate(blu, [e_and_f, e_not_and_g]);

            let temp1 = cols.temp1.populate(blu, [h, s1, ch, event.w[i], SHA_COMPRESS_K[i]]);

            let a_rr_2 = cols.a_rr_2.populate(blu, a, 2);
            let a_rr_13 = cols.a_rr_13.populate(blu, a, 13);
            let a_rr_22 = cols.a_rr_22.populate(blu, a, 22);

            cols.a_rr_2_witness = a_rr_2.into();
            cols.a_rr_13_witness = a_rr_13.into();
            cols.a_rr_22_witness = a_rr_22.into();

            let s0 = cols.s0.populate(blu, [a_rr_2, a_rr_13, a_rr_22]);

            let a_and_b = cols.a_and_b.populate(blu, [a, b]);
            let a_and_c = cols.a_and_c.populate(blu, [a, c]);
            let b_and_c = cols.b_and_c.populate(blu, [b, c]);

            let maj = cols.maj.populate(blu, [a_and_b, a_and_c, b_and_c]);

            let temp2 = cols.temp2.populate(blu, [s0, maj]);
            let d_add_temp1 = cols.d_add_temp1.populate(blu, [d, temp1]);
            let temp1_add_temp2 = cols.temp1_add_temp2.populate(blu, [temp1, temp2]);

            h = g;
            g = f;
            f = e;
            e = d_add_temp1;
            d = c;
            c = b;
            b = a;
            a = temp1_add_temp2;

            cols.is_real = F::one();
        }
    }
}
