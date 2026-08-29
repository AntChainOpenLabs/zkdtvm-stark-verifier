use core::borrow::BorrowMut;

use dt_stark::sumcheck::trace::{CompressedMatrix, PaddingRow};
use p3_field::AbstractField;
use p3_matrix::dense::RowMajorMatrix;

use crate::{
    config::F, primitives_dt::range::air::RangeCheckerCols, system_dt::RecursionRangePool,
};

#[derive(Debug, Default, Clone, Copy)]
pub struct RangeCheckerTraceGenerator<const NUM_BITS: usize>;

impl<const NUM_BITS: usize> RangeCheckerTraceGenerator<NUM_BITS> {
    pub fn trace_height(pool: &RecursionRangePool) -> usize {
        pool.requests_for_bits(NUM_BITS).count().max(1).next_power_of_two()
    }

    pub fn generate_trace_compressed_from_pool(pool: &RecursionRangePool) -> CompressedMatrix<F> {
        let mut requests = pool.requests_for_bits(NUM_BITS).collect::<Vec<_>>();
        requests.sort_unstable_by_key(|request| request.value);
        if requests.is_empty() {
            let main =
                RowMajorMatrix::new(Self::zero_row(), RangeCheckerCols::<u8, NUM_BITS>::width());
            return CompressedMatrix::new(main, PaddingRow::None, 1);
        }

        let height = requests.len().max(1).next_power_of_two();
        let width = RangeCheckerCols::<u8, NUM_BITS>::width();
        let mut trace = Vec::with_capacity(width * requests.len());
        for request in &requests {
            Self::push_row(&mut trace, request.value, request.count);
        }

        let main = RowMajorMatrix::new(trace, width);
        let padding = if requests.len() < height {
            PaddingRow::General(Self::zero_row())
        } else {
            PaddingRow::None
        };
        CompressedMatrix::new(main, padding, height)
    }

    fn zero_row() -> Vec<F> {
        let mut row = Vec::with_capacity(RangeCheckerCols::<u8, NUM_BITS>::width());
        Self::push_row(&mut row, 0, 0);
        row
    }

    fn push_row(trace: &mut Vec<F>, value: usize, mult: u32) {
        let width = RangeCheckerCols::<u8, NUM_BITS>::width();
        let mut row = vec![F::zero(); width];
        let cols: &mut RangeCheckerCols<F, NUM_BITS> = row.as_mut_slice().borrow_mut();
        cols.value = F::from_canonical_usize(value);
        for (i, bit) in cols.bits.iter_mut().enumerate() {
            *bit = F::from_bool(((value >> i) & 1) == 1);
        }
        cols.mult = F::from_canonical_u32(mult);
        trace.extend(row);
    }
}
