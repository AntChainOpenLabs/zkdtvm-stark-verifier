//! Verifies left shift.
//!
//! This module implements left shift (b << c) as a combination of bit and byte shifts.
//!
//! The shift amount c is decomposed into two components:
//!
//! - num_bits_to_shift = c % 8: Represents the fine-grained bit-level shift.
//! - num_bytes_to_shift = c // 8: Represents the coarser byte-level shift.
//!
//! Bit shifting is done by multiplying b by 2^num_bits_to_shift. Byte shifting is done by shifting
//! words. The logic looks as follows:
//!
//! c = take the least significant 5 bits of c
//! num_bytes_to_shift = c // 8
//! num_bits_to_shift = c % 8
//!
//! # "Bit shift"
//! bit_shift_multiplier = pow(2, num_bits_to_shift)
//! bit_shift_result = bit_shift_multiplier * b
//!
//! # "Byte shift"
//! for i in range(WORD_SIZE):
//!     if i < num_bytes_to_shift:
//!         assert(a\[i\] == 0)
//!     else:
//!         assert(a\[i\] == bit_shift_result\[i - num_bytes_to_shift\])
//!
//! Notes:
//!
//! - Ideally, we would calculate b * pow(2, c), but pow(2, c) could overflow in F.
//! - Shifting by a multiple of 8 bits is easy (=num_bytes_to_shift) since we just shift words.

mod sll_polyair;
pub use sll_polyair::*;

use core::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use dt_core_executor::{
    events::{AluEvent, ByteLookupEvent, ByteRecord},
    ALUTypeRecord, ExecutionRecord, Opcode, Program, DEFAULT_PC_INC,
};
use dt_derive::AlignedBorrow;
use dt_primitives::consts::WORD_SIZE;
use dt_stark::{
    air::MachineAir,
    sumcheck::trace::{CompressedMatrix, PaddingRow},
};
use hashbrown::HashMap;
use itertools::Itertools;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::{ParallelIterator, ParallelSlice};

use crate::{
    adapter::{ALUTypeRegisterOp, CPUState},
    air::DTCoreAirBuilder,
    utils::{next_power_of_two, padded_rows_threshold},
};

/// The number of main trace columns for `ShiftLeft`.
pub const NUM_SHIFT_LEFT_COLS: usize = size_of::<ShiftLeftCols<u8>>();

/// The number of bits in a byte.
pub const BYTE_SIZE: usize = 8;

/// A chip that implements bitwise operations for the opcodes SLL and SLLI.
#[derive(Default)]
pub struct ShiftLeft;

/// The column layout for the chip.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct ShiftLeftCols<T> {
    /// cpu state
    pub cpu_state: CPUState<T>,
    ///register operation
    pub mem_ops: ALUTypeRegisterOp<T>,

    /// The least significant byte of `c`. Used to verify `shift_by_n_bits` and `shift_by_n_bytes`.
    pub c_least_sig_byte: [T; BYTE_SIZE],

    /// A boolean array whose `i`th element indicates whether `num_bits_to_shift = i`.
    pub shift_by_n_bits: [T; BYTE_SIZE],

    /// The number to multiply to shift `b` by `num_bits_to_shift`. (i.e., `2^num_bits_to_shift`)
    pub bit_shift_multiplier: T,

    /// The result of multiplying `b` by `bit_shift_multiplier`.
    pub bit_shift_result: [T; WORD_SIZE],

    /// The carry propagated when multiplying `b` by `bit_shift_multiplier`.
    pub bit_shift_result_carry: [T; WORD_SIZE],

    /// A boolean array whose `i`th element indicates whether `num_bytes_to_shift = i`.
    pub shift_by_n_bytes: [T; WORD_SIZE],

    pub is_real: T,
}

impl<F: Field> MachineAir<F> for ShiftLeft {
    type Record = ExecutionRecord;

    type Program = Program;

    fn name(&self) -> String {
        "ShiftLeft".to_string()
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _: &mut ExecutionRecord,
    ) -> CompressedMatrix<F> {
        let shift_left_events = input.shift_left_events.clone();
        let shard = input.execution_shard();
        let mut rows: Vec<[F; NUM_SHIFT_LEFT_COLS]> = Vec::with_capacity(shift_left_events.len());
        for (record, event) in shift_left_events.iter() {
            let mut row = [F::zero(); NUM_SHIFT_LEFT_COLS];
            let cols: &mut ShiftLeftCols<F> = row.as_mut_slice().borrow_mut();
            let mut blu = Vec::new();
            self.event_to_row(record, event, cols, &mut blu, shard);
            rows.push(row);
        }

        let real_nb_rows = rows.len();
        let size_log2 = input.fixed_log2_rows::<F, _>(self);
        let padded_nb_rows = padded_rows_threshold(next_power_of_two(real_nb_rows, size_log2));

        let padded_row_template = {
            let mut row = [F::zero(); NUM_SHIFT_LEFT_COLS];
            let cols: &mut ShiftLeftCols<F> = row.as_mut_slice().borrow_mut();
            cols.shift_by_n_bits[0] = F::one();
            cols.shift_by_n_bytes[0] = F::one();
            cols.bit_shift_multiplier = F::one();
            row.to_vec()
        };

        let main = RowMajorMatrix::new(
            rows.into_iter().flatten().collect::<Vec<_>>(),
            NUM_SHIFT_LEFT_COLS,
        );
        CompressedMatrix::new(main, PaddingRow::General(padded_row_template), padded_nb_rows)
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        let chunk_size = std::cmp::max(input.shift_left_events.len() / num_cpus::get(), 1);
        let shard = input.execution_shard();
        let blu_batches = input
            .shift_left_events
            .par_chunks(chunk_size)
            .map(|events| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                events.iter().for_each(|(record, event)| {
                    let mut row = [F::zero(); NUM_SHIFT_LEFT_COLS];
                    let cols: &mut ShiftLeftCols<F> = row.as_mut_slice().borrow_mut();
                    self.event_to_row(record, event, cols, &mut blu, shard);
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
            !shard.shift_left_events.is_empty()
        }
    }

    fn padding_row(&self) -> Vec<F> {
        let mut row = [F::zero(); NUM_SHIFT_LEFT_COLS];
        let cols: &mut ShiftLeftCols<F> = row.as_mut_slice().borrow_mut();
        cols.shift_by_n_bits[0] = F::one();
        cols.shift_by_n_bytes[0] = F::one();
        cols.bit_shift_multiplier = F::one();
        row.to_vec()
    }

    fn local_only(&self) -> bool {
        true
    }
}
impl ShiftLeft {
    fn event_to_row<F: Field>(
        &self,
        record: &ALUTypeRecord,
        event: &AluEvent,
        cols: &mut ShiftLeftCols<F>,
        blu: &mut impl ByteRecord,
        shard: u32,
    ) {
        cols.cpu_state.populate(blu, record.clk, event.pc, shard);
        cols.mem_ops.populate(blu, *record);

        let b = event.b.to_le_bytes();
        let shamt = (event.c & 0x1F) as usize; // RISC-V SLL 只取低 5 位
        let bit_shift = shamt % 8;
        let byte_shift = shamt / 8;

        // c least byte to bits
        for i in 0..8 {
            cols.c_least_sig_byte[i] = F::from_canonical_u32((event.c >> i) & 1);
        }

        for i in 0..8 {
            cols.shift_by_n_bits[i] = F::from_bool(bit_shift == i);
        }
        let multiplier = 1u32 << bit_shift;
        cols.bit_shift_multiplier = F::from_canonical_u32(multiplier);

        let mut carry = 0u32;
        let mut res_bytes = [0u8; 4];
        let mut carry_bytes = [0u8; 4];
        for i in 0..4 {
            let v = (b[i] as u32) * multiplier + carry;
            res_bytes[i] = (v % 256) as u8;
            carry = v / 256;
            carry_bytes[i] = carry as u8;
        }
        cols.bit_shift_result = res_bytes.map(F::from_canonical_u8);
        cols.bit_shift_result_carry = carry_bytes.map(F::from_canonical_u8);

        for i in 0..4 {
            cols.shift_by_n_bytes[i] = F::from_bool(byte_shift == i);
        }

        cols.is_real = F::one();

        blu.add_u8_range_checks(&res_bytes);
        blu.add_u8_range_checks(&carry_bytes);
    }
}

impl<F> BaseAir<F> for ShiftLeft {
    fn width(&self) -> usize {
        NUM_SHIFT_LEFT_COLS
    }
}
impl<AB> Air<AB> for ShiftLeft
where
    AB: DTCoreAirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &ShiftLeftCols<AB::Var> = (*local).borrow();

        let execution_shard: AB::Expr = builder.current_shard().into();
        let shard: AB::Expr = local.cpu_state.shard.into();
        let one: AB::Expr = AB::F::one().into();
        let zero: AB::Expr = AB::F::zero().into();
        let base = AB::F::from_canonical_u32(256);

        builder.assert_bool(local.is_real);

        //cpu state
        CPUState::<AB::F>::eval(
            builder,
            local.cpu_state,
            local.cpu_state.pc + AB::F::from_canonical_u32(DEFAULT_PC_INC),
            AB::Expr::from_canonical_u32(DEFAULT_PC_INC),
            local.is_real.into(),
            execution_shard,
        );

        let opcode = AB::F::from_canonical_u32(Opcode::SLL as u32);

        // reconstruct c[0], lower 3bits->bit level shift, [3..5)->byte level shift
        let c_word = local.mem_ops.op_c_value();
        let mut reconstructed_c0 = zero.clone();
        for i in 0..8 {
            builder.assert_bool(local.c_least_sig_byte[i]);
            reconstructed_c0 =
                reconstructed_c0 + local.c_least_sig_byte[i] * AB::F::from_canonical_u32(1 << i);
        }
        //if not real, rec_c0 = 0, c_word[0] = 0u8
        builder.assert_eq(reconstructed_c0, c_word[0]);

        // bit_shift = c_bits[0..3]
        let bit_shift_amount = local.c_least_sig_byte[0] +
            local.c_least_sig_byte[1] * AB::F::from_canonical_u32(2) +
            local.c_least_sig_byte[2] * AB::F::from_canonical_u32(4);

        let mut sum_bit_flags = zero.clone();
        for i in 0..BYTE_SIZE {
            builder.assert_bool(local.shift_by_n_bits[i]);
            builder
                .when(local.shift_by_n_bits[i])
                .assert_eq(bit_shift_amount.clone(), AB::F::from_canonical_usize(i));
            builder
                .when(local.shift_by_n_bits[i])
                .assert_eq(local.bit_shift_multiplier, AB::F::from_canonical_u32(1 << i));
            sum_bit_flags = sum_bit_flags + local.shift_by_n_bits[i];
        }

        builder.assert_eq(one.clone(), sum_bit_flags);
        // bit shift check: b * multiplier + last_carry = res + carry * base
        let b = local.mem_ops.op_b_value();
        for i in 0..WORD_SIZE {
            let mut v = b[i] * local.bit_shift_multiplier - local.bit_shift_result_carry[i] * base;
            if i > 0 {
                v = v.clone() + local.bit_shift_result_carry[i - 1].into();
            }
            builder.assert_eq(local.bit_shift_result[i], v);
        }

        // byte_shift = c_bits[3..5)
        let byte_shift_amount =
            local.c_least_sig_byte[3] + local.c_least_sig_byte[4] * AB::F::from_canonical_u32(2);
        let mut sum_byte_flags = zero.clone();
        for i in 0..WORD_SIZE {
            builder.assert_bool(local.shift_by_n_bytes[i]);
            builder
                .when(local.shift_by_n_bytes[i])
                .assert_eq(byte_shift_amount.clone(), AB::F::from_canonical_usize(i));
            sum_byte_flags = sum_byte_flags + local.shift_by_n_bytes[i];
        }
        builder.assert_eq(sum_byte_flags, one.clone());

        let a = local.mem_ops.op_a_value();
        let perform_calc = local.is_real - local.mem_ops.op_a_zero;

        for n in 0..WORD_SIZE {
            let mut guard = builder.when(perform_calc.clone());
            let mut shifting = guard.when(local.shift_by_n_bytes[n]);
            for i in 0..WORD_SIZE {
                if i < n {
                    shifting.assert_zero(a[i]);
                } else {
                    shifting.assert_eq(a[i], local.bit_shift_result[i - n]);
                }
            }
        }
        builder.slice_range_check_u8(&local.bit_shift_result, local.is_real);
        builder.slice_range_check_u8(&local.bit_shift_result_carry, local.is_real);
        ALUTypeRegisterOp::<AB::F>::eval(
            builder,
            shard,
            local.cpu_state.clk::<AB>(),
            local.cpu_state.pc.into(),
            opcode,
            local.mem_ops,
            local.is_real.into(),
        );

        builder.when(one - local.is_real).assert_zero(local.mem_ops.op_a_zero);
    }
}
