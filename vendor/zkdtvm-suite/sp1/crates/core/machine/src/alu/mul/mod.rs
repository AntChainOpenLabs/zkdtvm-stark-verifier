//! Implementation to check that b * c = product.
//!
//! We first extend the operands to 64 bits. We sign-extend them if the op code is signed. Then we
//! calculate the un-carried product and propagate the carry. Finally, we check that the appropriate
//! bits of the product match the result.
//!
//! b_64 = sign_extend(b) if signed operation else b
//! c_64 = sign_extend(c) if signed operation else c
//!
//! m = []
//! # 64-bit integers have 8 limbs.
//! # Calculate un-carried product.
//! for i in 0..8:
//!     for j in 0..8:
//!         if i + j < 8:
//!             m\[i + j\] += b_64\[i\] * c_64\[j\]
//!
//! # Propagate carry
//! for i in 0..8:
//!     x = m\[i\]
//!     if i > 0:
//!         x += carry\[i - 1\]
//!     carry\[i\] = x / 256
//!     m\[i\] = x % 256
//!
//! if upper_half:
//!     assert_eq(a, m\[4..8\])
//! if lower_half:
//!     assert_eq(a, m\[0..4\])

mod mul_polyair;
pub use mul_polyair::*;

use core::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use dt_core_executor::{
    events::{AluEvent, ByteLookupEvent, ByteRecord},
    ExecutionRecord, Opcode, Program, RTypeRecord, DEFAULT_PC_INC,
};
use dt_derive::AlignedBorrow;
use dt_stark::{
    air::MachineAir,
    sumcheck::trace::{CompressedMatrix, PaddingRow},
};
use hashbrown::HashMap;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::{ParallelBridge, ParallelIterator, ParallelSlice};

use crate::{
    adapter::{CPUState, RTypeRegisterOp},
    air::DTCoreAirBuilder,
    operations::MulOperation,
    utils::{next_power_of_two, padded_rows_threshold, zeroed_f_vec},
};

/// The number of main trace columns for `MulChip`.
pub const NUM_MUL_COLS: usize = size_of::<MulCols<u8>>();

/// A chip that implements multiplication for the multiplication opcodes.
#[derive(Default)]
pub struct MulChip;

/// The column layout for the chip.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct MulCols<T> {
    /// state
    pub cpu_state: CPUState<T>,
    /// mem ops
    pub mem_ops: RTypeRegisterOp<T>,
    // mul op
    pub mul_op: MulOperation<T>,

    /// Flag indicating whether the opcode is `MUL` (`u32 x u32`).
    pub is_mul: T,

    /// Flag indicating whether the opcode is `MULH` (`i32 x i32`, upper half).
    pub is_mulh: T,

    /// Flag indicating whether the opcode is `MULHU` (`u32 x u32`, upper half).
    pub is_mulhu: T,

    /// Flag indicating whether the opcode is `MULHSU` (`i32 x u32`, upper half).
    pub is_mulhsu: T,

    /// Selector to know whether this row is enabled.
    pub is_real: T,
}

impl<F: Field> MachineAir<F> for MulChip {
    type Record = ExecutionRecord;

    type Program = Program;

    fn name(&self) -> String {
        "Mul".to_string()
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _: &mut ExecutionRecord,
    ) -> CompressedMatrix<F> {
        let nb_rows = input.mul_events.len();
        let size_log2 = input.fixed_log2_rows::<F, _>(self);
        let padded_nb_rows = padded_rows_threshold(next_power_of_two(nb_rows, size_log2));

        let chunk_size = std::cmp::max((nb_rows + 1) / num_cpus::get(), 1);
        let shard = input.execution_shard();
        let mut values = zeroed_f_vec(nb_rows * NUM_MUL_COLS);
        values.chunks_mut(chunk_size * NUM_MUL_COLS).enumerate().par_bridge().for_each(
            |(i, rows)| {
                rows.chunks_mut(NUM_MUL_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut MulCols<F> = row.borrow_mut();
                    let mut byte_lookup_events = Vec::new();
                    let (record, event) = &input.mul_events[idx];
                    self.event_to_row(record, event, cols, &mut byte_lookup_events, shard);
                });
            },
        );

        let main = RowMajorMatrix::new(values, NUM_MUL_COLS);
        CompressedMatrix::new(main, PaddingRow::Zero { width: NUM_MUL_COLS }, padded_nb_rows)
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        let chunk_size = std::cmp::max(input.mul_events.len() / num_cpus::get(), 1);
        let shard = input.execution_shard();
        let blu_batches = input
            .mul_events
            .par_chunks(chunk_size)
            .map(|events| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                events.iter().for_each(|(record, event)| {
                    let mut row = [F::zero(); NUM_MUL_COLS];
                    let cols: &mut MulCols<F> = row.as_mut_slice().borrow_mut();
                    self.event_to_row(record, event, cols, &mut blu, shard);
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_batches.iter().collect::<Vec<_>>());
    }

    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            !shard.mul_events.is_empty()
        }
    }

    fn local_only(&self) -> bool {
        true
    }
}
impl MulChip {
    fn event_to_row<F: Field>(
        &self,
        record: &RTypeRecord,
        event: &AluEvent,
        cols: &mut MulCols<F>,
        blu: &mut impl ByteRecord,
        shard: u32,
    ) {
        cols.cpu_state.populate(blu, record.clk, event.pc, shard);
        cols.mem_ops.populate(blu, *record);

        // When rd=x0, skip MulOperation::populate to avoid generating byte lookup
        // events that would be unmatched in eval (perform_calc=0 skips all constraints).
        if !event.op_a_0 {
            cols.mul_op.populate(
                blu,
                event.b,
                event.c,
                event.opcode == Opcode::MULH,
                event.opcode == Opcode::MULHSU,
            );
        }

        cols.is_mul = F::from_bool(event.opcode == Opcode::MUL);
        cols.is_mulh = F::from_bool(event.opcode == Opcode::MULH);
        cols.is_mulhu = F::from_bool(event.opcode == Opcode::MULHU);
        cols.is_mulhsu = F::from_bool(event.opcode == Opcode::MULHSU);

        cols.is_real = F::one();
    }
}

impl<F> BaseAir<F> for MulChip {
    fn width(&self) -> usize {
        NUM_MUL_COLS
    }
}
impl<AB> Air<AB> for MulChip
where
    AB: DTCoreAirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &MulCols<AB::Var> = (*local).borrow();

        let execution_shard: AB::Expr = builder.current_shard().into();
        let shard: AB::Expr = local.cpu_state.shard.into();
        let clk: AB::Expr = local.cpu_state.clk::<AB>();
        let one: AB::Expr = AB::F::one().into();

        builder.assert_bool(local.is_real);

        // cpu state transition
        CPUState::<AB::F>::eval(
            builder,
            local.cpu_state,
            local.cpu_state.pc + AB::F::from_canonical_u32(DEFAULT_PC_INC),
            AB::Expr::from_canonical_u32(DEFAULT_PC_INC),
            local.is_real.into(),
            execution_shard,
        );

        // get opcode
        let opcode = {
            let mul = AB::F::from_canonical_u32(Opcode::MUL as u32);
            let mulh = AB::F::from_canonical_u32(Opcode::MULH as u32);
            let mulhu = AB::F::from_canonical_u32(Opcode::MULHU as u32);
            let mulhsu = AB::F::from_canonical_u32(Opcode::MULHSU as u32);
            builder
                .when(local.is_real)
                .assert_one(local.is_mul + local.is_mulh + local.is_mulhsu + local.is_mulhu);
            local.is_mul * mul +
                local.is_mulh * mulh +
                local.is_mulhu * mulhu +
                local.is_mulhsu * mulhsu
        };

        // is_real && (not op_a_zero) => eval mul op
        let perform_calc = local.is_real - local.mem_ops.op_a_zero;

        MulOperation::<AB::F>::eval(
            builder,
            (*local.mem_ops.op_a_value()).map(Into::into),
            (*local.mem_ops.op_b_value()).map(Into::into),
            (*local.mem_ops.op_c_value()).map(Into::into),
            local.mul_op,
            perform_calc,
            local.is_mul.into(),
            local.is_mulh.into(),
            local.is_mulhu.into(),
            local.is_mulhsu.into(),
        );

        // 4. 约束寄存器读写与指令合法性
        RTypeRegisterOp::<AB::F>::eval(
            builder,
            shard,
            clk,
            local.cpu_state.pc.into(),
            opcode,
            local.mem_ops,
            AB::Expr::zero(),
            local.is_real.into(),
        );

        // if not zero, op_a_zero => 0
        builder.when(one - local.is_real).assert_zero(local.mem_ops.op_a_zero);
    }
}
