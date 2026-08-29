use core::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use dt_core_executor::{
    events::{AluEvent, ByteLookupEvent, ByteRecord},
    AddiRecord, ExecutionRecord, Opcode, Program, DEFAULT_PC_INC,
};
use dt_derive::AlignedBorrow;
use dt_stark::{
    air::{DTAirBuilder, MachineAir},
    sumcheck::trace::{CompressedMatrix, PaddingRow},
    Word,
};
use hashbrown::HashMap;
use itertools::Itertools;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::{ParallelBridge, ParallelIterator};

use crate::{
    adapter::{AddiRegisterOp, CPUState},
    air::WordAirBuilder,
    memory::MemoryCols,
    operations::AddOperation,
    utils::{next_power_of_two, padded_rows_threshold, zeroed_f_vec},
};

/// The number of main trace columns for `AddSubChip`.
pub const NUM_ADDI_COLS: usize = size_of::<AddiCols<u8>>();

/// A chip that implements addition for the opcode ADD
#[derive(Default)]
pub struct AddiChip;

/// The column layout for the chip.
#[derive(AlignedBorrow, Default, Clone, Copy)]
#[repr(C)]
pub struct AddiCols<T> {
    /// Instance of `AddOperation` to handle addition logic in `AddSubChip`'s ALU operations.
    /// It's result will be `a` for the add operation and `b` for the sub operation.
    pub add_operation: AddOperation<T>,
    /// The effective b operand (byte-decomposed).
    /// When is_imm_b=0 (ADDI): equals the register read value from op_b_access.
    /// When is_imm_b=1 (LUI): equals the byte decomposition of the op_b immediate.
    /// Stored as a trace column to keep the operand at degree 1, avoiding the
    /// degree-5 violation that occurs when the conditional select expression (degree 2)
    /// flows into AddOperation's carry chain.
    pub effective_b: Word<T>,
    /// memory operations (supports both normal ADDI and LUI-as-ADD with optional b register)
    pub memory_operations: AddiRegisterOp<T>,
    ///cpu state
    pub cpu_state: CPUState<T>,
    /// Boolean to indicate whether the row is not a padding row.
    pub is_real: T,
}

impl<F: Field> MachineAir<F> for AddiChip {
    type Record = ExecutionRecord;

    type Program = Program;

    fn name(&self) -> String {
        "Addi".to_string()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        Some(padded_rows_threshold(next_power_of_two(
            input.addi_events.len(),
            input.fixed_log2_rows::<F, _>(self),
        )))
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _: &mut ExecutionRecord,
    ) -> CompressedMatrix<F> {
        let addi_events = input.addi_events.clone();
        let real_nb_rows = addi_events.len();
        let padded_nb_rows = <AddiChip as MachineAir<F>>::num_rows(self, input).unwrap();

        let chunk_size = std::cmp::max(real_nb_rows / num_cpus::get(), 1);
        let mut values = zeroed_f_vec(real_nb_rows * NUM_ADDI_COLS);

        values.chunks_mut(chunk_size * NUM_ADDI_COLS).enumerate().par_bridge().for_each(
            |(i, rows)| {
                rows.chunks_mut(NUM_ADDI_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut AddiCols<F> = row.borrow_mut();
                    let mut byte_lookup_events = Vec::new();
                    let (record, event) = &addi_events[idx];
                    self.event_to_row(
                        record,
                        event,
                        cols,
                        &mut byte_lookup_events,
                        input.execution_shard(),
                    );
                });
            },
        );

        let main = RowMajorMatrix::new(values, NUM_ADDI_COLS);
        CompressedMatrix::new(main, PaddingRow::Zero { width: NUM_ADDI_COLS }, padded_nb_rows)
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        let chunk_size = std::cmp::max(input.addi_events.len() / num_cpus::get(), 1);

        let event_iter = input.addi_events.chunks(chunk_size);
        let shard = input.execution_shard();
        let blu_batches = event_iter
            .par_bridge()
            .map(|events| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                events.iter().for_each(|(record, event)| {
                    let mut row = [F::zero(); NUM_ADDI_COLS];
                    let cols: &mut AddiCols<F> = row.as_mut_slice().borrow_mut();
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
            !shard.addi_events.is_empty()
        }
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl AddiChip {
    /// Create a row from an event.
    pub(crate) fn event_to_row<F: Field>(
        &self,
        record: &AddiRecord,
        event: &AluEvent,
        cols: &mut AddiCols<F>,
        blu: &mut impl ByteRecord,
        shard: u32,
    ) {
        cols.cpu_state.populate(blu, record.clk, event.pc, shard);
        cols.memory_operations.populate(blu, *record);
        // effective_b = the actual b operand value (byte-decomposed).
        // For ADDI: register value; for LUI: op_b immediate (always 0).
        cols.effective_b = Word::from(event.b);
        // When rd=x0, skip AddOperation::populate to avoid generating byte lookup
        // events that would be unmatched in eval (perform_calc=0 skips all constraints).
        if !event.op_a_0 {
            cols.add_operation.populate(blu, event.b, event.c);
        }
        cols.is_real = F::one();
    }
}

impl<F> BaseAir<F> for AddiChip {
    fn width(&self) -> usize {
        NUM_ADDI_COLS
    }
}

impl<AB> Air<AB> for AddiChip
where
    AB: DTAirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &AddiCols<AB::Var> = (*local).borrow();
        let execution_shard: AB::Expr = builder.current_shard().into();
        let shard: AB::Expr = local.cpu_state.shard.into();
        let clk: AB::Expr = local.cpu_state.clk::<AB>();
        //all the boolean flag should be checked in caller
        builder.assert_bool(local.is_real);
        CPUState::<AB::F>::eval(
            builder,
            local.cpu_state,
            local.cpu_state.pc + AB::F::from_canonical_u32(DEFAULT_PC_INC),
            AB::Expr::from_canonical_u32(DEFAULT_PC_INC),
            local.is_real.into(),
            execution_shard,
        );

        // Constrain effective_b: must equal the conditional select of register vs immediate.
        // effective_b[i] = reg_b[i] + is_imm_b * (imm_b[i] - reg_b[i])
        // Unconditional: padding rows are all zeros so 0 = 0 + 0*(0-0) holds naturally.
        // Degree: var(1) - var(1) - var(1)*(expr(1)-var(1)) = degree 2.
        let is_imm_b: AB::Expr = local.memory_operations.is_imm_b.into();
        let reg_b = local.memory_operations.op_b_access.value();
        let imm_b = Word::extend_expr::<AB>(local.memory_operations.op_b.into());
        for i in 0..dt_primitives::consts::WORD_SIZE {
            builder.assert_eq(
                local.effective_b[i],
                Into::<AB::Expr>::into(reg_b[i]) +
                    is_imm_b.clone() * (imm_b[i].clone() - Into::<AB::Expr>::into(reg_b[i])),
            );
        }

        let perform_calc = local.is_real - local.memory_operations.op_a_zero;
        // Evaluate the addition operation with degree-1 operands.
        // effective_b is now Word<AB::Var> (degree 1), so the carry chain stays
        // at degree 1, and when(is_real).assert_bool(carry) = degree 3.
        AddOperation::<AB::F>::eval(
            builder,
            local.effective_b,
            *local.memory_operations.op_c_value(),
            local.add_operation,
            perform_calc.clone(),
        );
        // Constrain: value written to rd == addition result.
        // Unconditional: on padding rows both sides are zero (zeroed_f_vec);
        // on op_a_zero=1 rows both sides are forced to zero (assert_word_zero + skip populate).
        builder.assert_word_eq(*local.memory_operations.op_a_value(), local.add_operation.value);
        //if is not real, op_a_zero should be 0
        builder
            .when(AB::Expr::one() - local.is_real)
            .assert_zero(local.memory_operations.op_a_zero);
        AddiRegisterOp::<AB::F>::eval(
            builder,
            shard,
            clk,
            local.cpu_state.pc.into(),
            AB::F::from_canonical_u8(Opcode::ADD as u8),
            local.memory_operations,
            local.is_real.into(),
        );
    }
}
