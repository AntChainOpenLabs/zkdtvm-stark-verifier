//! PolyAir adaptation of `SyscallInstrsChip`.

use std::{borrow::Borrow, ops::Deref};

use dt_core_executor::{
    events::{ByteRecord, MemoryAccessPosition, SyscallEvent},
    syscalls::SyscallCode,
    ByteOpcode, ExecutionRecord, Opcode, Program, RTypeRecord,
    Register::X5,
    DEFAULT_PC_INC,
};
use dt_stark::{
    air::{
        FullAir, FullAirBuilder, MachineAir, PairCol, PublicValues, DT_PROOF_NUM_PV_ELTS,
        PV_DIGEST_NUM_WORDS,
    },
    sumcheck::trace::CompressedMatrix,
    InteractionKind, Word,
};
use p3_air::BaseAir;
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use crate::{
    adapter::{
        register::r_type::rtype_register_op_gate_constraints,
        state::{cpu_state_gate_constraints, cpu_state_lookup},
    },
    bytes::polyair::{bitvec_lookup, bitvec_precompute_lc},
    memory::polyair::{
        memory_read_lookup, memory_read_precompute_lc, memory_readwrite_lookup,
        memory_readwrite_precompute_lc, memory_timestamp_gate_constraints,
    },
    operations::{
        baby_bear_range_check_gate_constraints, baby_bear_range_check_lookup,
        baby_bear_range_check_precompute_lc, is_zero_op_gate_constraints,
    },
    program::program_polyair::{program_lookup, program_precompute_lc},
    syscall::instructions::{columns::SyscallInstrColumns, NUM_SYSCALL_INSTR_COLS},
};

const NUM_LOOKUPS: usize = 20;
const MAX_LOOKUP_VALUES: usize = 16;

#[derive(Default, Clone, Copy)]
pub struct SyscallInstrsChipPolyAir;

impl SyscallInstrsChipPolyAir {
    pub const fn new() -> Self {
        Self
    }
}

fn reduce_word<AB: FullAirBuilder>(word: Word<AB::VarMaybeExt>) -> AB::VarMaybeExt {
    let byte_base = AB::VarMaybeExt::from(AB::F::from_canonical_u32(1 << 8));
    let limb_base = AB::VarMaybeExt::from(AB::F::from_canonical_u32(1 << 16));
    let low = word[0].clone() + word[1].clone() * byte_base.clone();
    let high = word[2].clone() + word[3].clone() * byte_base;
    low + high * limb_base
}

fn index_array<AB: FullAirBuilder>(
    array: &[AB::VarMaybeExt],
    bitmap: &[AB::VarMaybeExt],
) -> AB::VarMaybeExt {
    array
        .iter()
        .zip(bitmap.iter())
        .fold(AB::zero_maybe(), |acc, (value, bit)| acc + value.clone() * bit.clone())
}

fn index_word_array<AB: FullAirBuilder>(
    array: &[Word<AB::VarMaybeExt>],
    bitmap: &[AB::VarMaybeExt],
) -> Word<AB::VarMaybeExt> {
    Word(core::array::from_fn(|i| {
        array
            .iter()
            .zip(bitmap.iter())
            .fold(AB::zero_maybe(), |acc, (word, bit)| acc + word[i].clone() * bit.clone())
    }))
}

impl<AB: FullAirBuilder> FullAir<AB> for SyscallInstrsChipPolyAir
where
    AB::VarMaybeExt: Clone,
{
    fn width(&self) -> usize {
        NUM_SYSCALL_INSTR_COLS
    }

    fn required_max_beta_power(&self) -> usize {
        MAX_LOOKUP_VALUES + 1
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        (0..NUM_SYSCALL_INSTR_COLS).map(PairCol::Main).collect()
    }

    fn precompute_lc(&self, builder: &mut AB) {
        let main = builder.main();
        let local: &SyscallInstrColumns<AB::VarMaybeExt> =
            unsafe { core::mem::transmute(main.as_ptr()) };

        let shard = local.cpu_state.shard.clone();
        let clk = local.cpu_state.clk_0_16.clone() +
            local.cpu_state.clk_16_28.clone() *
                AB::VarMaybeExt::from(AB::F::from_canonical_u32(1 << 16));
        let opcode = AB::VarMaybeExt::from(Opcode::ECALL.as_field::<AB::F>());
        let zero = AB::zero_maybe();
        let state_kind =
            AB::VarMaybeExt::from(AB::F::from_canonical_usize(InteractionKind::State as usize));
        let byte_kind =
            AB::VarMaybeExt::from(AB::F::from_canonical_usize(InteractionKind::Byte as usize));
        let syscall_kind =
            AB::VarMaybeExt::from(AB::F::from_canonical_usize(InteractionKind::Syscall as usize));
        let u16_opcode =
            AB::VarMaybeExt::from(AB::F::from_canonical_u8(ByteOpcode::U16Range as u8));
        let bit_opcode =
            AB::VarMaybeExt::from(AB::F::from_canonical_u8(ByteOpcode::BitRange as u8));
        let twelve = AB::VarMaybeExt::from(AB::F::from_canonical_u32(12));

        let next_clk = clk.clone() +
            AB::VarMaybeExt::from(AB::F::from_canonical_u32(DEFAULT_PC_INC)) +
            local.num_extra_cycles.clone();

        // #1-4: CPUState (custom send_state increment for syscall extra cycles)
        builder.retain_precomputed(builder.lookup_denominator(
            state_kind.clone(),
            vec![shard.clone(), clk.clone(), local.cpu_state.pc.clone()],
        ));
        builder.retain_precomputed(
            builder.lookup_denominator(
                state_kind,
                vec![shard.clone(), next_clk, local.next_pc.clone()],
            ),
        );
        builder.retain_precomputed(builder.lookup_denominator(
            byte_kind.clone(),
            vec![
                u16_opcode,
                local.cpu_state.clk_0_16.clone(),
                zero.clone(),
                zero.clone(),
                zero.clone(),
            ],
        ));
        builder.retain_precomputed(builder.lookup_denominator(
            byte_kind,
            vec![bit_opcode, local.cpu_state.clk_16_28.clone(), zero.clone(), twelve, zero.clone()],
        ));

        // #5: program lookup for ECALL
        program_precompute_lc(
            builder,
            local.cpu_state.pc.clone(),
            opcode,
            local.mem_ops.op_a.clone(),
            [local.mem_ops.op_b.clone(), zero.clone(), zero.clone(), zero.clone()],
            [local.mem_ops.op_c.clone(), zero.clone(), zero.clone(), zero.clone()],
            local.mem_ops.op_a_zero.clone(),
            zero.clone(),
            zero,
        );

        // #6-9: op_b memory read
        memory_read_precompute_lc(
            builder,
            &local.mem_ops.op_b_access.access,
            local.mem_ops.op_b.clone(),
            shard.clone(),
            clk.clone() +
                AB::VarMaybeExt::from(AB::F::from_canonical_u8(MemoryAccessPosition::B as u8)),
        );

        // #10-13: op_c memory read
        memory_read_precompute_lc(
            builder,
            &local.mem_ops.op_c_access.access,
            local.mem_ops.op_c.clone(),
            shard.clone(),
            clk.clone() +
                AB::VarMaybeExt::from(AB::F::from_canonical_u8(MemoryAccessPosition::C as u8)),
        );

        // #14-17: op_a memory read-write at x5
        memory_readwrite_precompute_lc(
            builder,
            &local.mem_ops.op_a_access.access,
            &local.mem_ops.op_a_access.prev_value,
            AB::VarMaybeExt::from(AB::F::from_canonical_u32(X5 as u32)),
            shard,
            clk.clone() +
                AB::VarMaybeExt::from(AB::F::from_canonical_u8(MemoryAccessPosition::A as u8)),
        );

        let syscall_code = local.mem_ops.op_a_access.prev_value.clone();
        let syscall_id = syscall_code[0].clone();
        // #18: send_syscall
        builder.retain_precomputed(builder.lookup_denominator(
            syscall_kind,
            vec![
                local.cpu_state.shard.clone(),
                clk,
                syscall_id,
                reduce_word::<AB>(local.mem_ops.op_b_access.access.value.clone()),
                reduce_word::<AB>(local.mem_ops.op_c_access.access.value.clone()),
            ],
        ));

        // #19: BabyBear range check for halt / commit_deferred operand
        baby_bear_range_check_precompute_lc(
            builder,
            local.operand_to_check[3].clone(),
            local.operand_range_check_cols.most_sig_byte_lt_120.clone(),
        );

        // #20: real-row booleans
        let mut conditional_bits = vec![
            local.is_enter_unconstrained.result.clone(),
            local.is_hint_len.result.clone(),
            local.is_halt_check.result.clone(),
            local.is_commit.result.clone(),
            local.is_commit_deferred_proofs.result.clone(),
        ];
        conditional_bits.extend(local.index_bitmap.clone());
        bitvec_precompute_lc(builder, conditional_bits);
    }

    fn eval(&self, builder: &mut AB) {
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        let local: &SyscallInstrColumns<AB::VarMaybeExt> = unsafe {
            &*(local_binding.deref().as_ptr() as *const SyscallInstrColumns<AB::VarMaybeExt>)
        };

        let one = AB::one_maybe();
        let zero_word =
            Word([AB::zero_maybe(), AB::zero_maybe(), AB::zero_maybe(), AB::zero_maybe()]);

        let pv = builder.public();
        let public_values_slice: [AB::VarMaybeExt; DT_PROOF_NUM_PV_ELTS] =
            core::array::from_fn(|i| pv[i].clone().into());
        let public_values: &PublicValues<Word<AB::VarMaybeExt>, AB::VarMaybeExt> =
            public_values_slice.as_slice().borrow();

        let shard = local.cpu_state.shard.clone();
        let clk = local.cpu_state.clk_0_16.clone() +
            local.cpu_state.clk_16_28.clone() *
                AB::VarMaybeExt::from(AB::F::from_canonical_u32(1 << 16));
        let is_real = local.is_real.clone();

        // is_real boolean: BitVec #20 uses mult=is_real so it only enforces payload bits
        // boolean on real rows. An explicit gate is required so is_real itself is
        // constrained to {0,1} on padding rows.
        builder.assert_zero(is_real.clone() * (one.clone() - is_real.clone()));

        // CPU state shard constraint.
        cpu_state_gate_constraints(
            builder,
            shard.clone(),
            public_values.execution_shard.clone(),
            is_real.clone(),
        );

        // Memory timestamp gate constraints for op_b/op_c/op_a(X5).
        memory_timestamp_gate_constraints(
            builder,
            &local.mem_ops.op_b_access.access,
            shard.clone(),
            clk.clone() +
                AB::VarMaybeExt::from(AB::F::from_canonical_u8(MemoryAccessPosition::B as u8)),
            is_real.clone(),
        );
        memory_timestamp_gate_constraints(
            builder,
            &local.mem_ops.op_c_access.access,
            shard.clone(),
            clk.clone() +
                AB::VarMaybeExt::from(AB::F::from_canonical_u8(MemoryAccessPosition::C as u8)),
            is_real.clone(),
        );
        memory_timestamp_gate_constraints(
            builder,
            &local.mem_ops.op_a_access.access,
            shard,
            clk + AB::VarMaybeExt::from(AB::F::from_canonical_u8(MemoryAccessPosition::A as u8)),
            is_real.clone(),
        );

        // RType gate constraints: x0 linkage.
        rtype_register_op_gate_constraints(
            builder,
            local.mem_ops.op_a_zero.clone(),
            local.mem_ops.op_a_access.access.value.0.clone(),
            is_real.clone(),
        );

        let syscall_code = local.mem_ops.op_a_access.prev_value.clone();
        let syscall_id = syscall_code[0].clone();
        let send_to_table = syscall_code[1].clone();
        let num_extra_cycles = syscall_code[2].clone();

        // Padding rows must not activate syscall interaction.
        builder.when(one.clone() - is_real.clone()).assert_zero(send_to_table);

        // is_halt / commit-related zero checks.
        is_zero_op_gate_constraints(
            builder,
            syscall_id.clone() -
                AB::VarMaybeExt::from(AB::F::from_canonical_u32(SyscallCode::HALT.syscall_id())),
            local.is_halt_check.inverse.clone(),
            local.is_halt_check.result.clone(),
            is_real.clone(),
        );
        is_zero_op_gate_constraints(
            builder,
            syscall_id.clone() -
                AB::VarMaybeExt::from(AB::F::from_canonical_u32(
                    SyscallCode::ENTER_UNCONSTRAINED.syscall_id(),
                )),
            local.is_enter_unconstrained.inverse.clone(),
            local.is_enter_unconstrained.result.clone(),
            is_real.clone(),
        );
        is_zero_op_gate_constraints(
            builder,
            syscall_id.clone() -
                AB::VarMaybeExt::from(AB::F::from_canonical_u32(
                    SyscallCode::HINT_LEN.syscall_id(),
                )),
            local.is_hint_len.inverse.clone(),
            local.is_hint_len.result.clone(),
            is_real.clone(),
        );
        is_zero_op_gate_constraints(
            builder,
            syscall_id.clone() -
                AB::VarMaybeExt::from(AB::F::from_canonical_u32(
                    SyscallCode::COMMIT.syscall_id(),
                )),
            local.is_commit.inverse.clone(),
            local.is_commit.result.clone(),
            is_real.clone(),
        );
        is_zero_op_gate_constraints(
            builder,
            syscall_id -
                AB::VarMaybeExt::from(AB::F::from_canonical_u32(
                    SyscallCode::COMMIT_DEFERRED_PROOFS.syscall_id(),
                )),
            local.is_commit_deferred_proofs.inverse.clone(),
            local.is_commit_deferred_proofs.result.clone(),
            is_real.clone(),
        );

        // is_halt linkage and next_pc/extra_cycles.
        builder.assert_zero(
            local.is_halt.clone() - local.is_halt_check.result.clone() * is_real.clone(),
        );
        builder.when(is_real.clone()).when(one.clone() - local.is_halt.clone()).assert_zero(
            local.next_pc.clone() -
                local.cpu_state.pc.clone() -
                AB::VarMaybeExt::from(AB::F::from_canonical_u32(4)),
        );
        builder.assert_zero(local.num_extra_cycles.clone() - num_extra_cycles * is_real.clone());

        // ENTER_UNCONSTRAINED / HINT_LEN semantics for op_a.
        for i in 0..4 {
            builder
                .when(is_real.clone())
                .when(local.is_enter_unconstrained.result.clone())
                .assert_zero(
                    local.mem_ops.op_a_access.access.value[i].clone() - zero_word[i].clone(),
                );
        }
        for i in 0..4 {
            builder
                .when(is_real.clone())
                .when(
                    one.clone() -
                        (local.is_enter_unconstrained.result.clone() +
                            local.is_hint_len.result.clone()),
                )
                .assert_zero(
                    local.mem_ops.op_a_access.access.value[i].clone() -
                        local.mem_ops.op_a_access.prev_value[i].clone(),
                );
        }

        // Range-check multiplicity linkage for halt / commit_deferred.
        builder.assert_zero(
            local.ecall_range_check_operand.clone() -
                is_real.clone() *
                    (local.is_halt_check.result.clone() +
                        local.is_commit_deferred_proofs.result.clone()),
        );

        baby_bear_range_check_gate_constraints(
            builder,
            local.operand_to_check.0.clone(),
            local.operand_range_check_cols.most_sig_byte_lt_120.clone(),
            local.ecall_range_check_operand.clone(),
        );

        // Commit-related constraints.
        let is_commit_related =
            local.is_commit.result.clone() + local.is_commit_deferred_proofs.result.clone();
        let mut bitmap_sum = AB::zero_maybe();
        for (i, bit) in local.index_bitmap.iter().enumerate() {
            bitmap_sum = bitmap_sum + bit.clone();
            builder.when(is_real.clone()).when(bit.clone()).assert_zero(
                local.mem_ops.op_b_access.access.value[0].clone() -
                    AB::VarMaybeExt::from(AB::F::from_canonical_u32(i as u32)),
            );
        }
        builder
            .when(is_real.clone())
            .when(is_commit_related.clone())
            .assert_one(bitmap_sum.clone());
        builder.when(is_real.clone()).when(one - is_commit_related.clone()).assert_zero(bitmap_sum);
        for i in 0..3 {
            builder
                .when(is_real.clone())
                .when(is_commit_related.clone())
                .assert_zero(local.mem_ops.op_b_access.access.value[i + 1].clone());
        }

        let expected_commit_digest_word =
            index_word_array::<AB>(&public_values.committed_value_digest, &local.index_bitmap);
        let digest_word = local.mem_ops.op_c_access.access.value.clone();
        for i in 0..4 {
            builder
                .when(is_real.clone())
                .when(local.is_commit.result.clone())
                .assert_zero(expected_commit_digest_word[i].clone() - digest_word[i].clone());
        }

        let expected_deferred_digest =
            index_array::<AB>(&public_values.deferred_proofs_digest, &local.index_bitmap);
        for i in 0..4 {
            builder
                .when(is_real.clone())
                .when(local.is_commit_deferred_proofs.result.clone())
                .assert_zero(digest_word[i].clone() - local.operand_to_check[i].clone());
        }
        builder
            .when(is_real)
            .when(local.is_commit_deferred_proofs.result.clone())
            .assert_zero(expected_deferred_digest - reduce_word::<AB>(digest_word));

        // Halt / unimpl constraints.
        builder.when(local.is_halt.clone()).assert_zero(local.next_pc.clone());
        for i in 0..4 {
            builder.when(local.is_halt.clone()).assert_zero(
                local.mem_ops.op_b_access.access.value[i].clone() -
                    local.operand_to_check[i].clone(),
            );
        }
        builder.when(local.is_halt.clone()).assert_zero(
            reduce_word::<AB>(local.mem_ops.op_b_access.access.value.clone()) -
                public_values.exit_code.clone(),
        );
    }

    fn lookup(&self, builder: &mut AB) {
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        let local: &SyscallInstrColumns<AB::VarMaybeExt> = unsafe {
            &*(local_binding.deref().as_ptr() as *const SyscallInstrColumns<AB::VarMaybeExt>)
        };

        let is_real = local.is_real.clone();
        let send_to_table = local.mem_ops.op_a_access.prev_value[1].clone();

        // #1-4: CPUState
        cpu_state_lookup(builder, is_real.clone());

        // #5: program
        program_lookup(builder, is_real.clone());

        // #6-9: op_b memory read
        memory_read_lookup(builder, is_real.clone());

        // #10-13: op_c memory read
        memory_read_lookup(builder, is_real.clone());

        // #14-17: op_a memory read-write
        memory_readwrite_lookup(builder, is_real.clone());

        // #18: send_syscall
        builder.send(send_to_table);

        // #19: BabyBear range check
        baby_bear_range_check_lookup(builder, local.ecall_range_check_operand.clone());

        // #20: real-row booleans (is_real boolean enforced by explicit gate in eval)
        bitvec_lookup(builder, is_real);
    }
}

// =============================================================================
// MachineAir implementation (delegation to SyscallInstrsChip)
// =============================================================================
// BitVec #20 payload helper
// =============================================================================

/// Compute the BitVec #20 payload for one real event row.
///
/// Bit ordering matches `bitvec_precompute_lc` (conditional_bits):
///   bit 0: is_enter_unconstrained.result
///   bit 1: is_hint_len.result
///   bit 2: is_halt_check.result
///   bit 3: is_commit.result
///   bit 4: is_commit_deferred_proofs.result
///   bits 5-12: index_bitmap[0..7]  (PV_DIGEST_NUM_WORDS = 8)
fn syscall_bitvec20_value(record: &RTypeRecord) -> u16 {
    let sid = record.a.previous_record().value.to_le_bytes()[0] as u32;
    let is_enter = (sid == SyscallCode::ENTER_UNCONSTRAINED.syscall_id()) as u16;
    let is_hint = (sid == SyscallCode::HINT_LEN.syscall_id()) as u16;
    let is_halt = (sid == SyscallCode::HALT.syscall_id()) as u16;
    let is_commit = (sid == SyscallCode::COMMIT.syscall_id()) as u16;
    let is_commit_deferred = (sid == SyscallCode::COMMIT_DEFERRED_PROOFS.syscall_id()) as u16;

    // index_bitmap[digest_idx] = 1 iff is_commit || is_commit_deferred.
    // Shift into bits 5..12.
    let index_bits: u16 = if is_commit == 1 || is_commit_deferred == 1 {
        let digest_idx = record.op_b_value() as usize;
        if digest_idx < PV_DIGEST_NUM_WORDS {
            1u16 << digest_idx
        } else {
            0
        }
    } else {
        0
    };

    is_enter |
        (is_hint << 1) |
        (is_halt << 2) |
        (is_commit << 3) |
        (is_commit_deferred << 4) |
        (index_bits << 5)
}

use crate::syscall::instructions::SyscallInstrsChip;

impl<F: Field> BaseAir<F> for SyscallInstrsChipPolyAir {
    fn width(&self) -> usize {
        NUM_SYSCALL_INSTR_COLS
    }
}

impl<F: Field> MachineAir<F> for SyscallInstrsChipPolyAir {
    type Record = ExecutionRecord;
    type Program = Program;

    fn name(&self) -> String {
        <SyscallInstrsChip as MachineAir<F>>::name(&SyscallInstrsChip) + "PolyAir"
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        use core::borrow::BorrowMut;
        use dt_core_executor::events::{ByteLookupEvent, ByteRecord};
        use hashbrown::HashMap;
        use itertools::Itertools;
        use p3_maybe_rayon::prelude::{ParallelBridge, ParallelIterator};

        let events = &input.syscall_events;
        if events.is_empty() {
            return;
        }

        let chunk_size = std::cmp::max(events.len() / num_cpus::get(), 1);

        let blu_batches = events
            .chunks(chunk_size)
            .par_bridge()
            .map(|chunk| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                for (record, event) in chunk {
                    let mut row = [F::zero(); NUM_SYSCALL_INSTR_COLS];
                    let cols: &mut SyscallInstrColumns<F> = row.as_mut_slice().borrow_mut();
                    SyscallInstrsChip.event_to_row(record, event, cols, &mut blu);
                    blu.add_bit_vec_lookup(syscall_bitvec20_value(record));
                }
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_batches.iter().collect_vec());
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        SyscallInstrsChip.generate_trace(input, output)
    }

    fn included(&self, shard: &Self::Record) -> bool {
        <SyscallInstrsChip as MachineAir<F>>::included(&SyscallInstrsChip, shard)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{programs::tests::keccak_program, syscall::instructions::SyscallInstrsChip};
    use dt_core_executor::{ExecutionRecord, Executor};
    use dt_stark::{
        air::{
            collect_reserved_poly,
            full_air_builders::{
                evaluator::{
                    bound_var_main_prep, bound_var_mat, first_round_evaluation,
                    nonfirst_round_evaluation,
                },
                permutation::generate_permutation_trace_,
                precompute::{precompute_linear_combination, PrecomputeRowBuilder},
            },
            FullAir, MachineAir,
        },
        DTCoreOpts,
    };
    use p3_baby_bear::BabyBear;
    use p3_field::{extension::BinomialExtensionField, Field, TwoAdicField};
    use p3_matrix::{dense::RowMajorMatrix, Matrix};

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    const BATCH_SIZE: usize = 3;
    const NUM_PRECOMPUTED: usize = NUM_LOOKUPS;

    fn ef(x: u32) -> EF {
        EF::from(F::from_canonical_u32(x))
    }

    fn challenge_beta() -> EF {
        EF::two_adic_generator(4) + ef(7)
    }

    fn beta_powers() -> Vec<EF> {
        let beta = challenge_beta();
        let n = <SyscallInstrsChipPolyAir as FullAir<PrecomputeRowBuilder<'_, F, F, EF>>>::required_max_beta_power(
            &SyscallInstrsChipPolyAir,
        );
        (0..=n).map(|i| beta.exp_u64(i as u64)).collect()
    }

    fn beta_septix(beta: EF) -> EF {
        dt_stark::septic_curve_params::compute_beta_septix::<
            F,
            EF,
            dt_stark::septic_curve_params::BabyBearCurveParams,
        >(beta)
    }

    fn reducer() -> Vec<EF> {
        // Gate constraints: 75 base + 1 is_real boolean gate (was implicit in BitVec #20
        // when mult=one_maybe; now explicit since the unconditional BitVec was removed)
        // Lookup batch: ceil(20/3) = 7 (same ceiling as before)
        // Cumulative sum: 3
        const NUM_GATE_CONSTRAINTS: usize = 76;
        const NUM_REDUCER_CONSTRAINTS: usize =
            NUM_GATE_CONSTRAINTS + NUM_LOOKUPS.div_ceil(BATCH_SIZE) + 3;
        (0..NUM_REDUCER_CONSTRAINTS as u32).map(|i| ef(i + 1)).collect()
    }

    fn trim_rows<T: Clone + Send + Sync>(
        matrix: &RowMajorMatrix<T>,
        num_rows: usize,
    ) -> RowMajorMatrix<T> {
        let width = matrix.width();
        RowMajorMatrix::new(matrix.values[..num_rows * width].to_vec(), width)
    }

    fn reserved_poly_matrix(
        air: &SyscallInstrsChipPolyAir,
        main: &RowMajorMatrix<F>,
    ) -> RowMajorMatrix<EF> {
        let reserved_poly = <SyscallInstrsChipPolyAir as FullAir<
            PrecomputeRowBuilder<'_, F, F, EF>,
        >>::reserved_poly(air);
        let mut values = Vec::new();
        for row_idx in 0..main.height() {
            let row_binding = main.row_slice(row_idx);
            let row: &[F] = row_binding.deref();
            let reserved = collect_reserved_poly(row, &[], &reserved_poly);
            values.extend(reserved.into_iter().map(EF::from));
        }
        RowMajorMatrix::new(values, reserved_poly.len())
    }

    fn sample_trace_and_public() -> Option<(RowMajorMatrix<F>, Vec<F>)> {
        let program = keccak_program();
        let mut runtime = Executor::new(program, DTCoreOpts::default());
        runtime.run().unwrap();

        for record in &runtime.records {
            let shard = *record.clone();
            let chip = SyscallInstrsChip;
            if <SyscallInstrsChip as MachineAir<F>>::included(&chip, &shard) {
                let public = shard.public_values.to_vec::<F>();
                let trace =
                    chip.generate_trace(&shard, &mut ExecutionRecord::default()).decompress();
                return Some((trace, public));
            }
        }

        None
    }

    #[test]
    fn test_syscall_instrs_constraint_check() {
        let Some((main, public)) = sample_trace_and_public() else {
            eprintln!("No SyscallInstrs trace found — skipping test");
            return;
        };

        let air = SyscallInstrsChipPolyAir::new();
        let height = main.height();
        assert!(height >= 2);

        let alpha = ef(123);
        let beta = challenge_beta();
        let beta_powers = beta_powers();
        let beta_septix = beta_septix(beta);

        let precomputed_full = precompute_linear_combination(
            &air,
            None,
            &main,
            &public,
            alpha,
            &beta_powers,
            beta_septix,
            NUM_PRECOMPUTED,
        );
        let (permutation_full, local_sum) = generate_permutation_trace_(
            &air,
            None,
            &main,
            &precomputed_full,
            alpha,
            &beta_powers,
            BATCH_SIZE,
            NUM_LOOKUPS,
        );

        let precomputed = trim_rows(&precomputed_full, height);
        let permutation = trim_rows(&permutation_full, height);
        let reserved = reserved_poly_matrix(&air, &main);
        let constraint_reducer = reducer();
        let global = EF::zero();

        let first = first_round_evaluation(
            &air,
            &public,
            None,
            &main,
            &precomputed,
            &permutation,
            alpha,
            &beta_powers,
            beta_septix,
            global,
            F::one(),
            F::one(),
            local_sum,
            BATCH_SIZE,
            &constraint_reducer,
        );
        assert!(first.iter().all(|x| x.is_zero()), "first_round_evaluation failed: {:?}", first);

        let nonfirst = nonfirst_round_evaluation(
            &air,
            &public,
            &reserved,
            &precomputed,
            &permutation,
            alpha,
            &beta_powers,
            beta_septix,
            global,
            EF::one(),
            EF::one(),
            local_sum,
            BATCH_SIZE,
            &constraint_reducer,
        );
        assert!(
            nonfirst.iter().all(|x| x.is_zero()),
            "nonfirst_round_evaluation failed: {:?}",
            nonfirst
        );
    }

    fn random_syscall_instrs_trace(log_n: usize, _seed: u64) -> (RowMajorMatrix<F>, Vec<F>) {
        assert!(log_n < usize::BITS as usize);
        let target_height = 1usize << log_n;
        let (base, public) = sample_trace_and_public().expect("sample trace should exist");
        let base_height = base.height();
        assert!(base_height >= 1, "sample trace must contain at least one row");
        assert!(
            target_height >= base_height,
            "target height {} smaller than sample trace height {}",
            target_height,
            base_height
        );
        if target_height == base_height {
            return (base, public);
        }
        let width = base.width();
        let last_row_start = (base_height - 1) * width;
        let last_row = &base.values[last_row_start..last_row_start + width];
        let mut values = Vec::with_capacity(target_height * width);
        values.extend_from_slice(&base.values);
        for _ in base_height..target_height {
            values.extend_from_slice(last_row);
        }
        (RowMajorMatrix::new(values, width), public)
    }

    #[test]
    #[ignore = "performance test; run manually"]
    fn perf_multi_round_sumcheck() {
        let air = SyscallInstrsChipPolyAir::new();
        let log_n = std::env::var("POLYAIR_PERF_LOG_N")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(20);
        let seed = std::env::var("POLYAIR_PERF_SEED")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(42);

        let (main, public) = random_syscall_instrs_trace(log_n, seed);
        let height = main.height();
        assert!(height >= 2);
        std::println!("perf_multi_round: log_n={}, h={}, seed={}", log_n, height, seed);

        let alpha = ef(123);
        let beta = challenge_beta();
        let beta_powers = beta_powers();
        let beta_septix = beta_septix(beta);
        let constraint_reducer = reducer();
        let global = EF::zero();
        let reserved_poly_desc = <SyscallInstrsChipPolyAir as FullAir<
            PrecomputeRowBuilder<'_, F, F, EF>,
        >>::reserved_poly(&air);

        // --- Precompute phase ---
        let t_precompute = std::time::Instant::now();
        let precomputed_full = precompute_linear_combination(
            &air,
            None,
            &main,
            &public,
            alpha,
            &beta_powers,
            beta_septix,
            NUM_PRECOMPUTED,
        );
        let precompute_elapsed = t_precompute.elapsed();
        std::println!("  precompute_linear_combination: {:?}", precompute_elapsed);

        let t_perm = std::time::Instant::now();
        let (permutation_full, local_sum) = generate_permutation_trace_(
            &air,
            None,
            &main,
            &precomputed_full,
            alpha,
            &beta_powers,
            BATCH_SIZE,
            NUM_LOOKUPS,
        );
        let perm_elapsed = t_perm.elapsed();
        std::println!("  generate_permutation_trace_: {:?}", perm_elapsed);

        let mut precomputed = trim_rows(&precomputed_full, height);
        let mut permutation = trim_rows(&permutation_full, height);

        // --- Round 0: first_round_evaluation ---
        let t_total = std::time::Instant::now();
        let t_round = std::time::Instant::now();
        let _first = first_round_evaluation(
            &air,
            &public,
            None,
            &main,
            &precomputed,
            &permutation,
            alpha,
            &beta_powers,
            beta_septix,
            global,
            F::one(),
            F::one(),
            local_sum,
            BATCH_SIZE,
            &constraint_reducer,
        );
        std::println!("  round 0 (first_round): {:?}", t_round.elapsed());

        // --- Rounds 1..log_n-1: fold + nonfirst_round_evaluation ---
        let mut reserved = bound_var_main_prep(&main, None, &reserved_poly_desc, ef(42));
        precomputed = bound_var_mat(&precomputed_full, ef(42));
        permutation = bound_var_mat(&permutation_full, ef(42));
        let mut selector_first = EF::one() - ef(42);
        let mut selector_last = ef(42);

        for round in 1..log_n {
            let challenge = ef((round as u32) + 100);
            let t_round = std::time::Instant::now();

            let _nonfirst = nonfirst_round_evaluation(
                &air,
                &public,
                &reserved,
                &precomputed,
                &permutation,
                alpha,
                &beta_powers,
                beta_septix,
                global,
                selector_first,
                selector_last,
                local_sum,
                BATCH_SIZE,
                &constraint_reducer,
            );

            let round_elapsed = t_round.elapsed();
            std::println!("  round {} (nonfirst): {:?}", round, round_elapsed);

            if round < log_n - 1 {
                reserved = bound_var_mat(&reserved, challenge);
                precomputed = bound_var_mat(&precomputed, challenge);
                permutation = bound_var_mat(&permutation, challenge);
                selector_first *= EF::one() - challenge;
                selector_last *= challenge;
            }
        }

        let total_eval_elapsed = t_total.elapsed();
        std::println!("  ---");
        std::println!("  total precompute: {:?}", precompute_elapsed);
        std::println!("  total perm_gen:   {:?}", perm_elapsed);
        std::println!("  total eval ({} rounds): {:?}", log_n, total_eval_elapsed);
        std::println!(
            "  GRAND TOTAL (precompute + perm + eval): {:?}",
            precompute_elapsed + perm_elapsed + total_eval_elapsed
        );
    }
}
