use dt_core_executor::{
    events::MemoryAccessPosition, ExecutionRecord, Opcode, Program, DEFAULT_PC_INC,
};
use dt_stark::{
    air::{FullAir, FullAirBuilder, MachineAir, PairCol},
    sumcheck::trace::CompressedMatrix,
};
use p3_air::BaseAir;
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::{
    operations::{address_op_gate_constraints, address_op_lookup, address_op_precompute_lc},
    store_word::{StoreWordChip, StoreWordCols, NUM_STORE_WORD_COLUMNS},
};
use crate::{
    adapter::{
        register::b_type::{
            btype_register_op_gate_constraints, btype_register_op_lookup,
            btype_register_op_precompute_lc,
        },
        state::{cpu_state_gate_constraints, cpu_state_lookup, cpu_state_precompute_lc},
    },
    bytes::polyair::{bitvec_lookup, bitvec_precompute_lc},
    memory::polyair::{
        memory_readwrite_lookup, memory_readwrite_precompute_lc, memory_timestamp_gate_constraints,
    },
};

/// 23 lookups: #1-4 CPUState + #5-13 BTypeRegisterOp + #14-18 AddressOp
///           + #19-22 memory_readwrite + #23 BitVec
const NUM_LOOKUPS: usize = 23;
/// BitVec has 16 elements (largest payload).
const MAX_LOOKUP_VALUES: usize = 16;
const PV_EXECUTION_SHARD_IDX: usize = 44;

// ============================================================================
// Main column offsets within `StoreWordCols<u8>` (NUM_STORE_WORD_COLUMNS = 56).
//
// Layout (#[repr(C)]):
//   [0]      cpu_state.shard
//   [1]      cpu_state.clk_16_28
//   [2]      cpu_state.clk_0_16
//   [3]      cpu_state.pc                            ← precompute-only
//   [4]      mem_ops.op_a                            ← precompute-only
//   [5..9]   mem_ops.op_a_access.access.value
//   [9..14]  mem_ops.op_a_access.access.{ts fields}
//   [14]     mem_ops.op_a_zero
//   [15]     mem_ops.op_b                            ← precompute-only
//   [16..20] mem_ops.op_b_access.access.value
//   [20..25] mem_ops.op_b_access.access.{ts fields}
//   [25..29] mem_ops.op_c_imm
//   [29..33] address_operation.addr_word.value
//   [33]     address_operation.addr_range_checker.most_sig_byte_lt_120
//   [34..38] address_operation.offset_bit
//   [38]     address_operation.addr_ls_two_bits
//   [39]     address_operation.aligned_address
//   [40]     address_operation.most_sig_bytes_zero.inverse
//   [41]     address_operation.most_sig_bytes_zero.result
//   [42..46] memory_access.prev_value              ← precompute-only
//   [46..50] memory_access.access.value
//   [50..55] memory_access.access.{ts fields}
//   [55]     is_real
// ============================================================================

const COL_CPU_SHARD: usize = 0;
const COL_CPU_CLK_16_28: usize = 1;
const COL_CPU_CLK_0_16: usize = 2;
const COL_OP_A_VALUE: usize = 5;
const COL_OP_A_ACCESS_TS: usize = 9;
const COL_OP_A_ZERO: usize = 14;
const COL_OP_B_VALUE: usize = 16;
const COL_OP_B_ACCESS_TS: usize = 20;
const COL_OP_C_IMM: usize = 25;
const COL_ADDR_WORD: usize = 29;
const COL_ADDR_RANGE_CHECKER: usize = 33;
const COL_OFFSET_BIT: usize = 34;
const COL_ADDR_LS_TWO_BITS: usize = 38;
const COL_ALIGNED_ADDRESS: usize = 39;
const COL_MSBZ_INVERSE: usize = 40;
const COL_MSBZ_RESULT: usize = 41;
const COL_MEM_VALUE: usize = 46;
const COL_MEM_TS: usize = 50;
const COL_IS_REAL: usize = 55;

// ============================================================================
// Reserved-poly slice layout (RES_NUM_COLS = 49).
// Excludes (precompute-only): cpu_state.pc, op_a, op_b, memory_access.prev_value.
// ============================================================================

const RES_IS_REAL: usize = 0;
const RES_CPU_SHARD: usize = 1;
const RES_CPU_CLK_16_28: usize = 2;
const RES_CPU_CLK_0_16: usize = 3;
const RES_OP_A_ZERO: usize = 4;
const RES_ADDR_RANGE_CHECKER: usize = 5;
const RES_ADDR_LS_TWO_BITS: usize = 6;
const RES_ALIGNED_ADDRESS: usize = 7;
const RES_MSBZ_INVERSE: usize = 8;
const RES_MSBZ_RESULT: usize = 9;
const RES_OP_A_VALUE: usize = 10;
const RES_OP_A_ACCESS_TS: usize = 14;
const RES_OP_B_VALUE: usize = 19;
const RES_OP_B_ACCESS_TS: usize = 23;
const RES_OP_C_IMM: usize = 28;
const RES_ADDR_WORD: usize = 32;
const RES_OFFSET_BIT: usize = 36;
const RES_MEM_VALUE: usize = 40;
const RES_MEM_TS: usize = 44;
const RES_NUM_COLS: usize = 49;

#[derive(Default, Clone, Copy)]
pub struct StoreWordChipPolyAir;

impl<AB: FullAirBuilder> FullAir<AB> for StoreWordChipPolyAir
where
    AB::VarMaybeExt: Clone,
{
    fn width(&self) -> usize {
        NUM_STORE_WORD_COLUMNS
    }

    fn required_max_beta_power(&self) -> usize {
        MAX_LOOKUP_VALUES + 1
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        let mut cols = Vec::with_capacity(RES_NUM_COLS);
        cols.push(PairCol::Main(COL_IS_REAL));
        cols.push(PairCol::Main(COL_CPU_SHARD));
        cols.push(PairCol::Main(COL_CPU_CLK_16_28));
        cols.push(PairCol::Main(COL_CPU_CLK_0_16));
        cols.push(PairCol::Main(COL_OP_A_ZERO));
        cols.push(PairCol::Main(COL_ADDR_RANGE_CHECKER));
        cols.push(PairCol::Main(COL_ADDR_LS_TWO_BITS));
        cols.push(PairCol::Main(COL_ALIGNED_ADDRESS));
        cols.push(PairCol::Main(COL_MSBZ_INVERSE));
        cols.push(PairCol::Main(COL_MSBZ_RESULT));
        for i in 0..4 {
            cols.push(PairCol::Main(COL_OP_A_VALUE + i));
        }
        for i in 0..5 {
            cols.push(PairCol::Main(COL_OP_A_ACCESS_TS + i));
        }
        for i in 0..4 {
            cols.push(PairCol::Main(COL_OP_B_VALUE + i));
        }
        for i in 0..5 {
            cols.push(PairCol::Main(COL_OP_B_ACCESS_TS + i));
        }
        for i in 0..4 {
            cols.push(PairCol::Main(COL_OP_C_IMM + i));
        }
        for i in 0..4 {
            cols.push(PairCol::Main(COL_ADDR_WORD + i));
        }
        for i in 0..4 {
            cols.push(PairCol::Main(COL_OFFSET_BIT + i));
        }
        for i in 0..4 {
            cols.push(PairCol::Main(COL_MEM_VALUE + i));
        }
        for i in 0..5 {
            cols.push(PairCol::Main(COL_MEM_TS + i));
        }
        debug_assert_eq!(cols.len(), RES_NUM_COLS);
        cols
    }

    fn precompute_lc(&self, builder: &mut AB) {
        let main = builder.main();
        let local: &StoreWordCols<AB::VarMaybeExt> = unsafe { core::mem::transmute(main.as_ptr()) };

        let shard = local.cpu_state.shard.clone();
        let clk_0_16 = local.cpu_state.clk_0_16.clone();
        let clk_16_28 = local.cpu_state.clk_16_28.clone();
        let pc = local.cpu_state.pc.clone();
        let clk = clk_0_16.clone() +
            clk_16_28.clone() * AB::VarMaybeExt::from(AB::F::from_canonical_u32(1 << 16));
        let next_pc = pc.clone() + AB::VarMaybeExt::from(AB::F::from_canonical_u32(DEFAULT_PC_INC));

        let opcode = AB::VarMaybeExt::from(AB::F::from_canonical_u8(Opcode::SW as u8));

        let op_c_imm = &local.mem_ops.op_c_imm;

        // =====================================================================
        // #1-4: CPUState (recv_state, send_state, U16Range, BitRange)
        // =====================================================================
        cpu_state_precompute_lc(
            builder,
            shard.clone(),
            clk.clone(),
            clk_0_16,
            clk_16_28,
            pc.clone(),
            next_pc,
        );

        // =====================================================================
        // #5-13: BTypeRegisterOp (1 program + 4 op_a read + 4 op_b read)
        // =====================================================================
        btype_register_op_precompute_lc(
            builder,
            pc,
            opcode,
            local.mem_ops.op_a.clone(),
            local.mem_ops.op_b.clone(),
            [op_c_imm[0].clone(), op_c_imm[1].clone(), op_c_imm[2].clone(), op_c_imm[3].clone()],
            local.mem_ops.op_a_zero.clone(),
            &local.mem_ops.op_a_access.access,
            &local.mem_ops.op_b_access.access,
            shard.clone(),
            clk.clone(),
        );

        // =====================================================================
        // #14-18: AddressOperation (AddOp U8×2 + BabyBear LTU + AND + LTU_most_sig_zero)
        // =====================================================================
        address_op_precompute_lc(
            builder,
            &local.address_operation.addr_word.value,
            local.address_operation.addr_range_checker.most_sig_byte_lt_120.clone(),
            local.address_operation.addr_ls_two_bits.clone(),
        );

        // =====================================================================
        // #19-22: aligned memory readwrite
        // =====================================================================
        memory_readwrite_precompute_lc(
            builder,
            &local.memory_access.access,
            &local.memory_access.prev_value,
            local.address_operation.aligned_address.clone(),
            shard,
            clk.clone() +
                AB::VarMaybeExt::from(AB::F::from_canonical_u8(
                    MemoryAccessPosition::Memory as u8,
                )),
        );

        // =====================================================================
        // #23: BitVec [offset_bit[0..3], most_sig_bytes_zero.result]
        // (is_real removed — enforced by explicit boolean gate in eval)
        // =====================================================================
        bitvec_precompute_lc(
            builder,
            vec![
                local.address_operation.offset_bit[0].clone(),
                local.address_operation.offset_bit[1].clone(),
                local.address_operation.offset_bit[2].clone(),
                local.address_operation.offset_bit[3].clone(),
                local.address_operation.most_sig_bytes_zero.result.clone(),
            ],
        );
    }

    fn eval(&self, builder: &mut AB) {
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        use std::ops::Deref;
        let local: &[AB::VarMaybeExt] = local_binding.deref();

        let is_real = local[RES_IS_REAL].clone();
        let a_word: [AB::VarMaybeExt; 4] =
            core::array::from_fn(|i| local[RES_OP_A_VALUE + i].clone());
        let b_word: [AB::VarMaybeExt; 4] =
            core::array::from_fn(|i| local[RES_OP_B_VALUE + i].clone());
        let c_word: [AB::VarMaybeExt; 4] =
            core::array::from_fn(|i| local[RES_OP_C_IMM + i].clone());
        let addr_word: [AB::VarMaybeExt; 4] =
            core::array::from_fn(|i| local[RES_ADDR_WORD + i].clone());
        let offset: [AB::VarMaybeExt; 4] =
            core::array::from_fn(|i| local[RES_OFFSET_BIT + i].clone());
        let mem_word: [AB::VarMaybeExt; 4] =
            core::array::from_fn(|i| local[RES_MEM_VALUE + i].clone());

        // is_real boolean gate (removed from BitVec payload)
        let one = AB::one_maybe();
        builder.assert_zero(is_real.clone() * (one - is_real.clone()));

        // CPUState::eval
        let pv = builder.public();
        let execution_shard: AB::VarMaybeExt = pv[PV_EXECUTION_SHARD_IDX].clone().into();
        cpu_state_gate_constraints(
            builder,
            local[RES_CPU_SHARD].clone(),
            execution_shard,
            is_real.clone(),
        );

        // BTypeRegisterOp::eval
        btype_register_op_gate_constraints(
            builder,
            local[RES_OP_A_ZERO].clone(),
            a_word.clone(),
            is_real.clone(),
        );

        // AddressOperation::eval
        address_op_gate_constraints(
            builder,
            b_word.clone(),
            c_word,
            addr_word,
            local[RES_ADDR_RANGE_CHECKER].clone(),
            offset.clone(),
            local[RES_ADDR_LS_TWO_BITS].clone(),
            local[RES_ALIGNED_ADDRESS].clone(),
            local[RES_MSBZ_INVERSE].clone(),
            local[RES_MSBZ_RESULT].clone(),
            is_real.clone(),
        );

        // memory timestamp gate constraints
        let clk = local[RES_CPU_CLK_0_16].clone() +
            local[RES_CPU_CLK_16_28].clone() *
                AB::VarMaybeExt::from(AB::F::from_canonical_u32(1 << 16));
        let shard = local[RES_CPU_SHARD].clone();

        let op_a_access = crate::memory::MemoryAccessCols::<AB::VarMaybeExt> {
            value: dt_stark::Word(a_word.clone()),
            prev_shard: local[RES_OP_A_ACCESS_TS].clone(),
            prev_clk: local[RES_OP_A_ACCESS_TS + 1].clone(),
            compare_clk: local[RES_OP_A_ACCESS_TS + 2].clone(),
            diff_16bit_limb: local[RES_OP_A_ACCESS_TS + 3].clone(),
            diff_12bit_limb: local[RES_OP_A_ACCESS_TS + 4].clone(),
        };
        memory_timestamp_gate_constraints(
            builder,
            &op_a_access,
            shard.clone(),
            clk.clone() +
                AB::VarMaybeExt::from(AB::F::from_canonical_u8(MemoryAccessPosition::A as u8)),
            is_real.clone(),
        );
        let op_b_access = crate::memory::MemoryAccessCols::<AB::VarMaybeExt> {
            value: dt_stark::Word(b_word),
            prev_shard: local[RES_OP_B_ACCESS_TS].clone(),
            prev_clk: local[RES_OP_B_ACCESS_TS + 1].clone(),
            compare_clk: local[RES_OP_B_ACCESS_TS + 2].clone(),
            diff_16bit_limb: local[RES_OP_B_ACCESS_TS + 3].clone(),
            diff_12bit_limb: local[RES_OP_B_ACCESS_TS + 4].clone(),
        };
        memory_timestamp_gate_constraints(
            builder,
            &op_b_access,
            shard.clone(),
            clk.clone() +
                AB::VarMaybeExt::from(AB::F::from_canonical_u8(MemoryAccessPosition::B as u8)),
            is_real.clone(),
        );
        let mem_access = crate::memory::MemoryAccessCols::<AB::VarMaybeExt> {
            value: dt_stark::Word(mem_word.clone()),
            prev_shard: local[RES_MEM_TS].clone(),
            prev_clk: local[RES_MEM_TS + 1].clone(),
            compare_clk: local[RES_MEM_TS + 2].clone(),
            diff_16bit_limb: local[RES_MEM_TS + 3].clone(),
            diff_12bit_limb: local[RES_MEM_TS + 4].clone(),
        };
        memory_timestamp_gate_constraints(
            builder,
            &mem_access,
            shard,
            clk + AB::VarMaybeExt::from(AB::F::from_canonical_u8(
                MemoryAccessPosition::Memory as u8,
            )),
            is_real.clone(),
        );

        // store word consistency: offset == 0 and stored_word == a_word
        builder.assert_zero(is_real.clone() - offset[0].clone());
        for i in 0..4 {
            builder.when(is_real.clone()).assert_zero(mem_word[i].clone() - a_word[i].clone());
        }
    }

    fn lookup(&self, builder: &mut AB) {
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        use std::ops::Deref;
        let local: &[AB::VarMaybeExt] = local_binding.deref();

        let is_real = local[RES_IS_REAL].clone();

        // #1-4: CPUState
        cpu_state_lookup(builder, is_real.clone());
        // #5-13: BTypeRegisterOp
        btype_register_op_lookup(builder, is_real.clone());
        // #14-18: AddressOperation
        address_op_lookup(builder, is_real.clone(), local[RES_MSBZ_RESULT].clone());
        // #19-22: aligned memory readwrite
        memory_readwrite_lookup(builder, is_real.clone());
        // #23: BitVec boolean (conditioned on is_real)
        bitvec_lookup(builder, is_real);
    }
}

// =============================================================================
// MachineAir implementation (delegation to StoreWordChip)
// =============================================================================

impl<F: Field> BaseAir<F> for StoreWordChipPolyAir {
    fn width(&self) -> usize {
        NUM_STORE_WORD_COLUMNS
    }
}

impl<F: Field> MachineAir<F> for StoreWordChipPolyAir {
    type Record = ExecutionRecord;
    type Program = Program;

    fn name(&self) -> String {
        "StoreWordPolyAir".to_string()
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        StoreWordChip.generate_trace(input, output)
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        use core::borrow::BorrowMut;
        use dt_core_executor::events::{ByteLookupEvent, ByteRecord};
        use hashbrown::HashMap;
        use itertools::Itertools;
        use p3_maybe_rayon::prelude::{ParallelBridge, ParallelIterator};

        let chunk_size = std::cmp::max(input.store_word_events.len() / num_cpus::get(), 1);
        let shard = input.execution_shard();

        let blu_batches = input
            .store_word_events
            .chunks(chunk_size)
            .par_bridge()
            .map(|events| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                for (record, event) in events {
                    let mut row = [F::zero(); NUM_STORE_WORD_COLUMNS];
                    let cols: &mut StoreWordCols<F> = row.as_mut_slice().borrow_mut();
                    StoreWordChip.event_to_row(record, event, cols, &mut blu, shard);

                    // PolyAir-only: emit BitVec for each real row.
                    // Payload: [offset_bit[0..3], most_sig_bytes_zero.result]
                    let addr = event.b.wrapping_add(event.c);
                    let ls_two_bits = (addr & 0b11) as usize;
                    let addr_bytes = addr.to_le_bytes();
                    let most_sig_bytes_zero =
                        (addr_bytes[1] as u32 + addr_bytes[2] as u32 + addr_bytes[3] as u32) == 0;

                    let value: u16 = ((ls_two_bits == 0) as u16) |
                        ((ls_two_bits == 1) as u16) << 1 |
                        ((ls_two_bits == 2) as u16) << 2 |
                        ((ls_two_bits == 3) as u16) << 3 |
                        (most_sig_bytes_zero as u16) << 4;
                    blu.add_bit_vec_lookup(value);
                }
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_batches.iter().collect_vec());
    }

    fn included(&self, shard: &Self::Record) -> bool {
        <StoreWordChip as MachineAir<F>>::included(&StoreWordChip, shard)
    }
}

#[cfg(test)]
mod tests {
    use super::{super::store_word::StoreWordChip, *};

    const NUM_PRECOMPUTED: usize = NUM_LOOKUPS;
    const BATCH_SIZE: usize = 3;
    use dt_core_executor::{ExecutionRecord, Executor};
    use dt_stark::{
        air::{
            full_air_builders::{
                collect_reserved_poly,
                evaluator::{
                    bound_var_main_prep, bound_var_mat, first_round_evaluation,
                    nonfirst_round_evaluation,
                },
                permutation::generate_permutation_trace_,
                precompute::{precompute_linear_combination, PrecomputeRowBuilder},
            },
            MachineAir,
        },
        DTCoreOpts,
    };
    use p3_baby_bear::BabyBear;
    use p3_field::{extension::BinomialExtensionField, Field, TwoAdicField};
    use p3_matrix::{dense::RowMajorMatrix, Matrix};

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    fn make_public_values(execution_shard: u32) -> Vec<F> {
        let mut pv = vec![F::zero(); PV_EXECUTION_SHARD_IDX + 1];
        pv[PV_EXECUTION_SHARD_IDX] = F::from_canonical_u32(execution_shard);
        pv
    }

    fn ef(x: u32) -> EF {
        EF::from(F::from_canonical_u32(x))
    }

    fn challenge_beta() -> EF {
        EF::two_adic_generator(4) + ef(7)
    }

    fn beta_powers() -> Vec<EF> {
        let beta = challenge_beta();
        let n = <StoreWordChipPolyAir as FullAir<
            PrecomputeRowBuilder<'_, F, F, EF>,
        >>::required_max_beta_power(&StoreWordChipPolyAir);
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
        // Gate constraints: 38 (37 original + 1 is_real boolean gate)
        // Lookup batch: ceil(23/3) = 8
        // Cumulative sum: 3
        const NUM_GATE_CONSTRAINTS: usize = 38;
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
        air: &StoreWordChipPolyAir,
        main: &RowMajorMatrix<F>,
    ) -> RowMajorMatrix<EF> {
        let reserved_poly =
            <StoreWordChipPolyAir as FullAir<PrecomputeRowBuilder<'_, F, F, EF>>>::reserved_poly(
                air,
            );
        let mut values = Vec::new();
        for row_idx in 0..main.height() {
            let row_binding = main.row_slice(row_idx);
            use std::ops::Deref;
            let row: &[F] = row_binding.deref();
            let reserved = collect_reserved_poly(row, &[], &reserved_poly);
            values.extend(reserved.into_iter().map(EF::from));
        }
        RowMajorMatrix::new(values, reserved_poly.len())
    }

    fn sample_trace() -> RowMajorMatrix<F> {
        use crate::programs::tests::keccak_program;
        let program = keccak_program();
        let mut runtime = Executor::new(program, DTCoreOpts::default());
        runtime.run().unwrap();
        let shard = runtime.records[0].clone();

        let chip = StoreWordChip;
        chip.generate_trace(&shard, &mut ExecutionRecord::default()).decompress()
    }

    #[test]
    fn test_store_word_first_and_nonfirst_round_evaluation_satisfied() {
        let air = StoreWordChipPolyAir;
        let main = sample_trace();
        let height = main.height();

        let alpha = ef(123);
        let beta = challenge_beta();
        let beta_powers = beta_powers();
        let beta_septix = beta_septix(beta);
        let public = make_public_values(1);

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

    fn random_store_word_trace(log_n: usize, _seed: u64) -> RowMajorMatrix<F> {
        assert!(log_n < usize::BITS as usize);
        let target_height = 1usize << log_n;
        let base = sample_trace();
        let base_height = base.height();
        assert!(base_height >= 1, "sample trace must contain at least one row");
        assert!(
            target_height >= base_height,
            "target height {} smaller than sample trace height {}",
            target_height,
            base_height
        );
        if target_height == base_height {
            return base;
        }

        let last_row_start = (base_height - 1) * NUM_STORE_WORD_COLUMNS;
        let last_row = &base.values[last_row_start..last_row_start + NUM_STORE_WORD_COLUMNS];
        let mut values = Vec::with_capacity(target_height * NUM_STORE_WORD_COLUMNS);
        values.extend_from_slice(&base.values);
        for _ in base_height..target_height {
            values.extend_from_slice(last_row);
        }
        RowMajorMatrix::new(values, NUM_STORE_WORD_COLUMNS)
    }

    #[test]
    #[ignore = "performance test; run manually"]
    fn perf_multi_round_sumcheck() {
        let air = StoreWordChipPolyAir;
        let log_n = std::env::var("POLYAIR_PERF_LOG_N")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(20);
        let seed = std::env::var("POLYAIR_PERF_SEED")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(42);

        let main = random_store_word_trace(log_n, seed);
        let height = main.height();
        assert_eq!(height, 1 << log_n);
        assert!(height >= 2);
        std::println!("perf_multi_round: log_n={}, h={}, seed={}", log_n, height, seed);

        let alpha = ef(123);
        let beta = challenge_beta();
        let beta_powers = beta_powers();
        let beta_septix = beta_septix(beta);
        let public = make_public_values(1);
        let constraint_reducer = reducer();
        let global = EF::zero();
        let reserved_poly_desc = <StoreWordChipPolyAir as FullAir<
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

    // =========================================================================
    // generate_dependencies tests
    // =========================================================================

    #[test]
    fn bitvec_mult_matches_send_count() {
        use dt_core_executor::ByteOpcode;

        use crate::programs::tests::keccak_program;
        let program = keccak_program();
        let mut runtime = Executor::new(program, DTCoreOpts::default());
        runtime.run().unwrap();
        let shard = runtime.records[0].clone();

        assert!(!shard.store_word_events.is_empty(), "fixture must yield store_word events");

        let mut deps = ExecutionRecord::default();
        <StoreWordChipPolyAir as MachineAir<F>>::generate_dependencies(
            &StoreWordChipPolyAir,
            &shard,
            &mut deps,
        );

        let bitvec_total: usize = deps
            .byte_lookups
            .iter()
            .filter(|(ev, _)| ev.opcode == ByteOpcode::BitVec)
            .map(|(_, m)| *m)
            .sum();

        let expected = shard.store_word_events.len();
        assert!(expected > 0, "test fixture must include store_word events");
        assert_eq!(
            bitvec_total, expected,
            "BitVec BLU emit count must equal number of real events (mult conditioned on is_real)",
        );
    }
}
