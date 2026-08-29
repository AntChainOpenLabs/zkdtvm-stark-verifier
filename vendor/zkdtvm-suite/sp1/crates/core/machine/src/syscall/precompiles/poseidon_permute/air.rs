use core::borrow::Borrow;

use dt_core_executor::syscalls::SyscallCode;
use dt_stark::air::{DTAirBuilder, InteractionScope, SubAirBuilder};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::{
    columns::{Poseidon2MemCols, NUM_POSEIDON2_MEM_COLS},
    poseidon2_inner::{num_cols, Poseidon2Air},
    Poseidon2PermuteChip, STATE_NUM_WORDS,
};

use crate::{
    air::{MemoryAirBuilder, WordAirBuilder},
    memory::MemoryCols,
};

impl<F: Field> BaseAir<F> for Poseidon2PermuteChip<F> {
    fn width(&self) -> usize {
        NUM_POSEIDON2_MEM_COLS
    }
}
impl<AB> Air<AB> for Poseidon2PermuteChip<AB::F>
where
    AB: DTAirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &Poseidon2MemCols<AB::Var> = (*local).borrow();
        let input_state = local.poseidon2_cols.inputs;
        let output_state = *local.poseidon2_cols.output_state();
        // Constrain memory r/w
        for i in 0..STATE_NUM_WORDS as u32 {
            //memory read
            builder.when(local.is_real).assert_word_eq(
                *local.state_mem_read[i as usize].value(),
                *local.state_mem_read[i as usize].prev_value(),
            );
            let read_val: AB::Expr = local.state_mem_read[i as usize].value().reduce::<AB>();
            builder.when(local.is_real).assert_eq(input_state[i as usize], read_val);

            builder.eval_memory_access(
                local.shard,
                local.clk,
                local.state_addr + AB::Expr::from_canonical_u32(i * 4),
                &local.state_mem_read[i as usize],
                local.is_real,
            );

            //memory write
            let write_val: AB::Expr = local.state_mem_write[i as usize].value().reduce::<AB>();
            builder.when(local.is_real).assert_eq(output_state[i as usize], write_val);
            builder.eval_memory_access(
                local.shard,
                local.clk + AB::Expr::one(),
                local.state_addr + AB::Expr::from_canonical_u32(i * 4),
                &local.state_mem_write[i as usize],
                local.is_real,
            );
        }
        for i in 0..STATE_NUM_WORDS {
            builder.slice_range_check_u8(&local.state_mem_read[i].value().0, local.is_real);
            builder.slice_range_check_u8(&local.state_mem_write[i].value().0, local.is_real);
        }
        builder.receive_syscall(
            local.shard,
            local.clk,
            AB::F::from_canonical_u32(SyscallCode::POSEIDON2_PERMUTE.syscall_id()),
            local.state_addr,
            AB::Expr::zero(),
            local.is_real,
            InteractionScope::Local,
        );
        builder.assert_bool(local.is_real);
        let mut sub_builder =
            SubAirBuilder::<AB, Poseidon2Air<AB::F>, AB::F>::new(builder, 0..num_cols());

        self.p3_poseidon2_permute.eval(&mut sub_builder);
    }
}
