use core::panic;

use crate::{
    events::{Poseidon2PermuteEvent, PrecompileEvent},
    syscalls::{Syscall, SyscallCode, SyscallContext},
};
use dt_primitives::{SCField, SC_POSEIDON2_INSTRUCTION};
use p3_field::{AbstractField, PrimeField32};
use p3_symmetric::Permutation;
pub struct Poseidon2PermuteSyscall;

pub(crate) const STATE_SIZE: usize = 24;
pub(crate) const STATE_NUM_WORDS: usize = STATE_SIZE;

impl Syscall for Poseidon2PermuteSyscall {
    fn num_extra_cycles(&self) -> u32 {
        1
    }

    fn execute(
        &self,
        ctx: &mut SyscallContext,
        syscall_code: SyscallCode,
        arg1: u32,
        arg2: u32,
    ) -> Option<u32> {
        // let perm = runtime_poseidon2_init();

        let start_clk = ctx.clk;
        let state_ptr = arg1;
        if arg2 != 0 {
            panic!("Expected arg2 to be 0, got {arg2}");
        }
        let mut state_read_records = Vec::new();
        let mut state_write_records = Vec::new();

        let (state_records, mut state) = ctx.mr_slice(state_ptr, STATE_NUM_WORDS);
        state_read_records.extend_from_slice(&state_records);
        let saved_state = state.clone();
        //TODO: use mem::transmute to convert state to [SCField;STATE_SIZE]
        let mut state: [SCField; STATE_SIZE] = state
            .iter_mut()
            .map(|val| SCField::from_canonical_u32(*val))
            .collect::<Vec<SCField>>()
            .try_into()
            .unwrap();

        SC_POSEIDON2_INSTRUCTION.permute_mut(&mut state);
        //TODO: use mem::transmute
        let state = state.into_iter().map(|val| val.as_canonical_u32()).collect::<Vec<u32>>();

        //  Increment the clk by 1 before writing because we read from memory at start_clk.
        ctx.clk += 1;
        let write_records = ctx.mw_slice(state_ptr, state.as_slice());
        state_write_records.extend_from_slice(&write_records);
        // finish poseidon2 permutation
        // records poseidon2 events and updates ctx

        let shard = ctx.current_shard();
        let event = PrecompileEvent::Poseidon2Permute(Poseidon2PermuteEvent {
            shard,
            clk: start_clk,
            pre_state: saved_state.as_slice().try_into().unwrap(),
            post_state: state.as_slice().try_into().unwrap(),
            state_read_records,
            state_write_records,
            state_addr: state_ptr,
            local_mem_access: ctx.postprocess(),
        });
        let syscall_event =
            ctx.rt.syscall_event(start_clk, None, None, syscall_code, arg1, arg2, ctx.next_pc);
        ctx.add_precompile_event(syscall_code, syscall_event, event);

        None
    }
}
