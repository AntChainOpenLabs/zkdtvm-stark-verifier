use core::{fmt::Debug, mem::size_of};
use std::borrow::{Borrow, BorrowMut};

use itertools::Itertools;
use p3_field::{AbstractField, PrimeField32};
use serde::{Deserialize, Serialize};

use crate::{Word, PROOF_MAX_NUM_PVS};

/// The number of non padded elements in the zkDTVM proofs public values vec.
pub const DT_PROOF_NUM_PV_ELTS: usize = size_of::<PublicValues<Word<u8>, u8>>();

/// The number of 32 bit words in the zkDTVM proof's committed value digest.
pub const PV_DIGEST_NUM_WORDS: usize = 8;

/// The number of field elements in the poseidon2 digest.
pub const POSEIDON_NUM_WORDS: usize = 8;

/// Canonical projective D11 state encoded as 33 base-field limbs.
#[derive(Serialize, Deserialize, Clone, Copy, Default, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct GlobalState<T> {
    pub x: [T; 11],
    pub y: [T; 11],
    pub z: [T; 11],
}

/// Total Global state transition authenticated by every core shard proof.
#[derive(Serialize, Deserialize, Clone, Copy, Default, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct GlobalStateInterval<T> {
    pub start: GlobalState<T>,
    pub end: GlobalState<T>,
}

/// Fixed-width proof-system-native Global claim.
#[derive(Serialize, Deserialize, Clone, Copy, Default, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct GlobalClaim<T> {
    pub has_global_opening: T,
    pub count: T,
    pub interval: GlobalStateInterval<T>,
}

pub const CORE_PUBLIC_VALUES_PREFIX: usize = 52;
pub const GLOBAL_CLAIM_WIDTH: usize = 68;
pub const GLOBAL_CLAIM_START: usize = CORE_PUBLIC_VALUES_PREFIX;
pub const GLOBAL_CLAIM_END: usize = GLOBAL_CLAIM_START + GLOBAL_CLAIM_WIDTH;

/// Stores all of a shard proof's public values.
#[derive(Serialize, Deserialize, Clone, Copy, Default, Debug)]
#[repr(C)]
pub struct PublicValues<W, T> {
    /// The hash of all the bytes that the guest program has written to public values.
    pub committed_value_digest: [W; PV_DIGEST_NUM_WORDS],

    /// The hash of all deferred proofs that have been witnessed in the VM. It will be rebuilt in
    /// recursive verification as the proofs get verified. The hash itself is a rolling poseidon2
    /// hash of each proof+vkey hash and the previous hash which is initially zero.
    pub deferred_proofs_digest: [T; POSEIDON_NUM_WORDS],

    /// The shard's start program counter.
    pub start_pc: T,

    /// The expected start program counter for the next shard.
    pub next_pc: T,

    /// The exit code of the program.  Only valid if halt has been executed.
    pub exit_code: T,

    /// The shard number.
    pub shard: T,

    /// The execution shard number.
    pub execution_shard: T,

    /// The largest address witnessed for initialization in the previous shard.
    pub previous_init_addr: T,

    /// The largest address witnessed for initialization in the current shard.
    pub last_init_addr: T,

    /// The largest address witnessed for finalization in the previous shard.
    pub previous_finalize_addr: T,

    /// The largest address witnessed for finalization in the current shard.
    pub last_finalize_addr: T,

    /// The clock cycle at the start of this execution shard (first instruction's clk).
    /// For non-execution shards (precompile, memory global), this is 0.
    pub start_clk: T,

    /// The clock cycle sent by the last instruction of this execution shard
    /// (= `start_clk` + 4 * `num_cpu_events`). For non-execution shards, this is 0.
    /// When `start_clk` == `exit_clk`, the shard has no State interactions.
    pub exit_clk: T,

    /// Padding to ensure the size of the public values struct is a multiple of 8.
    pub empty: [T; 1],

    /// Proof-system-native total Global interval claim.
    pub global: GlobalClaim<T>,
}

impl PublicValues<u32, u32> {
    /// Convert the public values into a vector of field elements.  This function will pad the
    /// vector to the maximum number of public values.
    #[must_use]
    pub fn to_vec<F: AbstractField>(&self) -> Vec<F> {
        let mut ret = vec![F::zero(); PROOF_MAX_NUM_PVS];

        let field_values = PublicValues::<Word<F>, F>::from(*self);
        let ret_ref_mut: &mut PublicValues<Word<F>, F> = ret.as_mut_slice().borrow_mut();
        *ret_ref_mut = field_values;
        ret
    }

    /// Resets the public values to zero.
    #[must_use]
    pub fn reset(&self) -> Self {
        let mut copy = *self;
        copy.shard = 0;
        copy.execution_shard = 0;
        copy.start_pc = 0;
        copy.next_pc = 0;
        copy.start_clk = 0;
        copy.exit_clk = 0;
        copy.previous_init_addr = 0;
        copy.last_init_addr = 0;
        copy.previous_finalize_addr = 0;
        copy.last_finalize_addr = 0;
        copy.global = GlobalClaim::default();
        copy
    }
}

impl<F: PrimeField32> PublicValues<Word<F>, F> {
    /// Returns the commit digest as a vector of little-endian bytes.
    pub fn commit_digest_bytes(&self) -> Vec<u8> {
        self.committed_value_digest
            .iter()
            .flat_map(|w| w.into_iter().map(|f| f.as_canonical_u32() as u8))
            .collect_vec()
    }
}

impl<T: Clone> Borrow<PublicValues<Word<T>, T>> for [T] {
    fn borrow(&self) -> &PublicValues<Word<T>, T> {
        let size = std::mem::size_of::<PublicValues<Word<u8>, u8>>();
        debug_assert!(self.len() >= size);
        let slice = &self[0..size];
        let (prefix, shorts, _suffix) = unsafe { slice.align_to::<PublicValues<Word<T>, T>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

impl<T: Clone> BorrowMut<PublicValues<Word<T>, T>> for [T] {
    fn borrow_mut(&mut self) -> &mut PublicValues<Word<T>, T> {
        let size = std::mem::size_of::<PublicValues<Word<u8>, u8>>();
        debug_assert!(self.len() >= size);
        let slice = &mut self[0..size];
        let (prefix, shorts, _suffix) = unsafe { slice.align_to_mut::<PublicValues<Word<T>, T>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &mut shorts[0]
    }
}

impl<F: AbstractField> From<PublicValues<u32, u32>> for PublicValues<Word<F>, F> {
    fn from(value: PublicValues<u32, u32>) -> Self {
        let PublicValues {
            committed_value_digest,
            deferred_proofs_digest,
            start_pc,
            next_pc,
            exit_code,
            shard,
            execution_shard,
            previous_init_addr,
            last_init_addr,
            previous_finalize_addr,
            last_finalize_addr,
            start_clk,
            exit_clk,
            global,
            ..
        } = value;

        let committed_value_digest: [_; PV_DIGEST_NUM_WORDS] =
            core::array::from_fn(|i| Word::from(committed_value_digest[i]));

        let deferred_proofs_digest: [_; POSEIDON_NUM_WORDS] =
            core::array::from_fn(|i| F::from_canonical_u32(deferred_proofs_digest[i]));

        let start_pc = F::from_canonical_u32(start_pc);
        let next_pc = F::from_canonical_u32(next_pc);
        let exit_code = F::from_canonical_u32(exit_code);
        let shard = F::from_canonical_u32(shard);
        let execution_shard = F::from_canonical_u32(execution_shard);
        let previous_init_addr = F::from_canonical_u32(previous_init_addr);
        let last_init_addr = F::from_canonical_u32(last_init_addr);
        let previous_finalize_addr = F::from_canonical_u32(previous_finalize_addr);
        let last_finalize_addr = F::from_canonical_u32(last_finalize_addr);
        let start_clk = F::from_canonical_u32(start_clk);
        let exit_clk = F::from_canonical_u32(exit_clk);
        let map_state = |state: GlobalState<u32>| GlobalState {
            x: state.x.map(F::from_canonical_u32),
            y: state.y.map(F::from_canonical_u32),
            z: state.z.map(F::from_canonical_u32),
        };
        let global = GlobalClaim {
            has_global_opening: F::from_canonical_u32(global.has_global_opening),
            count: F::from_canonical_u32(global.count),
            interval: GlobalStateInterval {
                start: map_state(global.interval.start),
                end: map_state(global.interval.end),
            },
        };

        Self {
            committed_value_digest,
            deferred_proofs_digest,
            start_pc,
            next_pc,
            exit_code,
            shard,
            execution_shard,
            previous_init_addr,
            last_init_addr,
            previous_finalize_addr,
            last_finalize_addr,
            start_clk,
            exit_clk,
            empty: [F::zero()],
            global,
        }
    }
}

const _: () = {
    assert!(GLOBAL_CLAIM_START == 52);
    assert!(GLOBAL_CLAIM_END == 120);
    assert!(DT_PROOF_NUM_PV_ELTS == 120);
};

#[cfg(test)]
mod tests {
    use core::mem::{offset_of, size_of};

    use crate::{
        air::{
            public_values, GlobalClaim, GlobalState, GlobalStateInterval, PublicValues,
            CORE_PUBLIC_VALUES_PREFIX, GLOBAL_CLAIM_END,
        },
        Word,
    };

    /// Check that the [`PI_DIGEST_NUM_WORDS`] number match the zkVM crate's.
    #[test]
    fn test_public_values_digest_num_words_consistency_zkvm() {
        assert_eq!(public_values::PV_DIGEST_NUM_WORDS, dt_zkvm::PV_DIGEST_NUM_WORDS);
    }

    #[test]
    fn global_claim_has_exact_public_value_offsets() {
        type BytePublicValues = PublicValues<Word<u8>, u8>;

        let global = offset_of!(BytePublicValues, global);
        let interval = global + offset_of!(GlobalClaim<u8>, interval);
        assert_eq!(size_of::<BytePublicValues>(), 120);
        assert_eq!(global, CORE_PUBLIC_VALUES_PREFIX);
        assert_eq!(global + offset_of!(GlobalClaim<u8>, has_global_opening), 52);
        assert_eq!(global + offset_of!(GlobalClaim<u8>, count), 53);
        assert_eq!(
            interval + offset_of!(GlobalStateInterval<u8>, start) + offset_of!(GlobalState<u8>, x),
            54
        );
        assert_eq!(
            interval + offset_of!(GlobalStateInterval<u8>, end) + offset_of!(GlobalState<u8>, x),
            87
        );
        assert_eq!(GLOBAL_CLAIM_END, 120);
    }
}
