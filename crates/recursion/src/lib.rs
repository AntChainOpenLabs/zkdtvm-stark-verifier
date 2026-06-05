#![allow(unused_variables, unused_imports, unused_mut, dead_code)]
#![allow(dropping_references)]
#![allow(unexpected_cfgs)]
#![allow(unused_unsafe)]
#![allow(unused_braces)]

use dt_derive::AlignedBorrow;
use p3_field::PrimeField64;
use serde::{Deserialize, Serialize};

use crate::air::{Block, RecursionPublicValues};

pub mod air;
pub mod builder;
pub mod chips;
pub mod machine;
pub mod runtime;
pub mod shape;
pub mod stark;
pub mod operations;
pub mod utils;

pub use runtime::*;

// Re-export the stark stuff from `dt_recursion_core` for now, until we will migrate it here.
// pub use dt_recursion_core::stark;

use crate::chips::poseidon2_skinny::WIDTH;

#[derive(
    AlignedBorrow, Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default,
)]
#[repr(transparent)]
pub struct Address<F>(pub F);

impl<F: PrimeField64> Address<F> {
    #[inline]
    pub fn as_usize(&self) -> usize {
        self.0.as_canonical_u64() as usize
    }
}

// -------------------------------------------------------------------------------------------------

/// The inputs and outputs to an operation of the base field ALU.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(C)]
pub struct BaseAluIo<V> {
    pub out: V,
    pub in1: V,
    pub in2: V,
}

pub type BaseAluEvent<F> = BaseAluIo<F>;

/// An instruction invoking the extension field ALU.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[repr(C)]
pub struct BaseAluInstr<F> {
    pub opcode: BaseAluOpcode,
    pub mult: F,
    pub addrs: BaseAluIo<Address<F>>,
}

// -------------------------------------------------------------------------------------------------

/// The inputs and outputs to an operation of the extension field ALU.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(C)]
pub struct ExtAluIo<V> {
    pub out: V,
    pub in1: V,
    pub in2: V,
}

pub type ExtAluEvent<F> = ExtAluIo<Block<F>>;

/// An instruction invoking the extension field ALU.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[repr(C)]
pub struct ExtAluInstr<F> {
    pub opcode: ExtAluOpcode,
    pub mult: F,
    pub addrs: ExtAluIo<Address<F>>,
}

// -------------------------------------------------------------------------------------------------

/// The inputs and outputs to the manual memory management/memory initialization table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemIo<V> {
    pub inner: V,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemInstr<F> {
    pub addrs: MemIo<Address<F>>,
    pub vals: MemIo<Block<F>>,
    pub mult: F,
    pub kind: MemAccessKind,
}

pub type MemEvent<F> = MemIo<Block<F>>;

// -------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemAccessKind {
    Read,
    Write,
}

/// The inputs and outputs to a Poseidon2 permutation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(C)]
pub struct Poseidon2Io<V> {
    pub input: [V; WIDTH],
    pub output: [V; WIDTH],
}

/// An instruction invoking the Poseidon2 permutation, used by the wide-layout chip
/// (`Poseidon2WideChip` / `Poseidon2WideKbChip`). This layout fits one full permutation per row,
/// so no per-round scratch addresses are needed.
///
/// Kept `#[repr(C)]` and `Copy` so it can cross the C++ FFI boundary unchanged.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[repr(C)]
pub struct Poseidon2WideInstr<F> {
    pub addrs: Poseidon2Io<Address<F>>,
    pub mults: [F; WIDTH],
}

/// Number of `WIDTH`-tuples of scratch addresses needed to chain intermediate states between
/// consecutive *rows* of the skinny chip's per-permutation block. Equals `ROWS_PER_PERMUTE - 1`.
///
/// The two skinny chips lay out a permutation differently:
///   * BabyBear  : one round per row, 21 rows per permutation -> `SKINNY_NUM_SCRATCH = 20`.
///   * KoalaBear : 5-row layout (2 ext + 1 internal-rounds row + 2 ext) -> `SKINNY_NUM_SCRATCH = 4`.
///
/// cbindgen:ignore
#[cfg(feature = "babybear")]
pub const SKINNY_NUM_SCRATCH: usize = 20;
/// cbindgen:ignore
#[cfg(feature = "koalabear")]
pub const SKINNY_NUM_SCRATCH: usize = 4;

/// An instruction invoking the Poseidon2 permutation, specialised for the skinny chip variants
/// (BabyBear `Poseidon2SkinnyChip` and KoalaBear `Poseidon2SkinnyKbChip`).
///
/// Carries the `scratch_addrs` needed to chain intermediate states across the chip's per-row
/// transitions via memory lookup, since transition constraints are evaluated on a single row
/// only. `scratch_addrs[r][i]` holds the i-th component of the state output of row `r`, which
/// equals the state input of row `r + 1`. The array length is fixed to `SKINNY_NUM_SCRATCH`
/// (= `ROWS_PER_PERMUTE - 1`), differing between BabyBear (one-round-per-row, 20 groups) and
/// KoalaBear (5-row layout, 8 groups) via cargo feature gating.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Poseidon2SkinnyInstr<F> {
    pub addrs: Poseidon2Io<Address<F>>,
    pub mults: [F; WIDTH],
    pub scratch_addrs: [[Address<F>; WIDTH]; SKINNY_NUM_SCRATCH],
}

pub type Poseidon2Event<F> = Poseidon2Io<F>;

/// The inputs and outputs to a select operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(C)]
pub struct SelectIo<V> {
    pub bit: V,
    pub out1: V,
    pub out2: V,
    pub in1: V,
    pub in2: V,
}

/// An instruction invoking the select operation.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[repr(C)]
pub struct SelectInstr<F> {
    pub addrs: SelectIo<Address<F>>,
    pub mult1: F,
    pub mult2: F,
}

/// The event encoding the inputs and outputs of a select operation.
pub type SelectEvent<F> = SelectIo<F>;

pub type Poseidon2WideEvent<F> = Poseidon2Io<F>;
pub type Poseidon2Instr<F> = Poseidon2WideInstr<F>;

/// An instruction that will save the public values to the execution record and will commit to
/// it's digest.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[repr(C)]
pub struct CommitPublicValuesInstr<F> {
    pub pv_addrs: RecursionPublicValues<Address<F>>,
}

/// The event for committing to the public values.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[repr(C)]
pub struct CommitPublicValuesEvent<F> {
    pub public_values: RecursionPublicValues<F>,
}

// -------------------------------------------------------------------------------------------------

/// The inputs and output to the polynomial evaluation operation. Coefficients are represented as
/// a vector from higher degree to lower.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(C)]
pub struct PolyEvalIo<V> {
    pub point: V,
    // assume the coefficient is arrayed from higher degree to lower
    pub coeff: Vec<V>,
    pub out: V,
}

/// An instruction that will eval a univariate polynomial at a point.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PolyEvalInstr<F> {
    pub addrs: PolyEvalIo<Address<F>>,
    pub mult: F,
    /// Chain addresses for linking accum across rows (len = num_coeffs - 1).
    pub chain_accum_addrs: Vec<Address<F>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct PolyEvalInstrFFI<'a, F> {
    pub point: &'a Address<F>,
    pub coeff_ptr: *const Address<F>,
    pub coeff_len: usize,
    pub out: &'a Address<F>,

    pub mult: &'a F,
    pub chain_accum_ptr: *const Address<F>,
    pub chain_accum_len: usize,
}

impl<'a, F> From<&'a PolyEvalInstr<F>> for PolyEvalInstrFFI<'a, F> {
    fn from(instr: &'a PolyEvalInstr<F>) -> Self {
        Self {
            point: &instr.addrs.point,
            coeff_ptr: instr.addrs.coeff.as_ptr(),
            coeff_len: instr.addrs.coeff.len(),
            out: &instr.addrs.out,

            mult: &instr.mult,
            chain_accum_ptr: instr.chain_accum_addrs.as_ptr(),
            chain_accum_len: instr.chain_accum_addrs.len(),
        }
    }
}

pub type PolyEvalEvent<F> = PolyEvalIo<F>;

#[derive(Clone, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct PolyEvalEventFFI<'a, F> {
    pub point: &'a F,
    pub coeff_ptr: *const F,
    pub coeff_len: usize,
    pub out: &'a F,
}

impl<'a, F> From<&'a PolyEvalEvent<F>> for PolyEvalEventFFI<'a, F> {
    fn from(event: &'a PolyEvalEvent<F>) -> Self {
        Self {
            point: &event.point,
            coeff_ptr: event.coeff.as_ptr(),
            coeff_len: event.coeff.len(),
            out: &event.out,
        }
    }
}

// -------------------------------------------------------------------------------------------------

/// The inputs and output to the extension polynomial evaluation operation. Coefficients are
/// represented as a vector from higher degree to lower.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(C)]
pub struct ExtPolyEvalIo<V> {
    pub point: V,
    // assume the coefficient is arrayed from higher degree to lower
    pub coeff: Vec<V>,
    pub out: V,
}

/// An instruction that will eval a univariate polynomial at a point.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[repr(C)]
pub struct ExtPolyEvalInstr<F> {
    pub addrs: ExtPolyEvalIo<Address<F>>,
    pub mult: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct ExtPolyEvalInstrFFI<'a, F> {
    pub point: &'a Address<F>,
    pub coeff_ptr: *const Address<F>,
    pub coeff_len: usize,
    pub out: &'a Address<F>,

    pub mult: &'a F,
}

impl<'a, F> From<&'a ExtPolyEvalInstr<F>> for ExtPolyEvalInstrFFI<'a, F> {
    fn from(instr: &'a ExtPolyEvalInstr<F>) -> Self {
        Self {
            point: &instr.addrs.point,
            coeff_ptr: instr.addrs.coeff.as_ptr(),
            coeff_len: instr.addrs.coeff.len(),
            out: &instr.addrs.out,

            mult: &instr.mult,
        }
    }
}

pub type ExtPolyEvalEvent<F> = ExtPolyEvalIo<Block<F>>;

#[derive(Clone, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct ExtPolyEvalEventFFI<'a, F> {
    pub point: &'a Block<F>,
    pub coeff_ptr: *const Block<F>,
    pub coeff_len: usize,
    pub out: &'a Block<F>,
}

impl<'a, F> From<&'a ExtPolyEvalEvent<F>> for ExtPolyEvalEventFFI<'a, F> {
    fn from(event: &'a ExtPolyEvalEvent<F>) -> Self {
        Self {
            point: &event.point,
            coeff_ptr: event.coeff.as_ptr(),
            coeff_len: event.coeff.len(),
            out: &event.out,
        }
    }
}

// -------------------------------------------------------------------------------------------------

/// The inputs and outputs to an extension field exp-reverse-bits operation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExtExpReverseBitsIo<V> {
    pub base: V,
    // The bits of the exponent in little-endian order in a vec.
    pub exp: Vec<V>,
    /// Previous accumulator values for each row (len = N).
    pub prev_acc_vec: Vec<V>,
    /// Current accumulator values for each row (len = N).
    pub acc_vec: Vec<V>,
}

/// An instruction invoking the exp-reverse-bits operation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExtExpReverseBitsInstr<F> {
    pub addrs: ExtExpReverseBitsIo<Address<F>>,
    pub mult: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct ExtExpReverseBitsInstrFFI<'a, F> {
    pub base: &'a Address<F>,
    pub exp_ptr: *const Address<F>,
    pub exp_len: usize,
    pub prev_acc_ptr: *const Address<F>,
    pub acc_ptr: *const Address<F>,

    pub mult: &'a F,
}

impl<'a, F> From<&'a ExtExpReverseBitsInstr<F>> for ExtExpReverseBitsInstrFFI<'a, F> {
    fn from(instr: &'a ExtExpReverseBitsInstr<F>) -> Self {
        Self {
            base: &instr.addrs.base,
            exp_ptr: instr.addrs.exp.as_ptr(),
            exp_len: instr.addrs.exp.len(),
            prev_acc_ptr: instr.addrs.prev_acc_vec.as_ptr(),
            acc_ptr: instr.addrs.acc_vec.as_ptr(),

            mult: &instr.mult,
        }
    }
}

/// The event encoding the inputs and outputs of an ext exp-reverse-bits operation. The `len` operand is
/// now stored as the length of the `exp` field.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExtExpReverseBitsEvent<F> {
    pub base: Block<F>,
    pub exp: Vec<F>,
    /// Previous accumulator values for each row (len = N).
    pub prev_acc_vec: Vec<Block<F>>,
    /// Current accumulator values for each row (len = N).
    pub acc_vec: Vec<Block<F>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct ExtExpReverseBitsEventFFI<'a, F> {
    pub base: &'a Block<F>,
    pub exp_ptr: *const F,
    pub exp_len: usize,
    pub prev_acc_ptr: *const Block<F>,
    pub acc_ptr: *const Block<F>,
}

impl<'a, F> From<&'a ExtExpReverseBitsEvent<F>> for ExtExpReverseBitsEventFFI<'a, F> {
    fn from(event: &'a ExtExpReverseBitsEvent<F>) -> Self {
        Self {
            base: &event.base,
            exp_ptr: event.exp.as_ptr(),
            exp_len: event.exp.len(),
            prev_acc_ptr: event.prev_acc_vec.as_ptr(),
            acc_ptr: event.acc_vec.as_ptr(),
        }
    }
}

// -------------------------------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrefixSumChecksIo<V> {
    pub x1_vec: Vec<V>,
    pub x2_vec: Vec<V>,
    /// Previous accumulator values for each row (len = N).
    pub prev_acc_vec: Vec<V>,
    /// Current accumulator values for each row (len = N).
    pub acc_vec: Vec<V>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PrefixSumChecksInstr<F> {
    pub addrs: PrefixSumChecksIo<Address<F>>,
    pub mult: F,
}

pub type PrefixSumChecksEvent<F> = PrefixSumChecksIo<Block<F>>;
