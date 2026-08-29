use crate::*;
#[cfg(feature = "debug")]
use backtrace::Backtrace;
use p3_field::{AbstractExtensionField, AbstractField};
use serde::{Deserialize, Serialize};

#[cfg(any(test, feature = "program_validation"))]
use smallvec::SmallVec;

use std::borrow::Borrow;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Instruction<F> {
    BaseAlu(BaseAluInstr<F>),
    ExtAlu(ExtAluInstr<F>),
    Mem(MemInstr<F>),
    Poseidon2(Box<Poseidon2Instr<F>>),
    /// Poseidon2 permutation specialised for the skinny chip (one-round-per-row layout).
    /// Uses `scratch_addrs` to chain rounds via memory lookup.
    /// Used by both BabyBear (`Poseidon2SkinnyChip`) and KoalaBear (`Poseidon2SkinnyKbChip`);
    /// the scratch array length differs by feature flag.
    Poseidon2Skinny(Box<Poseidon2SkinnyInstr<F>>),
    Select(SelectInstr<F>),
    HintBits(HintBitsInstr<F>),
    HintAddCurve(Box<HintAddCurveInstr<F>>),
    Print(PrintInstr<F>),
    HintExt2Felts(HintExt2FeltsInstr<F>),
    CommitPublicValues(Box<CommitPublicValuesInstr<F>>),
    Hint(HintInstr<F>),
    PolyEval(PolyEvalInstr<F>),
    ExtExpReverseBits(ExtExpReverseBitsInstr<F>),
    SumcheckRound(Box<SumcheckRoundInstr<F>>),
    PrefixSumChecks(Box<PrefixSumChecksInstr<F>>),
    #[cfg(feature = "debug")]
    DebugBacktrace(Backtrace),
}

impl<F: Copy> Instruction<F> {
    #[cfg(any(test, feature = "program_validation"))]
    #[allow(clippy::type_complexity)]
    #[must_use]
    pub(crate) fn io_addrs(&self) -> (SmallVec<[Address<F>; 4]>, SmallVec<[Address<F>; 4]>) {
        use smallvec::{smallvec as svec, *};
        use std::iter;

        match *self {
            Instruction::BaseAlu(BaseAluInstr { addrs: BaseAluIo { out, in1, in2 }, .. }) => {
                (svec![in1, in2], svec![out])
            }
            Instruction::ExtAlu(ExtAluInstr { addrs: ExtAluIo { out, in1, in2 }, .. }) => {
                (svec![in1, in2], svec![out])
            }
            Instruction::Mem(MemInstr { addrs: MemIo { inner }, .. }) => (svec![], svec![inner]),
            Instruction::Poseidon2(ref instr) => {
                let Poseidon2WideInstr { addrs: Poseidon2Io { input, output }, .. } =
                    instr.as_ref();
                (SmallVec::from_slice(input), SmallVec::from_slice(output))
            }
            Instruction::Poseidon2Skinny(ref instr) => {
                let Poseidon2SkinnyInstr { addrs: Poseidon2Io { input, output }, .. } =
                    instr.as_ref();
                (SmallVec::from_slice(input), SmallVec::from_slice(output))
            }
            Instruction::Select(SelectInstr {
                addrs: SelectIo { bit, out1, out2, in1, in2 },
                ..
            }) => (svec![bit, in1, in2], svec![out1, out2]),
            Instruction::HintBits(HintBitsInstr { ref output_addrs_mults, input_addr }) => {
                (svec![input_addr], output_addrs_mults.iter().map(|(a, _)| *a).collect())
            }
            Instruction::HintAddCurve(ref instr) => {
                let HintAddCurveInstr {
                    output_x_addrs_mults,
                    output_y_addrs_mults,
                    input1_x_addrs,
                    input1_y_addrs,
                    input2_x_addrs,
                    input2_y_addrs,
                } = instr.as_ref();
                (
                    [input1_x_addrs, input1_y_addrs, input2_x_addrs, input2_y_addrs]
                        .into_iter()
                        .flatten()
                        .copied()
                        .collect(),
                    [output_x_addrs_mults, output_y_addrs_mults]
                        .into_iter()
                        .flatten()
                        .map(|&(addr, _)| addr)
                        .collect(),
                )
            }
            Instruction::Print(_) => Default::default(),
            #[cfg(feature = "debug")]
            Instruction::DebugBacktrace(_) => Default::default(),
            Instruction::HintExt2Felts(HintExt2FeltsInstr { output_addrs_mults, input_addr }) => {
                (svec![input_addr], output_addrs_mults.iter().map(|(a, _)| *a).collect())
            }
            Instruction::CommitPublicValues(ref instr) => {
                let CommitPublicValuesInstr { pv_addrs } = instr.as_ref();
                (pv_addrs.as_array().to_vec().into(), svec![])
            }
            Instruction::Hint(HintInstr { ref output_addrs_mults }) => {
                (svec![], output_addrs_mults.iter().map(|(a, _)| *a).collect())
            }
            Instruction::PolyEval(PolyEvalInstr {
                addrs: PolyEvalIo { point, ref coeff, out },
                ..
            }) => (coeff.iter().copied().chain(iter::once(point)).collect(), svec![out]),
            Instruction::ExtExpReverseBits(ExtExpReverseBitsInstr {
                addrs: ExtExpReverseBitsIo { base, ref exp, ref acc_vec, .. },
                ..
            }) => (
                exp.iter().copied().chain(iter::once(base)).collect(),
                svec![*acc_vec.last().unwrap()],
            ),
            Instruction::SumcheckRound(ref instr) => {
                let SumcheckRoundInstr {
                    addrs: SumcheckRoundIo { ref coeffs, challenge, claim, new_claim },
                    ..
                } = **instr;
                (coeffs.iter().copied().chain([challenge, claim]).collect(), svec![new_claim])
            }
            Instruction::PrefixSumChecks(ref instr) => {
                let PrefixSumChecksInstr {
                    addrs: PrefixSumChecksIo { ref x1_vec, ref x2_vec, ref acc_vec, .. },
                    ..
                } = **instr;
                (
                    x1_vec.iter().copied().chain(x2_vec.iter().copied()).collect(),
                    svec![*acc_vec.last().unwrap()],
                )
            }
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HintBitsInstr<F> {
    /// Addresses and mults of the output bits.
    pub output_addrs_mults: Vec<(Address<F>, F)>,
    /// Input value to decompose.
    pub input_addr: Address<F>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PrintInstr<F> {
    pub field_elt_type: FieldEltType,
    pub addr: Address<F>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HintAddCurveInstr<F> {
    pub output_x_addrs_mults: Vec<(Address<F>, F)>,
    pub output_y_addrs_mults: Vec<(Address<F>, F)>,
    pub input1_x_addrs: Vec<Address<F>>,
    pub input1_y_addrs: Vec<Address<F>>,
    pub input2_x_addrs: Vec<Address<F>>,
    pub input2_y_addrs: Vec<Address<F>>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HintInstr<F> {
    /// Addresses and mults of the output felts.
    pub output_addrs_mults: Vec<(Address<F>, F)>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HintExt2FeltsInstr<F> {
    /// Addresses and mults of the output bits.
    pub output_addrs_mults: [(Address<F>, F); D],
    /// Input value to decompose.
    pub input_addr: Address<F>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum FieldEltType {
    Base,
    Extension,
}

pub fn base_alu<F: AbstractField>(
    opcode: BaseAluOpcode,
    mult: u32,
    out: u32,
    in1: u32,
    in2: u32,
) -> Instruction<F> {
    Instruction::BaseAlu(BaseAluInstr {
        opcode,
        mult: F::from_canonical_u32(mult),
        addrs: BaseAluIo {
            out: Address(F::from_canonical_u32(out)),
            in1: Address(F::from_canonical_u32(in1)),
            in2: Address(F::from_canonical_u32(in2)),
        },
    })
}

pub fn ext_alu<F: AbstractField>(
    opcode: ExtAluOpcode,
    mult: u32,
    out: u32,
    in1: u32,
    in2: u32,
) -> Instruction<F> {
    Instruction::ExtAlu(ExtAluInstr {
        opcode,
        mult: F::from_canonical_u32(mult),
        addrs: ExtAluIo {
            out: Address(F::from_canonical_u32(out)),
            in1: Address(F::from_canonical_u32(in1)),
            in2: Address(F::from_canonical_u32(in2)),
        },
    })
}

pub fn mem<F: AbstractField>(
    kind: MemAccessKind,
    mult: u32,
    addr: u32,
    val: u32,
) -> Instruction<F> {
    mem_single(kind, mult, addr, F::from_canonical_u32(val))
}

pub fn mem_single<F: AbstractField>(
    kind: MemAccessKind,
    mult: u32,
    addr: u32,
    val: F,
) -> Instruction<F> {
    mem_block(kind, mult, addr, Block::from(val))
}

pub fn mem_ext<F: AbstractField + Copy, EF: AbstractExtensionField<F>>(
    kind: MemAccessKind,
    mult: u32,
    addr: u32,
    val: EF,
) -> Instruction<F> {
    mem_block(kind, mult, addr, val.as_base_slice().into())
}

pub fn mem_block<F: AbstractField>(
    kind: MemAccessKind,
    mult: u32,
    addr: u32,
    val: Block<F>,
) -> Instruction<F> {
    Instruction::Mem(MemInstr {
        addrs: MemIo { inner: Address(F::from_canonical_u32(addr)) },
        vals: MemIo { inner: val },
        mult: F::from_canonical_u32(mult),
        kind,
    })
}

pub fn poseidon2<F: AbstractField>(
    mults: [u32; WIDTH],
    output: [u32; WIDTH],
    input: [u32; WIDTH],
) -> Instruction<F> {
    Instruction::Poseidon2(Box::new(Poseidon2Instr {
        mults: mults.map(F::from_canonical_u32),
        addrs: Poseidon2Io {
            output: output.map(F::from_canonical_u32).map(Address),
            input: input.map(F::from_canonical_u32).map(Address),
        },
    }))
}

#[allow(clippy::too_many_arguments)]
pub fn select<F: AbstractField>(
    mult1: u32,
    mult2: u32,
    bit: u32,
    out1: u32,
    out2: u32,
    in1: u32,
    in2: u32,
) -> Instruction<F> {
    Instruction::Select(SelectInstr {
        mult1: F::from_canonical_u32(mult1),
        mult2: F::from_canonical_u32(mult2),
        addrs: SelectIo {
            bit: Address(F::from_canonical_u32(bit)),
            out1: Address(F::from_canonical_u32(out1)),
            out2: Address(F::from_canonical_u32(out2)),
            in1: Address(F::from_canonical_u32(in1)),
            in2: Address(F::from_canonical_u32(in2)),
        },
    })
}

pub fn commit_public_values<F: AbstractField>(
    public_values_a: &RecursionPublicValues<u32>,
) -> Instruction<F> {
    let pv_a = public_values_a.as_array().map(|pv| Address(F::from_canonical_u32(pv)));
    let pv_address: &RecursionPublicValues<Address<F>> = pv_a.as_slice().borrow();

    Instruction::CommitPublicValues(Box::new(CommitPublicValuesInstr {
        pv_addrs: pv_address.clone(),
    }))
}

pub fn poly_eval<F: AbstractField>(mult: u32, point: F, coeff: Vec<F>, out: F) -> Instruction<F> {
    let num_coeffs = coeff.len();
    let num_chains = num_coeffs.saturating_sub(1);
    Instruction::PolyEval(PolyEvalInstr {
        mult: F::from_canonical_u32(mult),
        addrs: PolyEvalIo {
            point: Address(point),
            coeff: coeff.into_iter().map(Address).collect(),
            out: Address(out),
        },
        chain_accum_addrs: (0..num_chains)
            .map(|i| Address(F::from_canonical_usize(900000 + i)))
            .collect(),
    })
}

pub fn ext_exp_reverse_bits<F: AbstractField>(
    mult: u32,
    base: F,
    exp: Vec<F>,
    prev_acc_vec: Vec<F>,
    acc_vec: Vec<F>,
) -> Instruction<F> {
    Instruction::ExtExpReverseBits(ExtExpReverseBitsInstr {
        mult: F::from_canonical_u32(mult),
        addrs: ExtExpReverseBitsIo {
            base: Address(base),
            exp: exp.into_iter().map(Address).collect(),
            prev_acc_vec: prev_acc_vec.into_iter().map(Address).collect(),
            acc_vec: acc_vec.into_iter().map(Address).collect(),
        },
    })
}

pub fn sumcheck_round<F: AbstractField>(
    mult: u32,
    coeffs: Vec<F>,
    challenge: F,
    claim: F,
    new_claim: F,
) -> Instruction<F> {
    // Reverse from LOW-to-HIGH (caller convention) to HIGH-to-LOW (chip convention).
    let mut reversed = coeffs;
    reversed.reverse();
    let num_coeffs = reversed.len();
    let num_chains = num_coeffs.saturating_sub(1);
    Instruction::SumcheckRound(Box::new(SumcheckRoundInstr {
        mult: F::from_canonical_u32(mult),
        addrs: SumcheckRoundIo {
            coeffs: reversed.into_iter().map(Address).collect(),
            challenge: Address(challenge),
            claim: Address(claim),
            new_claim: Address(new_claim),
        },
        chain_rs_addrs: (0..num_chains)
            .map(|i| Address(F::from_canonical_usize(700000 + i)))
            .collect(),
        chain_ha_addrs: (0..num_chains)
            .map(|i| Address(F::from_canonical_usize(710000 + i)))
            .collect(),
    }))
}

pub fn prefix_sum_checks<F: AbstractField>(
    mult: u32,
    x1_vec: Vec<F>,
    x2_vec: Vec<F>,
    prev_acc_vec: Vec<F>,
    acc_vec: Vec<F>,
) -> Instruction<F> {
    Instruction::PrefixSumChecks(Box::new(PrefixSumChecksInstr {
        mult: F::from_canonical_u32(mult),
        addrs: PrefixSumChecksIo {
            x1_vec: x1_vec.into_iter().map(Address).collect(),
            x2_vec: x2_vec.into_iter().map(Address).collect(),
            prev_acc_vec: prev_acc_vec.into_iter().map(Address).collect(),
            acc_vec: acc_vec.into_iter().map(Address).collect(),
        },
    }))
}
