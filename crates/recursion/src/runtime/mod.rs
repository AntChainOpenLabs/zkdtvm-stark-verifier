pub mod instruction;
mod memory;
mod opcode;
mod program;
mod record;

// Avoid triggering annoying branch of thiserror derive macro.
use crate::air::{BinomialExtensionUtils, Block, RECURSIVE_PROOF_NUM_PV_ELTS};
use backtrace::Backtrace as Trace;
use dt_stark::air::BinomialExtension;
use dt_stark::{septic_curve::SepticCurve, septic_extension::SepticExtension, MachineRecord};
pub use instruction::Instruction;
use instruction::{
    FieldEltType, HintAddCurveInstr, HintBitsInstr, HintExt2FeltsInstr, HintInstr, PrintInstr,
};
use itertools::Itertools;
use memory::*;
pub use opcode::*;
use p3_field::{AbstractExtensionField, AbstractField, ExtensionField, PrimeField32};
use p3_maybe_rayon::prelude::*;
use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
use p3_symmetric::{CryptographicPermutation, Permutation};
pub use program::*;
pub use record::*;
use std::{
    array,
    borrow::Borrow,
    collections::VecDeque,
    fmt::Debug,
    io::{stdout, Write},
    iter::zip,
    marker::PhantomData,
    mem::MaybeUninit,
    sync::{atomic::{AtomicUsize, Ordering}, Arc, Mutex},
};
use thiserror::Error;

/// TODO expand glob import once things are organized enough
use crate::*;

/// The heap pointer address.
pub const HEAP_PTR: i32 = -4;
pub const STACK_SIZE: usize = 1 << 24;
pub const HEAP_START_ADDRESS: usize = STACK_SIZE + 4;
pub const MEMORY_SIZE: usize = 1 << 28;

/// The width of the Poseidon2 permutation.
pub const PERMUTATION_WIDTH: usize = 16;
/// cbindgen:ignore
#[cfg(feature = "babybear")]
pub const POSEIDON2_SBOX_DEGREE: u64 = 7;
/// cbindgen:ignore
#[cfg(feature = "koalabear")]
pub const POSEIDON2_SBOX_DEGREE: u64 = 3;
pub const HASH_RATE: usize = 8;

/// The current verifier implementation assumes that we are using a 256-bit hash with 32-bit
/// elements.
pub const DIGEST_SIZE: usize = 8;

pub const NUM_BITS: usize = 31;

pub const D: usize = 4;

type Perm<F, Diffusion, const SBOX: u64> =
    Poseidon2<F, Poseidon2ExternalMatrixGeneral, Diffusion, PERMUTATION_WIDTH, SBOX>;

/// TODO fully document.
/// Taken from [`dt_recursion_core::runtime::Runtime`].
/// Many missing things (compared to the old `Runtime`) will need to be implemented.
pub struct Runtime<
    'a,
    F: PrimeField32,
    EF: ExtensionField<F>,
    Diffusion,
    const SBOX_DEGREE: u64 = 7,
> {
    pub timestamp: usize,

    pub nb_poseidons: usize,

    pub nb_wide_poseidons: usize,

    pub nb_bit_decompositions: usize,

    pub nb_ext_ops: usize,

    pub nb_base_ops: usize,

    pub nb_memory_ops: usize,

    pub nb_branch_ops: usize,

    pub nb_select: usize,

    pub nb_exp_reverse_bits: usize,

    pub nb_print_f: usize,

    pub nb_print_e: usize,

    pub nb_poly_eval: usize,

    pub nb_ext_exp_reverse_bits: usize,

    /// The program.
    pub program: Arc<RecursionProgram<F>>,

    /// Memory. From canonical usize of an Address to a MemoryEntry.
    pub memory: MemVec<F>,

    /// The execution record.
    pub record: ExecutionRecord<F>,

    pub witness_stream: VecDeque<Block<F>>,

    /// The stream that print statements write to.
    pub debug_stdout: Box<dyn Write + Send + 'a>,

    /// Entries for dealing with the Poseidon2 hash state.
    perm: Option<Perm<F, Diffusion, SBOX_DEGREE>>,

    _marker_ef: PhantomData<EF>,

    _marker_diffusion: PhantomData<Diffusion>,
}

#[derive(Error, Debug)]
pub enum RuntimeError<F: Debug, EF: Debug> {
    #[error(
        "attempted to perform base field division {in1:?}/{in2:?}\n\
        \tin instruction {instr:#?}\n\
        \tnearest backtrace:\n{trace:#?}"
    )]
    DivFOutOfDomain { in1: F, in2: F, instr: BaseAluInstr<F>, trace: Option<Trace> },
    #[error(
        "attempted to perform extension field division {in1:?}/{in2:?}\n\
        \tin instruction {instr:#?}\n\
        \tnearest backtrace:\n{trace:#?}"
    )]
    DivEOutOfDomain { in1: EF, in2: EF, instr: ExtAluInstr<F>, trace: Option<Trace> },
    #[error("failed to print to `debug_stdout`: {0}")]
    DebugPrint(#[from] std::io::Error),
    #[error("attempted to read from empty witness stream")]
    EmptyWitnessStream,
}

/// Collects events for instruction types that contain Vec fields,
/// which cannot be stored in UnsafeRecord.
struct VecEventCollector<F> {
    poly_eval_events: Mutex<Vec<PolyEvalEvent<F>>>,
    ext_exp_reverse_bits_events: Mutex<Vec<ExtExpReverseBitsEvent<F>>>,
    prefix_sum_checks_events: Mutex<Vec<PrefixSumChecksEvent<F>>>,
    public_values: Mutex<Option<RecursionPublicValues<F>>>,
}

impl<F: Default + Copy> VecEventCollector<F> {
    fn new() -> Self {
        Self {
            poly_eval_events: Mutex::new(Vec::new()),
            ext_exp_reverse_bits_events: Mutex::new(Vec::new()),
            prefix_sum_checks_events: Mutex::new(Vec::new()),
            public_values: Mutex::new(None),
        }
    }

    fn into_parts(
        self,
    ) -> (
        RecursionPublicValues<F>,
        Vec<PolyEvalEvent<F>>,
        Vec<ExtExpReverseBitsEvent<F>>,
        Vec<PrefixSumChecksEvent<F>>,
    ) {
        let pv = self.public_values.into_inner().unwrap().unwrap_or_default();
        (
            pv,
            self.poly_eval_events.into_inner().unwrap(),
            self.ext_exp_reverse_bits_events.into_inner().unwrap(),
            self.prefix_sum_checks_events.into_inner().unwrap(),
        )
    }

    /// Absorb all events from  into , preserving insertion order.
    fn absorb(&self, other: Self) {
        self.poly_eval_events.lock().unwrap()
            .extend(other.poly_eval_events.into_inner().unwrap());
        self.ext_exp_reverse_bits_events.lock().unwrap()
            .extend(other.ext_exp_reverse_bits_events.into_inner().unwrap());
        self.prefix_sum_checks_events.lock().unwrap()
            .extend(other.prefix_sum_checks_events.into_inner().unwrap());
    }
}

impl<F: PrimeField32, EF: ExtensionField<F>, Diffusion, const SBOX_DEGREE: u64>
    Runtime<'_, F, EF, Diffusion, SBOX_DEGREE>
where
    Poseidon2<F, Poseidon2ExternalMatrixGeneral, Diffusion, PERMUTATION_WIDTH, SBOX_DEGREE>:
        CryptographicPermutation<[F; PERMUTATION_WIDTH]>,
{
    pub fn new(
        program: Arc<RecursionProgram<F>>,
        perm: Poseidon2<
            F,
            Poseidon2ExternalMatrixGeneral,
            Diffusion,
            PERMUTATION_WIDTH,
            SBOX_DEGREE,
        >,
    ) -> Self {
        let record = ExecutionRecord::<F> { program: program.clone(), ..Default::default() };
        let memory = MemVec::with_capacity(program.total_memory);
        Self {
            timestamp: 0,
            nb_poseidons: 0,
            nb_wide_poseidons: 0,
            nb_bit_decompositions: 0,
            nb_select: 0,
            nb_exp_reverse_bits: 0,
            nb_ext_ops: 0,
            nb_base_ops: 0,
            nb_memory_ops: 0,
            nb_branch_ops: 0,
            nb_print_f: 0,
            nb_print_e: 0,
            nb_poly_eval: 0,
            nb_ext_exp_reverse_bits: 0,
            program,
            memory,
            record,
            witness_stream: VecDeque::new(),
            debug_stdout: Box::new(stdout()),
            perm: Some(perm),
            _marker_ef: PhantomData,
            _marker_diffusion: PhantomData,
        }
    }

    pub fn print_stats(&self) {
        if tracing::event_enabled!(tracing::Level::DEBUG) {
            let mut stats = self.record.stats().into_iter().collect::<Vec<_>>();
            stats.sort_unstable();
            tracing::debug!("total events: {}", stats.iter().map(|(_, v)| *v).sum::<usize>());
            for (k, v) in stats {
                tracing::debug!("  {k}: {v}");
            }
        }
    }

    #[inline]
    unsafe fn execute_one(
        state: &mut ExecState<F, Diffusion, SBOX_DEGREE>,
        record: &UnsafeRecord<F>,
        vec_events: &VecEventCollector<F>,
        witness_stream: Option<&mut VecDeque<Block<F>>>,
        instruction: &Instruction<F>,
        offset: usize,
        resolved: &[u32],
    ) -> Result<(), RuntimeError<F, EF>> {
        let ExecEnv { memory, perm, debug_stdout: _ } = state.env;
        match instruction {
            Instruction::BaseAlu(BaseAluInstr { opcode, mult: _, addrs }) => {
                let in1 = memory.mr_at(resolved[0] as usize).val[0];
                let in2 = memory.mr_at(resolved[1] as usize).val[0];
                let out = match opcode {
                    BaseAluOpcode::AddF => in1 + in2,
                    BaseAluOpcode::SubF => in1 - in2,
                    BaseAluOpcode::MulF => in1 * in2,
                    BaseAluOpcode::DivF => match in1.try_div(in2) {
                        Some(x) => x,
                        None if in1.is_zero() => AbstractField::one(),
                        None => {
                            return Err(RuntimeError::DivFOutOfDomain {
                                in1,
                                in2,
                                instr: BaseAluInstr {
                                    opcode: *opcode,
                                    mult: F::zero(),
                                    addrs: *addrs,
                                },
                                trace: state.resolve_trace().cloned(),
                            });
                        }
                    },
                };
                memory.mw_at(resolved[2] as usize, Block::from(out));
                record.base_alu_events.get_unchecked(offset).get()
                    .write(MaybeUninit::new(BaseAluEvent { out, in1, in2 }));
            }
            Instruction::ExtAlu(ExtAluInstr { opcode, mult: _, addrs }) => {
                let in1 = memory.mr_at(resolved[0] as usize).val;
                let in2 = memory.mr_at(resolved[1] as usize).val;
                let out = match opcode {
                    ExtAluOpcode::AddE => {
                        #[cfg(target_arch = "x86_64")]
                        {
                            use core::arch::x86_64::*;
                            unsafe {
                                let p = _mm_set1_epi32(0x7f000001u32 as i32);
                                let a = _mm_loadu_si128(in1.0.as_ptr() as *const __m128i);
                                let b = _mm_loadu_si128(in2.0.as_ptr() as *const __m128i);
                                let t = _mm_add_epi32(a, b);
                                let u = _mm_sub_epi32(t, p);
                                let r = _mm_min_epu32(t, u);
                                let mut out = Block::default();
                                _mm_storeu_si128(out.0.as_mut_ptr() as *mut __m128i, r);
                                out
                            }
                        }
                        #[cfg(not(target_arch = "x86_64"))]
                        {
                            let ef = EF::from_base_fn(|i| in1.0[i]) + EF::from_base_fn(|i| in2.0[i]);
                            Block::from(ef.as_base_slice())
                        }
                    }
                    ExtAluOpcode::SubE => {
                        #[cfg(target_arch = "x86_64")]
                        {
                            use core::arch::x86_64::*;
                            unsafe {
                                let p = _mm_set1_epi32(0x7f000001u32 as i32);
                                let a = _mm_loadu_si128(in1.0.as_ptr() as *const __m128i);
                                let b = _mm_loadu_si128(in2.0.as_ptr() as *const __m128i);
                                let t = _mm_sub_epi32(a, b);
                                let u = _mm_add_epi32(t, p);
                                let r = _mm_min_epu32(t, u);
                                let mut out = Block::default();
                                _mm_storeu_si128(out.0.as_mut_ptr() as *mut __m128i, r);
                                out
                            }
                        }
                        #[cfg(not(target_arch = "x86_64"))]
                        {
                            let ef = EF::from_base_fn(|i| in1.0[i]) - EF::from_base_fn(|i| in2.0[i]);
                            Block::from(ef.as_base_slice())
                        }
                    }
                    ExtAluOpcode::MulE => {
                        let in1_ef = EF::from_base_fn(|i| in1.0[i]);
                        let in2_ef = EF::from_base_fn(|i| in2.0[i]);
                        Block::from((in1_ef * in2_ef).as_base_slice())
                    }
                    ExtAluOpcode::DivE => {
                        let in1_ef = EF::from_base_fn(|i| in1.0[i]);
                        let in2_ef = EF::from_base_fn(|i| in2.0[i]);
                        let out_ef = match in1_ef.try_div(in2_ef) {
                            Some(x) => x,
                            None if in1_ef.is_zero() => AbstractField::one(),
                            None => {
                                return Err(RuntimeError::DivEOutOfDomain {
                                    in1: in1_ef,
                                    in2: in2_ef,
                                    instr: ExtAluInstr {
                                        opcode: *opcode,
                                        mult: F::zero(),
                                        addrs: *addrs,
                                    },
                                    trace: state.resolve_trace().cloned(),
                                });
                            }
                        };
                        Block::from(out_ef.as_base_slice())
                    }
                };
                let out = out;
                memory.mw_at(resolved[2] as usize, out);
                record.ext_alu_events.get_unchecked(offset).get()
                    .write(MaybeUninit::new(ExtAluEvent { out, in1, in2 }));
            }
            Instruction::Mem(MemInstr {
                addrs: MemIo { inner: addr },
                vals: MemIo { inner: val },
                mult: _,
                kind,
            }) => {
                let (addr, val) = (*addr, *val);
                match kind {
                    MemAccessKind::Read => {
                        let mem_entry = memory.mr_at(resolved[0] as usize);
                        assert_eq!(
                            mem_entry.val, val,
                            "stored memory value should be the specified value"
                        );
                    }
                    MemAccessKind::Write => memory.mw_at(resolved[0] as usize, val),
                }
            }
            Instruction::Poseidon2(instr) => {
                let Poseidon2Instr { addrs: Poseidon2Io { input, output }, mults: _ } =
                    instr.as_ref();
                let in_vals = std::array::from_fn(|i| memory.mr_at(resolved[i] as usize).val[0]);
                let mut perm_output = in_vals;
                perm.permute_mut(&mut perm_output);

                for i in 0..16 {
                    memory.mw_at(resolved[16 + i] as usize, Block::from(perm_output[i]));
                }
                record.poseidon2_events.get_unchecked(offset).get().write(MaybeUninit::new(
                    Poseidon2Event { input: in_vals, output: perm_output },
                ));
            }
            Instruction::Poseidon2Skinny(instr) => {
                let Poseidon2SkinnyInstr { addrs: Poseidon2Io { input, output }, mults: _, scratch_addrs: _ } =
                    instr.as_ref();
                let in_vals = std::array::from_fn(|i| memory.mr_at(resolved[i] as usize).val[0]);
                let mut perm_output = in_vals;
                perm.permute_mut(&mut perm_output);

                for i in 0..16 {
                    memory.mw_at(resolved[16 + i] as usize, Block::from(perm_output[i]));
                }
                record.poseidon2_skinny_events.get_unchecked(offset).get().write(MaybeUninit::new(
                    Poseidon2Event { input: in_vals, output: perm_output },
                ));
            }
            Instruction::Select(SelectInstr {
                addrs: SelectIo { bit, out1, out2, in1, in2 },
                mult1: _,
                mult2: _,
            }) => {
                let bit = memory.mr_at(resolved[0] as usize).val[0];
                let in1 = memory.mr_at(resolved[1] as usize).val[0];
                let in2 = memory.mr_at(resolved[2] as usize).val[0];
                let bit_diff = bit * (in2 - in1);
                let out1_val = in1 + bit_diff;
                let out2_val = in2 - bit_diff;
                memory.mw_at(resolved[3] as usize, Block::from(out1_val));
                memory.mw_at(resolved[4] as usize, Block::from(out2_val));
                record.select_events.get_unchecked(offset).get().write(MaybeUninit::new(SelectEvent {
                    bit,
                    out1: out1_val,
                    out2: out2_val,
                    in1,
                    in2,
                }));
            }
            Instruction::HintBits(HintBitsInstr { output_addrs_mults, input_addr }) => {
                let num = memory.mr_at(resolved[0] as usize).val[0].as_canonical_u32();
                for (i, &(_addr, _mult)) in output_addrs_mults.iter().enumerate() {
                    let bit = Block::from(F::from_canonical_u32((num >> i) & 1));
                    memory.mw_at(resolved[1 + i] as usize, bit);
                    record.mem_var_events.get_unchecked(offset + i).get()
                        .write(MaybeUninit::new(MemEvent { inner: bit }));
                }
            }
            Instruction::HintAddCurve(instr) => {
                let HintAddCurveInstr {
                    output_x_addrs_mults,
                    output_y_addrs_mults,
                    input1_x_addrs,
                    input1_y_addrs,
                    input2_x_addrs,
                    input2_y_addrs,
                } = instr.as_ref();
                let input1_x = SepticExtension::<F>::from_base_fn(|i| {
                    memory.mr_at(resolved[i] as usize).val[0]
                });
                let input1_y = SepticExtension::<F>::from_base_fn(|i| {
                    memory.mr_at(resolved[7 + i] as usize).val[0]
                });
                let input2_x = SepticExtension::<F>::from_base_fn(|i| {
                    memory.mr_at(resolved[14 + i] as usize).val[0]
                });
                let input2_y = SepticExtension::<F>::from_base_fn(|i| {
                    memory.mr_at(resolved[21 + i] as usize).val[0]
                });
                let point1 = SepticCurve { x: input1_x, y: input1_y };
                let point2 = SepticCurve { x: input2_x, y: input2_y };
                let output = point1.add_incomplete(point2);

                let out_base = 28;
                for (i, (val, &(_addr, _mult))) in output.x.0.into_iter().zip(output_x_addrs_mults.iter()).enumerate()
                {
                    memory.mw_at(resolved[out_base + i] as usize, Block::from(val));
                    record.mem_var_events.get_unchecked(offset + i).get()
                        .write(MaybeUninit::new(MemEvent { inner: Block::from(val) }));
                }
                let y_base = offset + output_x_addrs_mults.len();
                let out_y_base = 28 + output_x_addrs_mults.len();
                for (i, (val, &(_addr, _mult))) in output.y.0.into_iter().zip(output_y_addrs_mults.iter()).enumerate()
                {
                    memory.mw_at(resolved[out_y_base + i] as usize, Block::from(val));
                    record.mem_var_events.get_unchecked(y_base + i).get()
                        .write(MaybeUninit::new(MemEvent { inner: Block::from(val) }));
                }
            }
            Instruction::CommitPublicValues(instr) => {
                let pv_values: [F; RECURSIVE_PROOF_NUM_PV_ELTS] =
                    array::from_fn(|i| memory.mr_at(resolved[i] as usize).val[0]);
                let pv: RecursionPublicValues<F> = *pv_values.as_slice().borrow();
                *vec_events.public_values.lock().unwrap() = Some(pv);
                record.commit_pv_hash_events.get_unchecked(offset).get().write(MaybeUninit::new(
                    CommitPublicValuesEvent { public_values: pv },
                ));
            }

            Instruction::Print(PrintInstr { field_elt_type, addr }) => match field_elt_type {
                FieldEltType::Base => {
                    let f = memory.mr_at(resolved[0] as usize).val[0];
                    tracing::trace!("PRINTF={f}");
                }
                FieldEltType::Extension => {
                    let ef = memory.mr_at(resolved[0] as usize).val;
                    tracing::trace!("PRINTEF={ef:?}");
                }
            },
            Instruction::HintExt2Felts(HintExt2FeltsInstr { output_addrs_mults, input_addr }) => {
                let fs = memory.mr_at(resolved[0] as usize).val;
                for (i, (f, &(_addr, _mult))) in fs.into_iter().zip(output_addrs_mults.iter()).enumerate() {
                    let felt = Block::from(f);
                    memory.mw_at(resolved[1 + i] as usize, felt);
                    record.mem_var_events.get_unchecked(offset + i).get()
                        .write(MaybeUninit::new(MemEvent { inner: felt }));
                }
            }
            Instruction::Hint(HintInstr { output_addrs_mults }) => {
                let witness_stream =
                    witness_stream.expect("hint should be called outside parallel contexts");
                if witness_stream.len() < output_addrs_mults.len() {
                    return Err(RuntimeError::EmptyWitnessStream);
                }
                let witness = witness_stream.drain(0..output_addrs_mults.len());
                for (i, (&(_addr, _mult), val)) in zip(output_addrs_mults, witness).enumerate() {
                    memory.mw_at(resolved[i] as usize, val);
                    record.mem_var_events.get_unchecked(offset + i).get()
                        .write(MaybeUninit::new(MemEvent { inner: val }));
                }
            }
            #[cfg(feature = "debug")]
            Instruction::DebugBacktrace(backtrace) => {
                state.last_trace = Some(backtrace.clone());
            }
            Instruction::PolyEval(PolyEvalInstr {
                addrs: PolyEvalIo { point, coeff, out },
                mult: _,
                chain_accum_addrs: _,
            }) => {
                let point = memory.mr_at(resolved[0] as usize).val[0];
                let coeffs: Vec<_> =
                    coeff.iter().enumerate().map(|(i, _)| memory.mr_at(resolved[1 + i] as usize).val[0]).collect();
                let result = coeffs[1..].iter().fold(coeffs[0], |acc, &x| acc * point + x);
                let out_idx = 1 + coeff.len();
                memory.mw_at(resolved[out_idx] as usize, Block::from(result));
                vec_events.poly_eval_events.lock().unwrap()
                    .push(PolyEvalEvent { out: result, point, coeff: coeffs });
            }
            Instruction::ExtExpReverseBits(ExtExpReverseBitsInstr {
                addrs: ExtExpReverseBitsIo { base, exp, prev_acc_vec, acc_vec },
                mult: _,
            }) => {
                let n = exp.len();
                let base_val = memory.mr_at(resolved[0] as usize).val;
                let exp_bits: Vec<_> =
                    (0..n).map(|i| memory.mr_at(resolved[1 + i] as usize).val[0]).collect();
                // prev_acc starts at offset 1+n, acc starts at 1+2n
                let mut prev_acc_vals = Vec::with_capacity(n);
                let mut acc_vals = Vec::with_capacity(n);
                let prev_acc_base = 1 + n;
                let acc_base = 1 + 2 * n;
                for i in 0..n {
                    let prev_acc_block = memory.mr_at(resolved[prev_acc_base + i] as usize).val;
                    prev_acc_vals.push(prev_acc_block);
                    let prev_ef = BinomialExtension::from_block(prev_acc_block);
                    let acc_ef = prev_ef * prev_ef * if exp_bits[i].as_canonical_u32() == 1 {
                        BinomialExtension::from_block(base_val)
                    } else {
                        BinomialExtension::from_base(F::one())
                    };
                    let acc_block = Block::from(acc_ef.0);
                    acc_vals.push(acc_block);
                    memory.mw_at(resolved[acc_base + i] as usize, acc_block);
                }
                vec_events.ext_exp_reverse_bits_events.lock().unwrap()
                    .push(ExtExpReverseBitsEvent {
                        base: base_val,
                        exp: exp_bits,
                        prev_acc_vec: prev_acc_vals,
                        acc_vec: acc_vals,
                    });
            }
            Instruction::PrefixSumChecks(ref instr) => {
                let PrefixSumChecksInstr {
                    addrs: PrefixSumChecksIo { x1_vec, x2_vec, prev_acc_vec, acc_vec },
                    mult: _,
                } = instr.as_ref();
                let n = x1_vec.len();
                // Layout: x1[0..n], x2[n..2n], prev_acc[2n..3n], acc[3n..4n]
                let x1_vals: Vec<_> = (0..n).map(|i| memory.mr_at(resolved[i] as usize).val).collect();
                let x2_vals: Vec<_> = (0..n).map(|i| memory.mr_at(resolved[n + i] as usize).val).collect();
                let prev_acc_base = 2 * n;
                let acc_base_idx = 3 * n;
                let mut prev_acc_vals = Vec::with_capacity(n);
                let mut acc_vals = Vec::with_capacity(n);
                for i in 0..n {
                    let prev_acc_block = memory.mr_at(resolved[prev_acc_base + i] as usize).val;
                    prev_acc_vals.push(prev_acc_block);
                    let prev_ef = EF::from_base_fn(|j| prev_acc_block.0[j]);
                    let x1_ef = EF::from_base_fn(|j| x1_vals[i].0[j]);
                    let x2_ef = EF::from_base_fn(|j| x2_vals[i].0[j]);
                    let eq_val = x1_ef * x2_ef + (EF::one() - x1_ef) * (EF::one() - x2_ef);
                    let acc_ef = prev_ef * eq_val;
                    let acc_block = Block::from(acc_ef.as_base_slice());
                    acc_vals.push(acc_block);
                    memory.mw_at(resolved[acc_base_idx + i] as usize, acc_block);
                }
                vec_events.prefix_sum_checks_events.lock().unwrap()
                    .push(PrefixSumChecksEvent {
                        x1_vec: x1_vals,
                        x2_vec: x2_vals,
                        prev_acc_vec: prev_acc_vals,
                        acc_vec: acc_vals,
                    });
            }
        }

        Ok(())
    }

    /// Execute a program using the pre-computed offset table.
    ///
    /// `offset_cursor` is an atomic counter that tracks the current position in the
    /// global `offsets` slice. Each instruction consumes one entry from the offset table.
    unsafe fn execute_raw_inner(
        env: &ExecEnv<F, Diffusion, SBOX_DEGREE>,
        program: &RawProgram<Instruction<F>>,
        offsets: &[usize],
        offset_cursor: &AtomicUsize,
        mut witness_stream: Option<&mut VecDeque<Block<F>>>,
        record: &UnsafeRecord<F>,
        vec_events: &VecEventCollector<F>,
        resolved_addrs: &[u32],
        addr_starts: &[u32],
    ) -> Result<(), RuntimeError<F, EF>> {
        let mut state = ExecState {
            env: env.clone(),
            #[cfg(feature = "debug")]
            last_trace: None,
        };

        for block in &program.seq_blocks {
            match block {
                SeqBlock::Basic(basic_block) => {
                    let instrs = &basic_block.instrs;
                    let mut local_idx = offset_cursor.load(Ordering::Relaxed);
                    let len = instrs.len();
                    for i in 0..len {
                        let instruction = unsafe { instrs.get_unchecked(i) };
                        let offset = unsafe { *offsets.get_unchecked(local_idx) };
                        let addr_start = (unsafe { *addr_starts.get_unchecked(local_idx) }) as usize;
                        let next_addr_end = if local_idx + 1 < addr_starts.len() {
                            (unsafe { *addr_starts.get_unchecked(local_idx + 1) }) as usize
                        } else {
                            resolved_addrs.len()
                        };
                        let resolved = unsafe { resolved_addrs.get_unchecked(addr_start..next_addr_end) };
                        local_idx += 1;
                        unsafe {
                            Self::execute_one(
                                &mut state,
                                record,
                                vec_events,
                                witness_stream.as_deref_mut(),
                                instruction,
                                offset,
                                resolved,
                            )
                        }?;
                    }
                    offset_cursor.store(local_idx, Ordering::Relaxed);
                }
                SeqBlock::Parallel(vec) => {
                    if vec.len() <= 1 {
                        for subprogram in vec {
                            Self::execute_raw_inner(
                                env,
                                subprogram,
                                offsets,
                                offset_cursor,
                                None,
                                record,
                                vec_events,
                                resolved_addrs,
                                addr_starts,
                            )?;
                        }
                    } else {
                        // Each parallel subprogram needs its own range of offsets.
                        // Pre-compute the instruction count per subprogram to assign
                        // non-overlapping offset ranges.
                        let sub_instr_counts: Vec<usize> = vec.iter()
                            .map(|sub| sub.iter().count())
                            .collect();
                        let base = offset_cursor.load(Ordering::Relaxed);
                        // Advance the global cursor past all subprograms.
                        let total: usize = sub_instr_counts.iter().sum();
                        offset_cursor.store(base + total, Ordering::Relaxed);

                        tracing::debug!(
                            "Parallel block: {} subprograms, {} total instructions, thread {:?}",
                            vec.len(),
                            total,
                            std::thread::current().id(),
                        );

                        // Build per-subprogram cursors with pre-assigned starting positions.
                        // Each sub-program gets its own VecEventCollector so that
                        // Mutex-based events are recorded in deterministic order
                        // (matching sequential execution).
                        let sub_tasks: Vec<(usize, &RawProgram<Instruction<F>>, VecEventCollector<F>)> = {
                            let mut pos = base;
                            sub_instr_counts.iter().zip(vec.iter()).map(|(&count, sub)| {
                                let start = pos;
                                pos += count;
                                (start, sub, VecEventCollector::new())
                            }).collect()
                        };

                        sub_tasks.par_iter().try_for_each(|(start, subprogram, sub_vec_events)| {
                            let sub_cursor = AtomicUsize::new(*start);
                            Self::execute_raw_inner(
                                env,
                                subprogram,
                                offsets,
                                &sub_cursor,
                                None,
                                record,
                                sub_vec_events,
                                resolved_addrs,
                                addr_starts,
                            )
                        })?;

                        // Merge per-subprogram events into the parent collector
                        // in deterministic (sequential) order.
                        for (_, _, sub_vec_events) in sub_tasks {
                            vec_events.absorb(sub_vec_events);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Run the program.

    pub fn run(&mut self) -> Result<(), RuntimeError<F, EF>> {
        let ec = &self.program.event_counts;
        tracing::info!(
            "RECURSION_INSTR_DIST base_alu={} ext_alu={} poseidon2={} select={} mem_const={} mem_var={} poly_eval={} ext_exp={} prefix={}",
            ec.base_alu_events, ec.ext_alu_events, ec.poseidon2_wide_events,
            ec.select_events, ec.mem_const_events, ec.mem_var_events,
            ec.poly_eval_events, ec.ext_exp_reverse_bits_events,
            ec.prefix_sum_checks_events,
        );
        let unsafe_record = UnsafeRecord::<F>::new(&self.program.event_counts);
        let vec_events = VecEventCollector::new();

        let env = ExecEnv {
            memory: &self.memory,
            perm: self.perm.as_ref().unwrap(),
            debug_stdout: &Mutex::new(&mut self.debug_stdout),
        };

        let offset_cursor = AtomicUsize::new(0);

        unsafe {
            Self::execute_raw_inner(
                &env,
                &self.program.inner,
                &self.program.offsets,
                &offset_cursor,
                Some(&mut self.witness_stream),
                &unsafe_record,
                &vec_events,
                &self.program.resolved_addrs,
                &self.program.addr_starts,
            )
        }?;

        let (public_values, poly, ext_exp, prefix) = vec_events.into_parts();
        self.record = unsafe {
            unsafe_record.into_record(
                self.program.clone(),
                public_values,
                poly,
                ext_exp,
                prefix,
            )
        };

        Ok(())
    }
}

struct ExecState<'a, 'b, F, Diffusion, const SBOX: u64 = 7> {
    pub env: ExecEnv<'a, 'b, F, Diffusion, SBOX>,
    #[cfg(feature = "debug")]
    pub last_trace: Option<Trace>,
}

impl<F, Diffusion, const SBOX: u64> ExecState<'_, '_, F, Diffusion, SBOX> {
    fn resolve_trace(&mut self) -> Option<&mut Trace> {
        cfg_if::cfg_if! {
            if #[cfg(feature = "debug")] {
                // False positive.
                #[allow(clippy::manual_inspect)]
                self.last_trace.as_mut().map(|trace| {
                    trace.resolve();
                    trace
                })
            } else {
                None
            }
        }
    }
}

impl<'a, 'b, F, Diffusion, const SBOX: u64> Clone for ExecState<'a, 'b, F, Diffusion, SBOX>
where
    ExecEnv<'a, 'b, F, Diffusion, SBOX>: Clone,
{
    fn clone(&self) -> Self {
        let Self {
            env,
            #[cfg(feature = "debug")]
            last_trace,
        } = self;
        Self {
            env: env.clone(),
            #[cfg(feature = "debug")]
            last_trace: last_trace.clone(),
        }
    }

    fn clone_from(&mut self, source: &Self) {
        let Self {
            env,
            #[cfg(feature = "debug")]
            last_trace,
        } = self;
        env.clone_from(&source.env);
        #[cfg(feature = "debug")]
        last_trace.clone_from(&source.last_trace);
    }
}

struct ExecEnv<'a, 'b, F, Diffusion, const SBOX: u64 = 7> {
    pub memory: &'a MemVec<F>,
    pub perm: &'a Perm<F, Diffusion, SBOX>,
    pub debug_stdout: &'a Mutex<dyn Write + Send + 'b>,
}

impl<F, Diffusion, const SBOX: u64> Clone for ExecEnv<'_, '_, F, Diffusion, SBOX> {
    fn clone(&self) -> Self {
        let Self { memory, perm, debug_stdout } = self;
        Self { memory, perm, debug_stdout }
    }

    fn clone_from(&mut self, source: &Self) {
        let Self { memory, perm, debug_stdout } = self;
        memory.clone_from(&source.memory);
        perm.clone_from(&source.perm);
        debug_stdout.clone_from(&source.debug_stdout);
    }
}
