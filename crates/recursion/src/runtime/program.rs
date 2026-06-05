use crate::*;
use dt_stark::{
    air::{MachineAir, MachineProgram},
    septic_digest::SepticDigest,
};
use p3_field::Field;
use serde::{Deserialize, Serialize};
use shape::RecursionShape;
use std::ops::Deref;

pub use basic_block::BasicBlock;
pub use raw::RawProgram;
pub use seq_block::SeqBlock;

/// A well-formed recursion program. See [`Self::new_unchecked`] for guaranteed (safety) invariants.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[repr(transparent)]
pub struct RecursionProgram<F>(RootProgram<F>);

impl<F> RecursionProgram<F> {
    /// # Safety
    /// The given program must be well formed. This is defined as the following:
    /// - reads are performed after writes, according to a "happens-before" relation; and
    /// - an address is written to at most once.
    ///
    /// The "happens-before" relation is defined as follows:
    /// - It is a strict partial order, meaning it is transitive, irreflexive, and asymmetric.
    /// - Instructions in a `BasicBlock` are linearly ordered.
    /// - `SeqBlock`s in a `RawProgram` are linearly ordered, meaning:
    ///     - Each `SeqBlock` has a set of initial instructions `I` and final instructions `O`.
    ///     - For `SeqBlock::Basic`:
    ///         - `I` is the singleton consisting of the first instruction in the enclosed
    ///           `BasicBlock`.
    ///         - `O` is the singleton consisting of the last instruction in the enclosed
    ///           `BasicBlock`.
    ///     - For `SeqBlock::Parallel`:
    ///         - `I` is the set of initial instructions `I` in the first `SeqBlock` of the enclosed
    ///           `RawProgram`.
    ///         - `O` is the set of final instructions in the last `SeqBlock` of the enclosed
    ///           `RawProgram`.
    ///     - For consecutive `SeqBlock`s, each element of the first one's `O` happens before the
    ///       second one's `I`.
    pub unsafe fn new_unchecked(program: RootProgram<F>) -> Self {
        Self(program)
    }

    pub fn into_inner(self) -> RootProgram<F> {
        self.0
    }

    pub fn shape_mut(&mut self) -> &mut Option<RecursionShape> {
        &mut self.0.shape
    }
}

impl<F> Default for RecursionProgram<F> {
    fn default() -> Self {
        // SAFETY: An empty program is always well formed.
        unsafe { Self::new_unchecked(RootProgram::default()) }
    }
}

impl<F> Deref for RecursionProgram<F> {
    type Target = RootProgram<F>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<F: Field> MachineProgram<F> for RecursionProgram<F> {
    fn pc_start(&self) -> F {
        F::zero()
    }

    fn initial_global_cumulative_sum(&self) -> SepticDigest<F> {
        SepticDigest::<F>::zero_for_field()
    }
}

impl<F: Field> RecursionProgram<F> {
    #[inline]
    pub fn fixed_log2_rows<A: MachineAir<F>>(&self, air: &A) -> Option<usize> {
        self.0
            .shape
            .as_ref()
            .map(|shape| {
                shape
                    .inner
                    .get(&air.name())
                    .unwrap_or_else(|| panic!("Chip {} not found in specified shape", air.name()))
            })
            .copied()
    }
}

#[cfg(any(test, feature = "program_validation"))]
pub use validation::*;

#[cfg(any(test, feature = "program_validation"))]
mod validation {
    use super::*;

    use std::{fmt::Debug, iter, mem};

    use p3_field::PrimeField32;
    use range_set_blaze::{MultiwayRangeSetBlazeRef, RangeSetBlaze};
    use smallvec::{smallvec, SmallVec};
    use thiserror::Error;

    #[derive(Error, Debug)]
    pub enum StructureError<F: Debug> {
        #[error("tried to read from uninitialized address {addr:?}. instruction: {instr:?}")]
        ReadFromUninit { addr: Address<F>, instr: Instruction<F> },
    }

    #[derive(Error, Debug)]
    pub enum SummaryError {
        #[error("`total_memory` is insufficient. configured: {configured}. required: {required}")]
        OutOfMemory { configured: usize, required: usize },
    }

    #[derive(Error, Debug)]
    pub enum ValidationError<F: Debug> {
        Structure(#[from] StructureError<F>),
        Summary(#[from] SummaryError),
    }

    impl<F: PrimeField32> RecursionProgram<F> {
        /// Validate the program without modifying its summary metadata.
        pub fn try_new_unmodified(
            program: RootProgram<F>,
        ) -> Result<Self, Box<ValidationError<F>>> {
            let written_addrs = try_written_addrs(smallvec![], &program.inner)
                .map_err(|e| ValidationError::from(*e))?;
            if let Some(required) = written_addrs.last().map(|x| x as usize + 1) {
                let configured = program.total_memory;
                if required > configured {
                    Err(Box::new(SummaryError::OutOfMemory { configured, required }.into()))?
                }
            }
            // SAFETY: We just checked all the invariants.
            Ok(unsafe { Self::new_unchecked(program) })
        }

        /// Validate the program, modifying summary metadata if necessary.
        pub fn try_new(mut program: RootProgram<F>) -> Result<Self, Box<StructureError<F>>> {
            let written_addrs = try_written_addrs(smallvec![], &program.inner)?;
            program.total_memory = written_addrs.last().map(|x| x as usize + 1).unwrap_or_default();
            // SAFETY: We just checked/enforced all the invariants.
            Ok(unsafe { Self::new_unchecked(program) })
        }
    }

    fn try_written_addrs<F: PrimeField32>(
        readable_addrs: SmallVec<[&RangeSetBlaze<u32>; 3]>,
        program: &RawProgram<Instruction<F>>,
    ) -> Result<RangeSetBlaze<u32>, Box<StructureError<F>>> {
        let mut written_addrs = RangeSetBlaze::<u32>::new();
        for block in &program.seq_blocks {
            match block {
                SeqBlock::Basic(basic_block) => {
                    for instr in &basic_block.instrs {
                        let (inputs, outputs) = instr.io_addrs();
                        inputs.into_iter().try_for_each(|i| {
                            let i_u32 = i.0.as_canonical_u32();
                            iter::once(&written_addrs)
                                .chain(readable_addrs.iter().copied())
                                .any(|s| s.contains(i_u32))
                                .then_some(())
                                .ok_or_else(|| {
                                    Box::new(StructureError::ReadFromUninit {
                                        addr: i,
                                        instr: instr.clone(),
                                    })
                                })
                        })?;
                        written_addrs.extend(outputs.iter().map(|o| o.0.as_canonical_u32()));
                    }
                }
                SeqBlock::Parallel(programs) => {
                    let par_written_addrs = programs
                        .iter()
                        .map(|subprogram| {
                            let sub_readable_addrs = iter::once(&written_addrs)
                                .chain(readable_addrs.iter().copied())
                                .collect();

                            try_written_addrs(sub_readable_addrs, subprogram)
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    written_addrs =
                        iter::once(mem::take(&mut written_addrs)).chain(par_written_addrs).union();
                }
            }
        }
        Ok(written_addrs)
    }

    impl<F: PrimeField32> RootProgram<F> {
        pub fn validate(self) -> Result<RecursionProgram<F>, Box<StructureError<F>>> {
            RecursionProgram::try_new(self)
        }
    }

    #[cfg(test)]
    pub fn linear_program<F: PrimeField32>(
        instrs: Vec<Instruction<F>>,
    ) -> Result<RecursionProgram<F>, Box<StructureError<F>>> {
        RootProgram::new(
            RawProgram { seq_blocks: vec![SeqBlock::Basic(BasicBlock { instrs })] },
            0, // Will be filled in.
            None,
        )
        .validate()
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RootProgram<F> {
    pub inner: raw::RawProgram<Instruction<F>>,
    pub total_memory: usize,
    pub shape: Option<RecursionShape>,
    /// Pre-computed event counts for UnsafeRecord allocation.
    #[serde(default)]
    pub event_counts: super::machine::RecursionAirEventCount,
    /// Per-instruction event offset table. `offsets[i]` is the offset for the i-th
    /// instruction (in program traversal order) into its corresponding event buffer.
    /// Computed once at construction time; zero runtime clone cost.
    #[serde(default)]
    pub offsets: Vec<usize>,
    /// Flat table of pre-resolved memory addresses (canonical u32 values).
    /// Eliminates Montgomery reduction from the hot path.
    #[serde(default)]
    pub resolved_addrs: Vec<u32>,
    /// `addr_starts[i]` is the starting index in `resolved_addrs` for the i-th instruction.
    #[serde(default)]
    pub addr_starts: Vec<u32>,
}

impl<F: Clone> Clone for RootProgram<F> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            total_memory: self.total_memory,
            shape: self.shape.clone(),
            event_counts: self.event_counts,
            offsets: self.offsets.clone(),
            resolved_addrs: self.resolved_addrs.clone(),
            addr_starts: self.addr_starts.clone(),
        }
    }
}

impl<F> Default for RootProgram<F> {
    fn default() -> Self {
        Self {
            inner: Default::default(),
            total_memory: Default::default(),
            shape: Default::default(),
            event_counts: Default::default(),
            offsets: Default::default(),
            resolved_addrs: Default::default(),
            addr_starts: Default::default(),
        }
    }
}

impl<F: p3_field::PrimeField64> RootProgram<F> {
    /// Construct a new `RootProgram` and pre-compute the offset table.
    pub fn new(
        inner: raw::RawProgram<Instruction<F>>,
        total_memory: usize,
        shape: Option<RecursionShape>,
    ) -> Self {
        let mut program = Self {
            inner,
            total_memory,
            shape,
            event_counts: Default::default(),
            offsets: Default::default(),
            resolved_addrs: Default::default(),
            addr_starts: Default::default(),
        };
        program.analyze_offsets();
        program
    }

    /// Pre-compute event counts and the per-instruction offset table.
    /// Called once at construction time. No cloning of instructions.
    pub fn analyze_offsets(&mut self)
    where
        F: p3_field::PrimeField64 + Send + Sync,
    {
        let mut counters = super::machine::RecursionAirEventCount::default();
        let mut offsets = Vec::new();
        let mut resolved_addrs = Vec::new();
        let mut addr_starts = Vec::new();
        Self::analyze_program_parallel(
            &self.inner, &mut counters, &mut offsets,
            &mut resolved_addrs, &mut addr_starts,
        );
        self.event_counts = counters;
        self.offsets = offsets;
        self.resolved_addrs = resolved_addrs;
        self.addr_starts = addr_starts;
    }

    fn analyze_program_parallel(
        program: &raw::RawProgram<Instruction<F>>,
        counters: &mut super::machine::RecursionAirEventCount,
        offsets: &mut Vec<usize>,
        resolved_addrs: &mut Vec<u32>,
        addr_starts: &mut Vec<u32>,
    ) where
        F: p3_field::PrimeField64 + Send + Sync,
    {
        for block in &program.seq_blocks {
            match block {
                SeqBlock::Basic(basic) => {
                    for instr in &basic.instrs {
                        offsets.push(Self::assign_offset(counters, instr));
                        *counters += instr;
                        addr_starts.push(resolved_addrs.len() as u32);
                        Self::resolve_addrs(instr, resolved_addrs);
                    }
                }
                SeqBlock::Parallel(programs) if programs.len() >= 2 => {
                    use rayon::prelude::*;

                    let n = programs.len();

                    // Process block 0 serially to learn per-block sizes.
                    let counters_before = *counters;
                    let offsets_before = offsets.len();
                    let resolved_before = resolved_addrs.len();
                    Self::analyze_program_serial(
                        &programs[0], counters, offsets, resolved_addrs, addr_starts,
                    );
                    let per_block_instrs = offsets.len() - offsets_before;
                    let per_block_resolved = resolved_addrs.len() - resolved_before;
                    let counter_delta = *counters - counters_before;

                    // Pre-allocate for remaining N-1 blocks.
                    let remaining = n - 1;
                    offsets.reserve(remaining * per_block_instrs);
                    addr_starts.reserve(remaining * per_block_instrs);
                    resolved_addrs.reserve(remaining * per_block_resolved);

                    // Extend vectors to final size (fill in parallel below).
                    let offsets_start = offsets.len();
                    let addr_starts_start = addr_starts.len();
                    let resolved_start = resolved_addrs.len();
                    unsafe {
                        offsets.set_len(offsets_start + remaining * per_block_instrs);
                        addr_starts.set_len(addr_starts_start + remaining * per_block_instrs);
                        resolved_addrs.set_len(resolved_start + remaining * per_block_resolved);
                    }

                    // Process blocks 1..N in parallel.
                    let offsets_ptr = offsets.as_mut_ptr() as usize;
                    let addr_starts_ptr = addr_starts.as_mut_ptr() as usize;
                    let resolved_ptr = resolved_addrs.as_mut_ptr() as usize;

                    programs[1..].par_iter().enumerate().for_each(|(idx, sub)| {
                        let block_idx = idx; // 0-based within remaining
                        let block_counters_base = counters_before + counter_delta * (block_idx + 1);
                        let off_base = offsets_start + block_idx * per_block_instrs;
                        let as_base = addr_starts_start + block_idx * per_block_instrs;
                        let res_base = resolved_start + block_idx * per_block_resolved;

                        let mut local_counters = block_counters_base;
                        let mut instr_idx = 0usize;
                        let mut resolved_idx = 0usize;

                        for sub_block in &sub.seq_blocks {
                            let instrs = match sub_block {
                                SeqBlock::Basic(bb) => &bb.instrs[..],
                                SeqBlock::Parallel(_) => continue,
                            };
                            for instr in instrs {
                                let offset_val = Self::assign_offset(&local_counters, instr);
                                local_counters += instr;
                                let resolved_count_before = resolved_idx;

                                // Write offset
                                unsafe {
                                    let p = (offsets_ptr as *mut usize).add(off_base + instr_idx);
                                    *p = offset_val;
                                }
                                // Write addr_start (base + local resolved index)
                                unsafe {
                                    let p = (addr_starts_ptr as *mut u32).add(as_base + instr_idx);
                                    *p = (res_base + resolved_idx) as u32;
                                }
                                // Write resolved addrs inline
                                unsafe {
                                    let base_p = (resolved_ptr as *mut u32).add(res_base + resolved_idx);
                                    resolved_idx += Self::resolve_addrs_to_ptr(instr, base_p);
                                }
                                let _ = resolved_count_before;
                                instr_idx += 1;
                            }
                        }
                    });

                    // Update counters to account for all N blocks.
                    *counters = counters_before + counter_delta * n;
                }
                SeqBlock::Parallel(programs) => {
                    for sub in programs {
                        Self::analyze_program_serial(
                            sub, counters, offsets, resolved_addrs, addr_starts,
                        );
                    }
                }
            }
        }
    }

    fn analyze_program_serial(
        program: &raw::RawProgram<Instruction<F>>,
        counters: &mut super::machine::RecursionAirEventCount,
        offsets: &mut Vec<usize>,
        resolved_addrs: &mut Vec<u32>,
        addr_starts: &mut Vec<u32>,
    ) where
        F: p3_field::PrimeField64,
    {
        for block in &program.seq_blocks {
            match block {
                SeqBlock::Basic(basic) => {
                    for instr in &basic.instrs {
                        offsets.push(Self::assign_offset(counters, instr));
                        *counters += instr;
                        addr_starts.push(resolved_addrs.len() as u32);
                        Self::resolve_addrs(instr, resolved_addrs);
                    }
                }
                SeqBlock::Parallel(programs) => {
                    for sub in programs {
                        Self::analyze_program_serial(
                            sub, counters, offsets, resolved_addrs, addr_starts,
                        );
                    }
                }
            }
        }
    }

    /// Like `resolve_addrs` but writes to a raw pointer, returns number of u32s written.
    unsafe fn resolve_addrs_to_ptr(instr: &Instruction<F>, out: *mut u32) -> usize
    where
        F: p3_field::PrimeField64,
    {
        use super::instruction::{
            HintBitsInstr, HintAddCurveInstr, PrintInstr,
            HintExt2FeltsInstr, HintInstr,
        };
        let mut idx = 0usize;
        macro_rules! push {
            ($val:expr) => { *out.add(idx) = $val; idx += 1; };
        }
        match instr {
            Instruction::BaseAlu(BaseAluInstr { addrs, .. }) => {
                push!(addrs.in1.as_usize() as u32);
                push!(addrs.in2.as_usize() as u32);
                push!(addrs.out.as_usize() as u32);
            }
            Instruction::ExtAlu(ExtAluInstr { addrs, .. }) => {
                push!(addrs.in1.as_usize() as u32);
                push!(addrs.in2.as_usize() as u32);
                push!(addrs.out.as_usize() as u32);
            }
            Instruction::Mem(MemInstr { addrs: MemIo { inner: addr }, .. }) => {
                push!(addr.as_usize() as u32);
            }
            Instruction::Poseidon2(instr) => {
                let Poseidon2Instr { addrs: Poseidon2Io { input, output }, .. } = instr.as_ref();
                for a in input { push!(a.as_usize() as u32); }
                for a in output { push!(a.as_usize() as u32); }
            }
            Instruction::Poseidon2Skinny(instr) => {
                let Poseidon2SkinnyInstr { addrs: Poseidon2Io { input, output }, .. } = instr.as_ref();
                for a in input { push!(a.as_usize() as u32); }
                for a in output { push!(a.as_usize() as u32); }
            }
            Instruction::Select(SelectInstr { addrs, .. }) => {
                push!(addrs.bit.as_usize() as u32);
                push!(addrs.in1.as_usize() as u32);
                push!(addrs.in2.as_usize() as u32);
                push!(addrs.out1.as_usize() as u32);
                push!(addrs.out2.as_usize() as u32);
            }
            Instruction::HintBits(HintBitsInstr { input_addr, output_addrs_mults, .. }) => {
                push!(input_addr.as_usize() as u32);
                for (a, _) in output_addrs_mults.iter() { push!(a.as_usize() as u32); }
            }
            Instruction::Hint(HintInstr { output_addrs_mults, .. }) => {
                for (a, _) in output_addrs_mults.iter() { push!(a.as_usize() as u32); }
            }
            Instruction::HintExt2Felts(HintExt2FeltsInstr { input_addr, output_addrs_mults, .. }) => {
                push!(input_addr.as_usize() as u32);
                for (a, _) in output_addrs_mults.iter() { push!(a.as_usize() as u32); }
            }
            Instruction::HintAddCurve(instr) => {
                let HintAddCurveInstr {
                    output_x_addrs_mults, output_y_addrs_mults,
                    input1_x_addrs, input1_y_addrs, input2_x_addrs, input2_y_addrs,
                } = instr.as_ref();
                for a in input1_x_addrs.iter() { push!(a.as_usize() as u32); }
                for a in input1_y_addrs.iter() { push!(a.as_usize() as u32); }
                for a in input2_x_addrs.iter() { push!(a.as_usize() as u32); }
                for a in input2_y_addrs.iter() { push!(a.as_usize() as u32); }
                for (a, _) in output_x_addrs_mults.iter() { push!(a.as_usize() as u32); }
                for (a, _) in output_y_addrs_mults.iter() { push!(a.as_usize() as u32); }
            }
            Instruction::CommitPublicValues(instr) => {
                for a in instr.pv_addrs.as_array() { push!(a.as_usize() as u32); }
            }
            Instruction::Print(PrintInstr { addr, .. }) => {
                push!(addr.as_usize() as u32);
            }
            Instruction::PolyEval(PolyEvalInstr {
                addrs: PolyEvalIo { point, coeff, out: o }, ..
            }) => {
                push!(point.as_usize() as u32);
                for a in coeff.iter() { push!(a.as_usize() as u32); }
                push!(o.as_usize() as u32);
            }
            Instruction::ExtExpReverseBits(ExtExpReverseBitsInstr {
                addrs: ExtExpReverseBitsIo { base, exp, prev_acc_vec, acc_vec }, ..
            }) => {
                push!(base.as_usize() as u32);
                for a in exp.iter() { push!(a.as_usize() as u32); }
                for a in prev_acc_vec.iter() { push!(a.as_usize() as u32); }
                for a in acc_vec.iter() { push!(a.as_usize() as u32); }
            }
            Instruction::PrefixSumChecks(ref instr) => {
                let PrefixSumChecksInstr {
                    addrs: PrefixSumChecksIo { x1_vec, x2_vec, prev_acc_vec, acc_vec }, ..
                } = instr.as_ref();
                for a in x1_vec.iter() { push!(a.as_usize() as u32); }
                for a in x2_vec.iter() { push!(a.as_usize() as u32); }
                for a in prev_acc_vec.iter() { push!(a.as_usize() as u32); }
                for a in acc_vec.iter() { push!(a.as_usize() as u32); }
            }
            #[cfg(feature = "debug")]
            Instruction::DebugBacktrace(_) => {}
        }
        idx
    }


    /// Pre-resolve all `Address<F>` fields in an instruction to canonical u32 values.
    /// Called once during `analyze_offsets`; the resulting flat table is consumed at
    /// runtime so that `execute_one` never needs Montgomery reduction.
    fn resolve_addrs(instr: &Instruction<F>, out: &mut Vec<u32>)
    where
        F: p3_field::PrimeField64,
    {
        use super::instruction::{
            HintBitsInstr, HintAddCurveInstr, PrintInstr,
            HintExt2FeltsInstr, HintInstr,
        };
        match instr {
            Instruction::BaseAlu(BaseAluInstr { addrs, .. }) => {
                out.push(addrs.in1.as_usize() as u32);
                out.push(addrs.in2.as_usize() as u32);
                out.push(addrs.out.as_usize() as u32);
            }
            Instruction::ExtAlu(ExtAluInstr { addrs, .. }) => {
                out.push(addrs.in1.as_usize() as u32);
                out.push(addrs.in2.as_usize() as u32);
                out.push(addrs.out.as_usize() as u32);
            }
            Instruction::Mem(MemInstr { addrs: MemIo { inner: addr }, .. }) => {
                out.push(addr.as_usize() as u32);
            }
            Instruction::Poseidon2(instr) => {
                let Poseidon2Instr { addrs: Poseidon2Io { input, output }, .. } = instr.as_ref();
                for a in input { out.push(a.as_usize() as u32); }
                for a in output { out.push(a.as_usize() as u32); }
            }
            Instruction::Poseidon2Skinny(instr) => {
                let Poseidon2SkinnyInstr { addrs: Poseidon2Io { input, output }, .. } = instr.as_ref();
                for a in input { out.push(a.as_usize() as u32); }
                for a in output { out.push(a.as_usize() as u32); }
            }
            Instruction::Select(SelectInstr {
                addrs: SelectIo { bit, out1, out2, in1, in2 }, ..
            }) => {
                out.push(bit.as_usize() as u32);
                out.push(in1.as_usize() as u32);
                out.push(in2.as_usize() as u32);
                out.push(out1.as_usize() as u32);
                out.push(out2.as_usize() as u32);
            }
            Instruction::HintBits(HintBitsInstr { output_addrs_mults, input_addr }) => {
                out.push(input_addr.as_usize() as u32);
                for &(addr, _) in output_addrs_mults.iter() {
                    out.push(addr.as_usize() as u32);
                }
            }
            Instruction::HintAddCurve(instr) => {
                let HintAddCurveInstr {
                    output_x_addrs_mults, output_y_addrs_mults,
                    input1_x_addrs, input1_y_addrs, input2_x_addrs, input2_y_addrs,
                } = instr.as_ref();
                for a in input1_x_addrs {
                    out.push(a.as_usize() as u32);
                }
                for a in input1_y_addrs {
                    out.push(a.as_usize() as u32);
                }
                for a in input2_x_addrs {
                    out.push(a.as_usize() as u32);
                }
                for a in input2_y_addrs {
                    out.push(a.as_usize() as u32);
                }
                for &(addr, _) in output_x_addrs_mults.iter() {
                    out.push(addr.as_usize() as u32);
                }
                for &(addr, _) in output_y_addrs_mults.iter() {
                    out.push(addr.as_usize() as u32);
                }
            }
            Instruction::CommitPublicValues(instr) => {
                for a in instr.pv_addrs.as_array() {
                    out.push(a.as_usize() as u32);
                }
            }
            Instruction::Print(PrintInstr { addr, .. }) => {
                out.push(addr.as_usize() as u32);
            }
            Instruction::HintExt2Felts(HintExt2FeltsInstr { output_addrs_mults, input_addr }) => {
                out.push(input_addr.as_usize() as u32);
                for &(addr, _) in output_addrs_mults.iter() {
                    out.push(addr.as_usize() as u32);
                }
            }
            Instruction::Hint(HintInstr { output_addrs_mults }) => {
                for &(addr, _) in output_addrs_mults.iter() {
                    out.push(addr.as_usize() as u32);
                }
            }
            Instruction::PolyEval(PolyEvalInstr {
                addrs: PolyEvalIo { point, coeff, out: o }, ..
            }) => {
                out.push(point.as_usize() as u32);
                for a in coeff.iter() {
                    out.push(a.as_usize() as u32);
                }
                out.push(o.as_usize() as u32);
            }
            Instruction::ExtExpReverseBits(ExtExpReverseBitsInstr {
                addrs: ExtExpReverseBitsIo { base, exp, prev_acc_vec, acc_vec }, ..
            }) => {
                out.push(base.as_usize() as u32);
                for a in exp.iter() { out.push(a.as_usize() as u32); }
                for a in prev_acc_vec.iter() { out.push(a.as_usize() as u32); }
                for a in acc_vec.iter() { out.push(a.as_usize() as u32); }
            }
            Instruction::PrefixSumChecks(ref instr) => {
                let PrefixSumChecksInstr {
                    addrs: PrefixSumChecksIo { x1_vec, x2_vec, prev_acc_vec, acc_vec }, ..
                } = instr.as_ref();
                for a in x1_vec.iter() { out.push(a.as_usize() as u32); }
                for a in x2_vec.iter() { out.push(a.as_usize() as u32); }
                for a in prev_acc_vec.iter() { out.push(a.as_usize() as u32); }
                for a in acc_vec.iter() { out.push(a.as_usize() as u32); }
            }
            #[cfg(feature = "debug")]
            Instruction::DebugBacktrace(_) => {}
        }
    }

    fn assign_offset(
        counters: &super::machine::RecursionAirEventCount,
        instr: &Instruction<F>,
    ) -> usize {
        match instr {
            Instruction::BaseAlu(_) => counters.base_alu_events,
            Instruction::ExtAlu(_) => counters.ext_alu_events,
            Instruction::Mem(_) => counters.mem_const_events,
            Instruction::Poseidon2(_) => counters.poseidon2_wide_events,
            Instruction::Poseidon2Skinny(_) => counters.poseidon2_skinny_events,
            Instruction::Select(_) => counters.select_events,
            Instruction::CommitPublicValues(_) => counters.commit_pv_hash_events,
            Instruction::HintBits(_)
            | Instruction::Hint(_)
            | Instruction::HintExt2Felts(_)
            | Instruction::HintAddCurve(_) => counters.mem_var_events,
            Instruction::PolyEval(_) => counters.poly_eval_events,
            Instruction::ExtExpReverseBits(_) => counters.ext_exp_reverse_bits_events,
            Instruction::PrefixSumChecks(_) => counters.prefix_sum_checks_events,
            Instruction::Print(_) => 0,
            #[cfg(feature = "debug")]
            Instruction::DebugBacktrace(_) => 0,
        }
    }
}

pub mod raw {
    use std::iter::Flatten;

    use super::*;

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct RawProgram<T> {
        pub seq_blocks: Vec<SeqBlock<T>>,
    }

    // `Default` without bounds on the type parameter.
    impl<T> Default for RawProgram<T> {
        fn default() -> Self {
            Self { seq_blocks: Default::default() }
        }
    }

    impl<T> RawProgram<T> {
        pub fn iter(&self) -> impl Iterator<Item = &'_ T> {
            self.seq_blocks.iter().flatten()
        }
        pub fn iter_mut(&mut self) -> impl Iterator<Item = &'_ mut T> {
            self.seq_blocks.iter_mut().flatten()
        }
    }

    impl<T> IntoIterator for RawProgram<T> {
        type Item = T;

        type IntoIter = Flatten<<Vec<SeqBlock<T>> as IntoIterator>::IntoIter>;

        fn into_iter(self) -> Self::IntoIter {
            self.seq_blocks.into_iter().flatten()
        }
    }

    impl<'a, T> IntoIterator for &'a RawProgram<T> {
        type Item = &'a T;

        type IntoIter = Flatten<<&'a Vec<SeqBlock<T>> as IntoIterator>::IntoIter>;

        fn into_iter(self) -> Self::IntoIter {
            self.seq_blocks.iter().flatten()
        }
    }

    impl<'a, T> IntoIterator for &'a mut RawProgram<T> {
        type Item = &'a mut T;

        type IntoIter = Flatten<<&'a mut Vec<SeqBlock<T>> as IntoIterator>::IntoIter>;

        fn into_iter(self) -> Self::IntoIter {
            self.seq_blocks.iter_mut().flatten()
        }
    }
}

pub mod seq_block {
    use std::iter::Flatten;

    use super::*;

    /// Segments that may be sequentially composed.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub enum SeqBlock<T> {
        /// One basic block.
        Basic(BasicBlock<T>),
        /// Many blocks to be run in parallel.
        Parallel(Vec<RawProgram<T>>),
    }

    impl<T> SeqBlock<T> {
        pub fn iter(&self) -> Iter<'_, T> {
            self.into_iter()
        }

        pub fn iter_mut(&mut self) -> IterMut<'_, T> {
            self.into_iter()
        }
    }

    // Bunch of iterator boilerplate.
    #[derive(Debug)]
    pub enum Iter<'a, T> {
        Basic(<&'a Vec<T> as IntoIterator>::IntoIter),
        Parallel(Box<Flatten<<&'a Vec<RawProgram<T>> as IntoIterator>::IntoIter>>),
    }

    impl<'a, T> Iterator for Iter<'a, T> {
        type Item = &'a T;

        fn next(&mut self) -> Option<Self::Item> {
            match self {
                Iter::Basic(it) => it.next(),
                Iter::Parallel(it) => it.next(),
            }
        }
    }

    impl<'a, T> IntoIterator for &'a SeqBlock<T> {
        type Item = &'a T;

        type IntoIter = Iter<'a, T>;

        fn into_iter(self) -> Self::IntoIter {
            match self {
                SeqBlock::Basic(basic_block) => Iter::Basic(basic_block.instrs.iter()),
                SeqBlock::Parallel(vec) => Iter::Parallel(Box::new(vec.iter().flatten())),
            }
        }
    }

    #[derive(Debug)]
    pub enum IterMut<'a, T> {
        Basic(<&'a mut Vec<T> as IntoIterator>::IntoIter),
        Parallel(Box<Flatten<<&'a mut Vec<RawProgram<T>> as IntoIterator>::IntoIter>>),
    }

    impl<'a, T> Iterator for IterMut<'a, T> {
        type Item = &'a mut T;

        fn next(&mut self) -> Option<Self::Item> {
            match self {
                IterMut::Basic(it) => it.next(),
                IterMut::Parallel(it) => it.next(),
            }
        }
    }

    impl<'a, T> IntoIterator for &'a mut SeqBlock<T> {
        type Item = &'a mut T;

        type IntoIter = IterMut<'a, T>;

        fn into_iter(self) -> Self::IntoIter {
            match self {
                SeqBlock::Basic(basic_block) => IterMut::Basic(basic_block.instrs.iter_mut()),
                SeqBlock::Parallel(vec) => IterMut::Parallel(Box::new(vec.iter_mut().flatten())),
            }
        }
    }

    #[derive(Debug, Clone)]
    pub enum IntoIter<T> {
        Basic(<Vec<T> as IntoIterator>::IntoIter),
        Parallel(Box<Flatten<<Vec<RawProgram<T>> as IntoIterator>::IntoIter>>),
    }

    impl<T> Iterator for IntoIter<T> {
        type Item = T;

        fn next(&mut self) -> Option<Self::Item> {
            match self {
                IntoIter::Basic(it) => it.next(),
                IntoIter::Parallel(it) => it.next(),
            }
        }
    }

    impl<T> IntoIterator for SeqBlock<T> {
        type Item = T;

        type IntoIter = IntoIter<T>;

        fn into_iter(self) -> Self::IntoIter {
            match self {
                SeqBlock::Basic(basic_block) => IntoIter::Basic(basic_block.instrs.into_iter()),
                SeqBlock::Parallel(vec) => IntoIter::Parallel(Box::new(vec.into_iter().flatten())),
            }
        }
    }
}

pub mod basic_block {
    use super::*;

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct BasicBlock<T> {
        pub instrs: Vec<T>,
    }

    // Less restrictive trait bounds.
    impl<T> Default for BasicBlock<T> {
        fn default() -> Self {
            Self { instrs: Default::default() }
        }
    }
}
