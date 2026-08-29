use chips::poseidon2_skinny::WIDTH;
use core::fmt::Debug;
#[cfg(not(feature = "verify"))]
use dt_recursion_core::air::RECURSIVE_PROOF_NUM_PV_ELTS;
use dt_recursion_core::{
    air::{Block, RecursionPublicValues},
    BaseAluInstr, BaseAluOpcode,
};
use dt_stark::septic_curve::SepticCurve;
use instruction::{
    FieldEltType, HintAddCurveInstr, HintBitsInstr, HintExt2FeltsInstr, HintInstr, PrintInstr,
};
use itertools::Itertools;
use p3_field::{AbstractExtensionField, AbstractField, Field, PrimeField64, TwoAdicField};
#[cfg(not(feature = "verify"))]
use std::mem::transmute;
use std::{
    borrow::{Borrow, Cow},
    collections::HashMap,
};
use vec_map::VecMap;

use dt_recursion_core::*;

use crate::prelude::*;

/// Poseidon2 lowering mode for the recursion compiler.
///
/// The DSL layer (`builder.poseidon2_permute_v2`) is intentionally chip-agnostic;
/// the compiler decides which physical chip layout to lower to via this flag.
///
/// - `Wide`   -> `Instruction::Poseidon2`       (one permutation per row; used by
///   `sc_compress_machine` / `sc_wrap_machine`).
/// - `Skinny` -> `Instruction::Poseidon2Skinny` (one round per row; used by `sc_shrink_machine`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Poseidon2Mode {
    #[default]
    Wide,
    Skinny,
}

/// The backend for the circuit compiler.
#[derive(Debug, Clone, Default)]
pub struct AsmCompiler<C: Config> {
    pub next_addr: C::F,
    /// Map the frame pointers of the variables to the "physical" addresses.
    pub virtual_to_physical: VecMap<Address<C::F>>,
    /// Map base or extension field constants to "physical" addresses and mults.
    pub consts: HashMap<Imm<C::F, C::EF>, (Address<C::F>, C::F)>,
    /// Map each "physical" address to its read count.
    pub addr_to_mult: VecMap<C::F>,
    /// Track virtual addresses holding known constant values for constant folding.
    pub const_vaddrs: HashMap<usize, Imm<C::F, C::EF>>,
    /// How to lower `DslIr::CircuitV2Poseidon2PermuteKoalaBear` -- to the wide chip
    /// (default, used by compress / wrap) or to the skinny chip (used by shrink).
    pub poseidon2_mode: Poseidon2Mode,
}

impl<C: Config> AsmCompiler<C> {
    /// Switch this compiler instance to lower Poseidon2 permutations to the **skinny** chip.
    /// The default lowering is wide; call this when building the shrink program so that the
    /// emitted `Instruction::Poseidon2Skinny` matches `sc_shrink_machine`'s registered chip.
    pub fn with_poseidon2_skinny(mut self) -> Self {
        self.poseidon2_mode = Poseidon2Mode::Skinny;
        self
    }
}

impl<C: Config> AsmCompiler<C>
where
    C::F: PrimeField64,
{
    /// evaluate a field polynomial at an extension field element, i.e., y = f(x)
    fn poly_eval(
        &mut self,
        y: Felt<C::F>,           // 输出y
        coeffs: Vec<Felt<C::F>>, // 多项式系数
        x: Felt<C::F>,           // 输入x
    ) -> Instruction<C::F> {
        let num_coeffs = coeffs.len();
        let num_chains = num_coeffs.saturating_sub(1);
        let chain_accum_addrs = (0..num_chains).map(|_| Self::alloc(&mut self.next_addr)).collect();
        Instruction::PolyEval(PolyEvalInstr {
            addrs: PolyEvalIo {
                out: y.write(self),
                point: x.read(self),
                coeff: coeffs.into_iter().map(|r| r.read(self)).collect(),
            },
            mult: C::F::zero(),
            chain_accum_addrs,
        })
    }

    fn ext_exp_reverse_bits(
        &mut self,
        result: Ext<C::F, C::EF>,
        base: Ext<C::F, C::EF>,
        exp: Vec<Felt<C::F>>,
    ) -> Instruction<C::F> {
        let n = exp.len();
        let temp_addrs: Vec<Address<C::F>> =
            (0..n.saturating_sub(1)).map(|_| Self::alloc(&mut self.next_addr)).collect();
        let mut acc_addrs = temp_addrs.clone();
        acc_addrs.push(result.write(self));
        let const_one_addr = self.read_const(Imm::EF(C::EF::one()));
        let mut prev_acc_addrs = vec![const_one_addr];
        prev_acc_addrs.extend(temp_addrs.iter().copied());
        let base_addr = base.read(self);
        for _ in 1..n {
            self.read_addr(base_addr);
        }
        Instruction::ExtExpReverseBits(ExtExpReverseBitsInstr {
            addrs: ExtExpReverseBitsIo {
                base: base_addr,
                exp: exp.into_iter().map(|r| r.read(self)).collect(),
                prev_acc_vec: prev_acc_addrs,
                acc_vec: acc_addrs,
            },
            mult: C::F::zero(),
        })
    }

    fn sumcheck_round(
        &mut self,
        result: Ext<C::F, C::EF>,
        coeffs: Vec<Ext<C::F, C::EF>>,
        challenge: Ext<C::F, C::EF>,
        claim: Ext<C::F, C::EF>,
    ) -> Instruction<C::F> {
        let num_coeffs = coeffs.len();
        let num_chains = num_coeffs.saturating_sub(1);
        let chain_rs_addrs = (0..num_chains).map(|_| Self::alloc(&mut self.next_addr)).collect();
        let chain_ha_addrs = (0..num_chains).map(|_| Self::alloc(&mut self.next_addr)).collect();
        Instruction::SumcheckRound(Box::new(SumcheckRoundInstr {
            addrs: SumcheckRoundIo {
                coeffs: coeffs.into_iter().map(|c| c.read(self)).collect(),
                challenge: challenge.read(self),
                claim: claim.read(self),
                new_claim: result.write(self),
            },
            mult: C::F::zero(),
            chain_rs_addrs,
            chain_ha_addrs,
        }))
    }

    fn prefix_sum_checks(
        &mut self,
        result: Ext<C::F, C::EF>,
        x1_vec: Vec<Ext<C::F, C::EF>>,
        x2_vec: Vec<Ext<C::F, C::EF>>,
    ) -> Instruction<C::F> {
        let n = x1_vec.len();
        let x1_addrs: Vec<_> = x1_vec.into_iter().map(|x| x.read(self)).collect();
        let x2_addrs: Vec<_> = x2_vec.into_iter().map(|x| x.read(self)).collect();

        let temp_addrs: Vec<Address<C::F>> =
            (0..n.saturating_sub(1)).map(|_| Self::alloc(&mut self.next_addr)).collect();

        let mut acc_addrs: Vec<Address<C::F>> = temp_addrs.clone();
        acc_addrs.push(result.write(self));

        let const_one_addr = self.read_const(Imm::EF(C::EF::one()));
        let mut prev_acc_addrs = vec![const_one_addr];
        prev_acc_addrs.extend(temp_addrs.iter().copied());

        Instruction::PrefixSumChecks(Box::new(PrefixSumChecksInstr {
            addrs: PrefixSumChecksIo {
                x1_vec: x1_addrs,
                x2_vec: x2_addrs,
                prev_acc_vec: prev_acc_addrs,
                acc_vec: acc_addrs,
            },
            mult: C::F::zero(),
        }))
    }

    /// Allocate a fresh address. Checks that the address space is not full.
    pub fn alloc(next_addr: &mut C::F) -> Address<C::F> {
        let id = Address(*next_addr);
        *next_addr += C::F::one();
        if next_addr.is_zero() {
            panic!("out of address space");
        }
        id
    }

    /// Map `fp` to its existing address without changing its mult.
    ///
    /// Ensures that `fp` has already been assigned an address.
    pub fn read_ghost_vaddr(&mut self, vaddr: usize) -> Address<C::F> {
        self.read_vaddr_internal(vaddr, false)
    }

    /// Map `fp` to its existing address and increment its mult.
    ///
    /// Ensures that `fp` has already been assigned an address.
    pub fn read_vaddr(&mut self, vaddr: usize) -> Address<C::F> {
        self.read_vaddr_internal(vaddr, true)
    }

    pub fn read_vaddr_internal(&mut self, vaddr: usize, increment_mult: bool) -> Address<C::F> {
        use vec_map::Entry;
        match self.virtual_to_physical.entry(vaddr) {
            Entry::Vacant(_) => panic!("expected entry: virtual_physical[{vaddr:?}]"),
            Entry::Occupied(entry) => {
                if increment_mult {
                    // This is a read, so we increment the mult.
                    match self.addr_to_mult.get_mut(entry.get().as_usize()) {
                        Some(mult) => *mult += C::F::one(),
                        None => panic!("expected entry: virtual_physical[{vaddr:?}]"),
                    }
                }
                *entry.into_mut()
            }
        }
    }

    /// Map `fp` to a fresh address and initialize the mult to 0.
    ///
    /// Ensures that `fp` has not already been written to.
    pub fn write_fp(&mut self, vaddr: usize) -> Address<C::F> {
        use vec_map::Entry;
        match self.virtual_to_physical.entry(vaddr) {
            Entry::Vacant(entry) => {
                let addr = Self::alloc(&mut self.next_addr);
                // This is a write, so we set the mult to zero.
                if let Some(x) = self.addr_to_mult.insert(addr.as_usize(), C::F::zero()) {
                    panic!("unexpected entry in addr_to_mult: {x:?}");
                }
                *entry.insert(addr)
            }
            Entry::Occupied(entry) => {
                panic!("unexpected entry: virtual_to_physical[{:?}] = {:?}", vaddr, entry.get())
            }
        }
    }

    pub fn alias_vaddr(&mut self, dst_vaddr: usize, src_vaddr: usize) {
        use vec_map::Entry;
        let src_addr = *self
            .virtual_to_physical
            .get(src_vaddr)
            .unwrap_or_else(|| panic!("alias source not yet assigned: vaddr={dst_vaddr}"));
        match self.virtual_to_physical.entry(dst_vaddr) {
            Entry::Vacant(entry) => {
                entry.insert(src_addr);
            }
            Entry::Occupied(entry) => {
                panic!(
                    "alias destination already written: vaddr={}, addr={:?}",
                    dst_vaddr,
                    entry.get()
                );
            }
        }
    }

    /// Increment the existing `mult` associated with `addr`.
    ///
    /// Ensures that `addr` has already been assigned a `mult`.
    pub fn read_addr(&mut self, addr: Address<C::F>) -> &mut C::F {
        self.read_addr_internal(addr, true)
    }

    /// Retrieves `mult` associated with `addr`.
    ///
    /// Ensures that `addr` has already been assigned a `mult`.
    pub fn read_ghost_addr(&mut self, addr: Address<C::F>) -> &mut C::F {
        self.read_addr_internal(addr, true)
    }

    fn read_addr_internal(&mut self, addr: Address<C::F>, increment_mult: bool) -> &mut C::F {
        use vec_map::Entry;
        match self.addr_to_mult.entry(addr.as_usize()) {
            Entry::Vacant(_) => panic!("expected entry: addr_to_mult[{:?}]", addr.as_usize()),
            Entry::Occupied(entry) => {
                // This is a read, so we increment the mult.
                let mult = entry.into_mut();
                if increment_mult {
                    *mult += C::F::one();
                }
                mult
            }
        }
    }

    /// Associate a `mult` of zero with `addr`.
    ///
    /// Ensures that `addr` has not already been written to.
    pub fn write_addr(&mut self, addr: Address<C::F>) -> &mut C::F {
        use vec_map::Entry;
        match self.addr_to_mult.entry(addr.as_usize()) {
            Entry::Vacant(entry) => entry.insert(C::F::zero()),
            Entry::Occupied(entry) => {
                panic!("unexpected entry: addr_to_mult[{:?}] = {:?}", addr.as_usize(), entry.get())
            }
        }
    }

    /// Read a constant (a.k.a. immediate).
    ///
    /// Increments the mult, first creating an entry if it does not yet exist.
    pub fn read_const(&mut self, imm: Imm<C::F, C::EF>) -> Address<C::F> {
        self.consts
            .entry(imm)
            .and_modify(|(_, x)| *x += C::F::one())
            .or_insert_with(|| (Self::alloc(&mut self.next_addr), C::F::one()))
            .0
    }

    /// Read a constant (a.k.a. immediate).
    ///
    /// Does not increment the mult. Creates an entry if it does not yet exist.
    pub fn read_ghost_const(&mut self, imm: Imm<C::F, C::EF>) -> Address<C::F> {
        self.consts.entry(imm).or_insert_with(|| (Self::alloc(&mut self.next_addr), C::F::zero())).0
    }

    fn mem_write_const(&mut self, dst: impl Reg<C>, src: Imm<C::F, C::EF>) -> Instruction<C::F> {
        Instruction::Mem(MemInstr {
            addrs: MemIo { inner: dst.write(self) },
            vals: MemIo { inner: src.as_block() },
            mult: C::F::zero(),
            kind: MemAccessKind::Write,
        })
    }

    fn base_alu(
        &mut self,
        opcode: BaseAluOpcode,
        dst: impl Reg<C>,
        lhs: impl Reg<C>,
        rhs: impl Reg<C>,
    ) -> Instruction<C::F> {
        Instruction::BaseAlu(BaseAluInstr {
            opcode,
            mult: C::F::zero(),
            addrs: BaseAluIo { out: dst.write(self), in1: lhs.read(self), in2: rhs.read(self) },
        })
    }

    fn ext_alu(
        &mut self,
        opcode: ExtAluOpcode,
        dst: impl Reg<C>,
        lhs: impl Reg<C>,
        rhs: impl Reg<C>,
    ) -> Instruction<C::F> {
        Instruction::ExtAlu(ExtAluInstr {
            opcode,
            mult: C::F::zero(),
            addrs: ExtAluIo { out: dst.write(self), in1: lhs.read(self), in2: rhs.read(self) },
        })
    }

    fn base_assert_eq(
        &mut self,
        lhs: impl Reg<C>,
        rhs: impl Reg<C>,
        mut f: impl FnMut(Instruction<C::F>),
    ) {
        use BaseAluOpcode::*;
        let [diff, out] = core::array::from_fn(|_| Self::alloc(&mut self.next_addr));
        f(self.base_alu(SubF, diff, lhs, rhs));
        f(self.base_alu(DivF, out, diff, Imm::F(C::F::zero())));
    }

    fn base_assert_ne(
        &mut self,
        lhs: impl Reg<C>,
        rhs: impl Reg<C>,
        mut f: impl FnMut(Instruction<C::F>),
    ) {
        use BaseAluOpcode::*;
        let [diff, out] = core::array::from_fn(|_| Self::alloc(&mut self.next_addr));

        f(self.base_alu(SubF, diff, lhs, rhs));
        f(self.base_alu(DivF, out, Imm::F(C::F::one()), diff));
    }

    fn ext_assert_eq(
        &mut self,
        lhs: impl Reg<C>,
        rhs: impl Reg<C>,
        mut f: impl FnMut(Instruction<C::F>),
    ) {
        use ExtAluOpcode::*;
        let [diff, out] = core::array::from_fn(|_| Self::alloc(&mut self.next_addr));

        f(self.ext_alu(SubE, diff, lhs, rhs));
        f(self.ext_alu(DivE, out, diff, Imm::EF(C::EF::zero())));
    }

    fn ext_assert_ne(
        &mut self,
        lhs: impl Reg<C>,
        rhs: impl Reg<C>,
        mut f: impl FnMut(Instruction<C::F>),
    ) {
        use ExtAluOpcode::*;
        let [diff, out] = core::array::from_fn(|_| Self::alloc(&mut self.next_addr));

        f(self.ext_alu(SubE, diff, lhs, rhs));
        f(self.ext_alu(DivE, out, Imm::EF(C::EF::one()), diff));
    }

    #[inline(always)]
    fn poseidon2_permute(
        &mut self,
        dst: [impl Reg<C>; WIDTH],
        src: [impl Reg<C>; WIDTH],
    ) -> Instruction<C::F> {
        Instruction::Poseidon2(Box::new(Poseidon2Instr {
            addrs: Poseidon2Io {
                input: src.map(|r| r.read(self)),
                output: dst.map(|r| r.write(self)),
            },
            mults: [C::F::zero(); WIDTH],
        }))
    }

    /// Same as `poseidon2_permute` but emits a `Poseidon2Skinny` instruction carrying
    /// `SKINNY_NUM_SCRATCH` (= `ROWS_PER_PERMUTE - 1`) groups of fresh scratch addresses used
    /// to chain adjacent rows of the skinny chip's per-permutation block via memory lookup.
    ///
    /// The scratch array length is fixed by the active cargo feature:
    ///   * BabyBear  : one round per row, 21 rows -> 20 scratch groups.
    ///   * KoalaBear : 9-row layout (4 ext + 1 internal-rounds + 4 ext) -> 8 scratch groups.
    ///
    /// Selected by `AsmCompiler::with_poseidon2_skinny()` for the shrink stage, which
    /// registers `Poseidon2SkinnyChip` / `Poseidon2SkinnyKbChip` in `sc_shrink_machine`.
    #[inline(always)]
    fn poseidon2_permute_skinny(
        &mut self,
        dst: [impl Reg<C>; WIDTH],
        src: [impl Reg<C>; WIDTH],
    ) -> Instruction<C::F> {
        let scratch_addrs: [[Address<C::F>; WIDTH]; SKINNY_NUM_SCRATCH] =
            core::array::from_fn(|_| core::array::from_fn(|_| Self::alloc(&mut self.next_addr)));
        Instruction::Poseidon2Skinny(Box::new(Poseidon2SkinnyInstr {
            addrs: Poseidon2Io {
                input: src.map(|r| r.read(self)),
                output: dst.map(|r| r.write(self)),
            },
            mults: [C::F::zero(); WIDTH],
            scratch_addrs,
        }))
    }

    #[inline(always)]
    fn select(
        &mut self,
        bit: impl Reg<C>,
        dst1: impl Reg<C>,
        dst2: impl Reg<C>,
        lhs: impl Reg<C>,
        rhs: impl Reg<C>,
    ) -> Instruction<C::F> {
        Instruction::Select(SelectInstr {
            addrs: SelectIo {
                bit: bit.read(self),
                out1: dst1.write(self),
                out2: dst2.write(self),
                in1: lhs.read(self),
                in2: rhs.read(self),
            },
            mult1: C::F::zero(),
            mult2: C::F::zero(),
        })
    }

    fn hint_bit_decomposition(
        &mut self,
        value: impl Reg<C>,
        output: impl IntoIterator<Item = impl Reg<C>>,
    ) -> Instruction<C::F> {
        Instruction::HintBits(HintBitsInstr {
            output_addrs_mults: output.into_iter().map(|r| (r.write(self), C::F::zero())).collect(),
            input_addr: value.read_ghost(self),
        })
    }

    fn add_curve(
        &mut self,
        output: SepticCurve<Felt<C::F>>,
        input1: SepticCurve<Felt<C::F>>,
        input2: SepticCurve<Felt<C::F>>,
    ) -> Instruction<C::F> {
        Instruction::HintAddCurve(Box::new(HintAddCurveInstr {
            output_x_addrs_mults: output
                .x
                .0
                .into_iter()
                .map(|r| (r.write(self), C::F::zero()))
                .collect(),
            output_y_addrs_mults: output
                .y
                .0
                .into_iter()
                .map(|r| (r.write(self), C::F::zero()))
                .collect(),
            input1_x_addrs: input1.x.0.into_iter().map(|value| value.read_ghost(self)).collect(),
            input1_y_addrs: input1.y.0.into_iter().map(|value| value.read_ghost(self)).collect(),
            input2_x_addrs: input2.x.0.into_iter().map(|value| value.read_ghost(self)).collect(),
            input2_y_addrs: input2.y.0.into_iter().map(|value| value.read_ghost(self)).collect(),
        }))
    }

    fn commit_public_values(
        &mut self,
        public_values: &RecursionPublicValues<Felt<C::F>>,
    ) -> Instruction<C::F> {
        public_values.digest.iter().for_each(|x| {
            let _ = x.read(self);
        });
        // let pv_addrs = public_values.as_array().map(|pv| pv.read_ghost(self));
        #[cfg(not(feature = "verify"))]
        let pv_addrs =
            unsafe {
                transmute::<
                    RecursionPublicValues<Felt<C::F>>,
                    [Felt<C::F>; RECURSIVE_PROOF_NUM_PV_ELTS],
                >(*public_values)
            }
            .map(|pv| pv.read_ghost(self));
        #[cfg(feature = "verify")]
        let pv_addrs = public_values.as_array().map(|pv| pv.read_ghost(self));
        let public_values_a: &RecursionPublicValues<Address<C::F>> = pv_addrs.as_slice().borrow();
        Instruction::CommitPublicValues(Box::new(CommitPublicValuesInstr {
            pv_addrs: *public_values_a,
        }))
    }

    fn print_f(&mut self, addr: impl Reg<C>) -> Instruction<C::F> {
        Instruction::Print(PrintInstr {
            field_elt_type: FieldEltType::Base,
            addr: addr.read_ghost(self),
        })
    }

    fn print_e(&mut self, addr: impl Reg<C>) -> Instruction<C::F> {
        Instruction::Print(PrintInstr {
            field_elt_type: FieldEltType::Extension,
            addr: addr.read_ghost(self),
        })
    }

    fn ext2felts(&mut self, felts: [impl Reg<C>; D], ext: impl Reg<C>) -> Instruction<C::F> {
        Instruction::HintExt2Felts(HintExt2FeltsInstr {
            output_addrs_mults: felts.map(|r| (r.write(self), C::F::zero())),
            input_addr: ext.read_ghost(self),
        })
    }

    fn hint(&mut self, output: impl Reg<C>, len: usize) -> Instruction<C::F> {
        let zero = C::F::zero();
        Instruction::Hint(HintInstr {
            output_addrs_mults: output
                .write_many(self, len)
                .into_iter()
                .map(|a| (a, zero))
                .collect(),
        })
    }
}

impl<C> AsmCompiler<C>
where
    C: Config<N = <C as Config>::F> + Debug,
    C::F: PrimeField64 + TwoAdicField,
{
    /// Emit a constant-folded result as a memory write, recording the vaddr as constant.
    fn emit_const_fold(&mut self, dst_vaddr: usize, val: Imm<C::F, C::EF>) -> Instruction<C::F> {
        let addr = self.write_fp(dst_vaddr);
        self.const_vaddrs.insert(dst_vaddr, val);
        Instruction::Mem(MemInstr {
            addrs: MemIo { inner: addr },
            vals: MemIo { inner: val.as_block() },
            mult: C::F::zero(),
            kind: MemAccessKind::Write,
        })
    }

    fn felt_const_val(&self, idx: u32) -> Option<C::F> {
        match self.const_vaddrs.get(&(idx as usize))? {
            Imm::F(v) => Some(*v),
            _ => None,
        }
    }

    fn ext_const_val(&self, idx: u32) -> Option<C::EF> {
        match self.const_vaddrs.get(&(idx as usize))? {
            Imm::EF(v) => Some(*v),
            _ => None,
        }
    }

    /// Propagate constant info when aliasing dst to src.
    fn propagate_const_alias(&mut self, dst_idx: u32, src_idx: u32) {
        if let Some(c) = self.const_vaddrs.get(&(src_idx as usize)).cloned() {
            self.const_vaddrs.insert(dst_idx as usize, c);
        }
    }

    /// Compiles one instruction, passing one or more instructions to `consumer`.
    ///
    /// We do not simply return a `Vec` for performance reasons --- results would be immediately fed
    /// to `flat_map`, so we employ fusion/deforestation to eliminate intermediate data structures.
    pub fn compile_one(
        &mut self,
        ir_instr: DslIr<C>,
        mut consumer: impl FnMut(Result<Instruction<C::F>, CompileOneErr<C>>),
    ) {
        // For readability. Avoids polluting outer scope.
        use BaseAluOpcode::*;
        use ExtAluOpcode::*;

        let mut f = |instr| consumer(Ok(instr));
        match ir_instr {
            DslIr::ImmV(dst, src) => {
                self.const_vaddrs.insert(dst.idx as usize, Imm::F(src));
                f(self.mem_write_const(dst, Imm::F(src)))
            }
            DslIr::ImmF(dst, src) => {
                self.const_vaddrs.insert(dst.idx as usize, Imm::F(src));
                f(self.mem_write_const(dst, Imm::F(src)))
            }
            DslIr::ImmE(dst, src) => {
                self.const_vaddrs.insert(dst.idx as usize, Imm::EF(src));
                f(self.mem_write_const(dst, Imm::EF(src)))
            }

            DslIr::AddV(dst, lhs, rhs) => {
                if let (Some(a), Some(b)) =
                    (self.felt_const_val(lhs.idx), self.felt_const_val(rhs.idx))
                {
                    f(self.emit_const_fold(dst.idx as usize, Imm::F(a + b)))
                } else {
                    f(self.base_alu(AddF, dst, lhs, rhs))
                }
            }
            DslIr::AddVI(dst, lhs, rhs) => {
                if rhs.is_zero() {
                    self.alias_vaddr(dst.idx as usize, lhs.idx as usize);
                    self.propagate_const_alias(dst.idx, lhs.idx);
                } else if let Some(a) = self.felt_const_val(lhs.idx) {
                    f(self.emit_const_fold(dst.idx as usize, Imm::F(a + rhs)))
                } else {
                    f(self.base_alu(AddF, dst, lhs, Imm::F(rhs)))
                }
            }
            DslIr::AddF(dst, lhs, rhs) => {
                if let (Some(a), Some(b)) =
                    (self.felt_const_val(lhs.idx), self.felt_const_val(rhs.idx))
                {
                    f(self.emit_const_fold(dst.idx as usize, Imm::F(a + b)))
                } else {
                    f(self.base_alu(AddF, dst, lhs, rhs))
                }
            }
            DslIr::AddFI(dst, lhs, rhs) => {
                if rhs.is_zero() {
                    self.alias_vaddr(dst.idx as usize, lhs.idx as usize);
                    self.propagate_const_alias(dst.idx, lhs.idx);
                } else if let Some(a) = self.felt_const_val(lhs.idx) {
                    f(self.emit_const_fold(dst.idx as usize, Imm::F(a + rhs)))
                } else {
                    f(self.base_alu(AddF, dst, lhs, Imm::F(rhs)))
                }
            }
            DslIr::AddE(dst, lhs, rhs) => {
                if let (Some(a), Some(b)) =
                    (self.ext_const_val(lhs.idx), self.ext_const_val(rhs.idx))
                {
                    f(self.emit_const_fold(dst.idx as usize, Imm::EF(a + b)))
                } else {
                    f(self.ext_alu(AddE, dst, lhs, rhs))
                }
            }
            DslIr::AddEI(dst, lhs, rhs) => {
                if rhs.is_zero() {
                    self.alias_vaddr(dst.idx as usize, lhs.idx as usize);
                    self.propagate_const_alias(dst.idx, lhs.idx);
                } else if let Some(a) = self.ext_const_val(lhs.idx) {
                    f(self.emit_const_fold(dst.idx as usize, Imm::EF(a + rhs)))
                } else {
                    f(self.ext_alu(AddE, dst, lhs, Imm::EF(rhs)))
                }
            }
            DslIr::AddEF(dst, lhs, rhs) => {
                if let (Some(a), Some(b)) =
                    (self.ext_const_val(lhs.idx), self.felt_const_val(rhs.idx))
                {
                    f(self.emit_const_fold(dst.idx as usize, Imm::EF(a + C::EF::from_base(b))))
                } else {
                    f(self.ext_alu(AddE, dst, lhs, rhs))
                }
            }
            DslIr::AddEFI(dst, lhs, rhs) => {
                if rhs.is_zero() {
                    self.alias_vaddr(dst.idx as usize, lhs.idx as usize);
                    self.propagate_const_alias(dst.idx, lhs.idx);
                } else if let Some(a) = self.ext_const_val(lhs.idx) {
                    f(self.emit_const_fold(dst.idx as usize, Imm::EF(a + C::EF::from_base(rhs))))
                } else {
                    f(self.ext_alu(AddE, dst, lhs, Imm::F(rhs)))
                }
            }
            DslIr::AddEFFI(dst, lhs, rhs) => {
                if let Some(a) = self.ext_const_val(lhs.idx) {
                    f(self.emit_const_fold(dst.idx as usize, Imm::EF(a + rhs)))
                } else {
                    f(self.ext_alu(AddE, dst, lhs, Imm::EF(rhs)))
                }
            }

            DslIr::SubV(dst, lhs, rhs) => {
                if let (Some(a), Some(b)) =
                    (self.felt_const_val(lhs.idx), self.felt_const_val(rhs.idx))
                {
                    f(self.emit_const_fold(dst.idx as usize, Imm::F(a - b)))
                } else {
                    f(self.base_alu(SubF, dst, lhs, rhs))
                }
            }
            DslIr::SubVI(dst, lhs, rhs) => {
                if rhs.is_zero() {
                    self.alias_vaddr(dst.idx as usize, lhs.idx as usize);
                    self.propagate_const_alias(dst.idx, lhs.idx);
                } else if let Some(a) = self.felt_const_val(lhs.idx) {
                    f(self.emit_const_fold(dst.idx as usize, Imm::F(a - rhs)))
                } else {
                    f(self.base_alu(SubF, dst, lhs, Imm::F(rhs)))
                }
            }
            DslIr::SubVIN(dst, lhs, rhs) => {
                if let Some(b) = self.felt_const_val(rhs.idx) {
                    f(self.emit_const_fold(dst.idx as usize, Imm::F(lhs - b)))
                } else {
                    f(self.base_alu(SubF, dst, Imm::F(lhs), rhs))
                }
            }
            DslIr::SubF(dst, lhs, rhs) => {
                if let (Some(a), Some(b)) =
                    (self.felt_const_val(lhs.idx), self.felt_const_val(rhs.idx))
                {
                    f(self.emit_const_fold(dst.idx as usize, Imm::F(a - b)))
                } else {
                    f(self.base_alu(SubF, dst, lhs, rhs))
                }
            }
            DslIr::SubFI(dst, lhs, rhs) => {
                if rhs.is_zero() {
                    self.alias_vaddr(dst.idx as usize, lhs.idx as usize);
                    self.propagate_const_alias(dst.idx, lhs.idx);
                } else if let Some(a) = self.felt_const_val(lhs.idx) {
                    f(self.emit_const_fold(dst.idx as usize, Imm::F(a - rhs)))
                } else {
                    f(self.base_alu(SubF, dst, lhs, Imm::F(rhs)))
                }
            }
            DslIr::SubFIN(dst, lhs, rhs) => {
                if let Some(b) = self.felt_const_val(rhs.idx) {
                    f(self.emit_const_fold(dst.idx as usize, Imm::F(lhs - b)))
                } else {
                    f(self.base_alu(SubF, dst, Imm::F(lhs), rhs))
                }
            }
            DslIr::SubE(dst, lhs, rhs) => {
                if let (Some(a), Some(b)) =
                    (self.ext_const_val(lhs.idx), self.ext_const_val(rhs.idx))
                {
                    f(self.emit_const_fold(dst.idx as usize, Imm::EF(a - b)))
                } else {
                    f(self.ext_alu(SubE, dst, lhs, rhs))
                }
            }
            DslIr::SubEI(dst, lhs, rhs) => {
                if rhs.is_zero() {
                    self.alias_vaddr(dst.idx as usize, lhs.idx as usize);
                    self.propagate_const_alias(dst.idx, lhs.idx);
                } else if let Some(a) = self.ext_const_val(lhs.idx) {
                    f(self.emit_const_fold(dst.idx as usize, Imm::EF(a - rhs)))
                } else {
                    f(self.ext_alu(SubE, dst, lhs, Imm::EF(rhs)))
                }
            }
            DslIr::SubEIN(dst, lhs, rhs) => {
                if let Some(b) = self.ext_const_val(rhs.idx) {
                    f(self.emit_const_fold(dst.idx as usize, Imm::EF(lhs - b)))
                } else {
                    f(self.ext_alu(SubE, dst, Imm::EF(lhs), rhs))
                }
            }
            DslIr::SubEFI(dst, lhs, rhs) => {
                if rhs.is_zero() {
                    self.alias_vaddr(dst.idx as usize, lhs.idx as usize);
                    self.propagate_const_alias(dst.idx, lhs.idx);
                } else if let Some(a) = self.ext_const_val(lhs.idx) {
                    f(self.emit_const_fold(dst.idx as usize, Imm::EF(a - C::EF::from_base(rhs))))
                } else {
                    f(self.ext_alu(SubE, dst, lhs, Imm::F(rhs)))
                }
            }
            DslIr::SubEF(dst, lhs, rhs) => {
                if let (Some(a), Some(b)) =
                    (self.ext_const_val(lhs.idx), self.felt_const_val(rhs.idx))
                {
                    f(self.emit_const_fold(dst.idx as usize, Imm::EF(a - C::EF::from_base(b))))
                } else {
                    f(self.ext_alu(SubE, dst, lhs, rhs))
                }
            }

            DslIr::MulV(dst, lhs, rhs) => {
                if let (Some(a), Some(b)) =
                    (self.felt_const_val(lhs.idx), self.felt_const_val(rhs.idx))
                {
                    f(self.emit_const_fold(dst.idx as usize, Imm::F(a * b)))
                } else {
                    f(self.base_alu(MulF, dst, lhs, rhs))
                }
            }
            DslIr::MulVI(dst, lhs, rhs) => {
                if rhs.is_one() {
                    self.alias_vaddr(dst.idx as usize, lhs.idx as usize);
                    self.propagate_const_alias(dst.idx, lhs.idx);
                } else if rhs.is_zero() {
                    f(self.emit_const_fold(dst.idx as usize, Imm::F(C::F::zero())))
                } else if let Some(a) = self.felt_const_val(lhs.idx) {
                    f(self.emit_const_fold(dst.idx as usize, Imm::F(a * rhs)))
                } else {
                    f(self.base_alu(MulF, dst, lhs, Imm::F(rhs)))
                }
            }
            DslIr::MulF(dst, lhs, rhs) => {
                if let (Some(a), Some(b)) =
                    (self.felt_const_val(lhs.idx), self.felt_const_val(rhs.idx))
                {
                    f(self.emit_const_fold(dst.idx as usize, Imm::F(a * b)))
                } else {
                    f(self.base_alu(MulF, dst, lhs, rhs))
                }
            }
            DslIr::MulFI(dst, lhs, rhs) => {
                if rhs.is_one() {
                    self.alias_vaddr(dst.idx as usize, lhs.idx as usize);
                    self.propagate_const_alias(dst.idx, lhs.idx);
                } else if rhs.is_zero() {
                    f(self.emit_const_fold(dst.idx as usize, Imm::F(C::F::zero())))
                } else if let Some(a) = self.felt_const_val(lhs.idx) {
                    f(self.emit_const_fold(dst.idx as usize, Imm::F(a * rhs)))
                } else {
                    f(self.base_alu(MulF, dst, lhs, Imm::F(rhs)))
                }
            }
            DslIr::MulE(dst, lhs, rhs) => {
                if let (Some(a), Some(b)) =
                    (self.ext_const_val(lhs.idx), self.ext_const_val(rhs.idx))
                {
                    f(self.emit_const_fold(dst.idx as usize, Imm::EF(a * b)))
                } else {
                    f(self.ext_alu(MulE, dst, lhs, rhs))
                }
            }
            DslIr::MulEI(dst, lhs, rhs) => {
                if rhs.is_one() {
                    self.alias_vaddr(dst.idx as usize, lhs.idx as usize);
                    self.propagate_const_alias(dst.idx, lhs.idx);
                } else if rhs.is_zero() {
                    f(self.emit_const_fold(dst.idx as usize, Imm::EF(C::EF::zero())))
                } else if let Some(a) = self.ext_const_val(lhs.idx) {
                    f(self.emit_const_fold(dst.idx as usize, Imm::EF(a * rhs)))
                } else {
                    f(self.ext_alu(MulE, dst, lhs, Imm::EF(rhs)))
                }
            }
            DslIr::MulEFI(dst, lhs, rhs) => {
                if rhs.is_one() {
                    self.alias_vaddr(dst.idx as usize, lhs.idx as usize);
                    self.propagate_const_alias(dst.idx, lhs.idx);
                } else if rhs.is_zero() {
                    f(self.emit_const_fold(dst.idx as usize, Imm::EF(C::EF::zero())))
                } else if let Some(a) = self.ext_const_val(lhs.idx) {
                    f(self.emit_const_fold(dst.idx as usize, Imm::EF(a * C::EF::from_base(rhs))))
                } else {
                    f(self.ext_alu(MulE, dst, lhs, Imm::F(rhs)))
                }
            }
            DslIr::MulEF(dst, lhs, rhs) => {
                if let (Some(a), Some(b)) =
                    (self.ext_const_val(lhs.idx), self.felt_const_val(rhs.idx))
                {
                    f(self.emit_const_fold(dst.idx as usize, Imm::EF(a * C::EF::from_base(b))))
                } else {
                    f(self.ext_alu(MulE, dst, lhs, rhs))
                }
            }

            DslIr::DivF(dst, lhs, rhs) => f(self.base_alu(DivF, dst, lhs, rhs)),
            DslIr::DivFI(dst, lhs, rhs) => {
                if rhs.is_one() {
                    self.alias_vaddr(dst.idx as usize, lhs.idx as usize);
                    self.propagate_const_alias(dst.idx, lhs.idx);
                } else {
                    f(self.base_alu(DivF, dst, lhs, Imm::F(rhs)))
                }
            }
            DslIr::DivFIN(dst, lhs, rhs) => f(self.base_alu(DivF, dst, Imm::F(lhs), rhs)),
            DslIr::DivE(dst, lhs, rhs) => f(self.ext_alu(DivE, dst, lhs, rhs)),
            DslIr::DivEI(dst, lhs, rhs) => {
                if rhs.is_one() {
                    self.alias_vaddr(dst.idx as usize, lhs.idx as usize);
                    self.propagate_const_alias(dst.idx, lhs.idx);
                } else {
                    f(self.ext_alu(DivE, dst, lhs, Imm::EF(rhs)))
                }
            }
            DslIr::DivEIN(dst, lhs, rhs) => f(self.ext_alu(DivE, dst, Imm::EF(lhs), rhs)),
            DslIr::DivEFI(dst, lhs, rhs) => {
                if rhs.is_one() {
                    self.alias_vaddr(dst.idx as usize, lhs.idx as usize);
                    self.propagate_const_alias(dst.idx, lhs.idx);
                } else {
                    f(self.ext_alu(DivE, dst, lhs, Imm::F(rhs)))
                }
            }
            DslIr::DivEFIN(dst, lhs, rhs) => f(self.ext_alu(DivE, dst, Imm::F(lhs), rhs)),
            DslIr::DivEF(dst, lhs, rhs) => f(self.ext_alu(DivE, dst, lhs, rhs)),

            DslIr::NegV(dst, src) => {
                if let Some(v) = self.felt_const_val(src.idx) {
                    f(self.emit_const_fold(dst.idx as usize, Imm::F(-v)))
                } else {
                    f(self.base_alu(SubF, dst, Imm::F(C::F::zero()), src))
                }
            }
            DslIr::NegF(dst, src) => {
                if let Some(v) = self.felt_const_val(src.idx) {
                    f(self.emit_const_fold(dst.idx as usize, Imm::F(-v)))
                } else {
                    f(self.base_alu(SubF, dst, Imm::F(C::F::zero()), src))
                }
            }
            DslIr::NegE(dst, src) => {
                if let Some(v) = self.ext_const_val(src.idx) {
                    f(self.emit_const_fold(dst.idx as usize, Imm::EF(-v)))
                } else {
                    f(self.ext_alu(SubE, dst, Imm::EF(C::EF::zero()), src))
                }
            }
            DslIr::InvV(dst, src) => f(self.base_alu(DivF, dst, Imm::F(C::F::one()), src)),
            DslIr::InvF(dst, src) => f(self.base_alu(DivF, dst, Imm::F(C::F::one()), src)),
            DslIr::InvE(dst, src) => f(self.ext_alu(DivE, dst, Imm::F(C::F::one()), src)),

            DslIr::Select(bit, dst1, dst2, lhs, rhs) => f(self.select(bit, dst1, dst2, lhs, rhs)),

            // // evaluate an extension field polynomial at an extension field element, i.e., y =
            // f(x) DslIr::EvalE(y, coeffs, x) => f(self.poly_eval(y, coeffs, x)),
            DslIr::AssertEqV(lhs, rhs) => self.base_assert_eq(lhs, rhs, f),
            DslIr::AssertEqF(lhs, rhs) => self.base_assert_eq(lhs, rhs, f),
            DslIr::AssertEqE(lhs, rhs) => self.ext_assert_eq(lhs, rhs, f),
            DslIr::AssertEqVI(lhs, rhs) => self.base_assert_eq(lhs, Imm::F(rhs), f),
            DslIr::AssertEqFI(lhs, rhs) => self.base_assert_eq(lhs, Imm::F(rhs), f),
            DslIr::AssertEqEI(lhs, rhs) => self.ext_assert_eq(lhs, Imm::EF(rhs), f),

            DslIr::AssertNeV(lhs, rhs) => self.base_assert_ne(lhs, rhs, f),
            DslIr::AssertNeF(lhs, rhs) => self.base_assert_ne(lhs, rhs, f),
            DslIr::AssertNeE(lhs, rhs) => self.ext_assert_ne(lhs, rhs, f),
            DslIr::AssertNeVI(lhs, rhs) => self.base_assert_ne(lhs, Imm::F(rhs), f),
            DslIr::AssertNeFI(lhs, rhs) => self.base_assert_ne(lhs, Imm::F(rhs), f),
            DslIr::AssertNeEI(lhs, rhs) => self.ext_assert_ne(lhs, Imm::EF(rhs), f),
            DslIr::CircuitV2Poseidon2PermuteKoalaBear(data) => {
                // KoalaBear lowering: chip selection is controlled by `self.poseidon2_mode`.
                // Default (Wide) matches `sc_compress_machine` / `sc_wrap_machine`; Skinny
                // is enabled via `AsmCompiler::with_poseidon2_skinny()` for `sc_shrink_machine`.
                match self.poseidon2_mode {
                    Poseidon2Mode::Wide => f(self.poseidon2_permute(data.0, data.1)),
                    Poseidon2Mode::Skinny => f(self.poseidon2_permute_skinny(data.0, data.1)),
                }
            }
            DslIr::CircuitV2Poseidon2PermuteBabyBear(data) => {
                // BabyBear lowering: same generic Poseidon2 / Poseidon2Skinny instructions
                // as the KoalaBear path; chip selection is controlled by `self.poseidon2_mode`.
                match self.poseidon2_mode {
                    Poseidon2Mode::Wide => f(self.poseidon2_permute(data.0, data.1)),
                    Poseidon2Mode::Skinny => f(self.poseidon2_permute_skinny(data.0, data.1)),
                }
            }
            DslIr::CircuitV2HintBitsF(output, value) => {
                f(self.hint_bit_decomposition(value, output))
            }
            DslIr::CircuitV2CommitPublicValues(public_values) => {
                f(self.commit_public_values(&public_values))
            }
            DslIr::CircuitV2HintAddCurve(data) => f(self.add_curve(data.0, data.1, data.2)),

            // evaluate a field polynomial at an extension field element, i.e., y = f(x)
            DslIr::CircuitV2PolyEval(y, coeffs, x) => f(self.poly_eval(y, coeffs, x)),
            DslIr::CircuitV2ExtExpReverseBits(dst, base, exp) => {
                f(self.ext_exp_reverse_bits(dst, base, exp))
            }

            DslIr::CircuitV2SumcheckRound(dst, coeffs, challenge, claim) => {
                f(self.sumcheck_round(dst, coeffs, challenge, claim))
            }

            DslIr::CircuitV2PrefixSumChecks(dst, x1_vec, x2_vec) => {
                f(self.prefix_sum_checks(dst, x1_vec, x2_vec))
            }

            DslIr::Parallel(_) => {
                unreachable!("parallel case should have been handled by compile_raw_program")
            }

            DslIr::PrintV(dst) => f(self.print_f(dst)),
            DslIr::PrintF(dst) => f(self.print_f(dst)),
            DslIr::PrintE(dst) => f(self.print_e(dst)),
            #[cfg(feature = "debug")]
            DslIr::DebugBacktrace(trace) => f(Instruction::DebugBacktrace(trace)),
            DslIr::CircuitV2HintFelts(output, len) => f(self.hint(output, len)),
            DslIr::CircuitV2HintExts(output, len) => f(self.hint(output, len)),
            DslIr::CircuitExt2Felt(felts, ext) => f(self.ext2felts(felts, ext)),
            DslIr::CycleTrackerV2Enter(name) => {
                consumer(Err(CompileOneErr::CycleTrackerEnter(name)))
            }
            DslIr::CycleTrackerV2Exit => consumer(Err(CompileOneErr::CycleTrackerExit)),
            DslIr::ReduceE(_) => {}
            instr => consumer(Err(CompileOneErr::Unsupported(instr))),
        }
    }

    /// A raw program (algebraic data type of instructions), not yet backfilled.
    fn compile_raw_program(
        &mut self,
        block: DslIrBlock<C>,
        instrs_prefix: Vec<SeqBlock<Instruction<C::F>>>,
    ) -> RawProgram<Instruction<C::F>> {
        // Consider refactoring the builder to use an AST instead of a list of operations.
        // Possible to remove address translation at this step.
        let mut seq_blocks = instrs_prefix;
        let mut maybe_bb: Option<BasicBlock<Instruction<C::F>>> = None;

        for op in block.ops {
            match op {
                DslIr::Parallel(par_blocks) => {
                    seq_blocks.extend(maybe_bb.take().map(SeqBlock::Basic));
                    seq_blocks.push(SeqBlock::Parallel(
                        par_blocks
                            .into_iter()
                            .map(|b| self.compile_raw_program(b, vec![]))
                            .collect(),
                    ))
                }
                op => {
                    let bb = maybe_bb.get_or_insert_with(Default::default);
                    self.compile_one(op, |item| match item {
                        Ok(instr) => bb.instrs.push(instr),
                        Err(
                            CompileOneErr::CycleTrackerEnter(_) | CompileOneErr::CycleTrackerExit,
                        ) => (),
                        Err(CompileOneErr::Unsupported(instr)) => {
                            panic!("unsupported instruction: {instr:?}")
                        }
                    });
                }
            }
        }

        seq_blocks.extend(maybe_bb.map(SeqBlock::Basic));

        RawProgram { seq_blocks }
    }

    fn backfill_all<'a>(
        &mut self,
        instrs: impl Iterator<Item = &'a mut Instruction<<C as Config>::F>>,
    ) {
        let mut backfill = |(mult, addr): (&mut C::F, &Address<C::F>)| {
            *mult = self.addr_to_mult.remove(addr.as_usize()).unwrap()
        };

        for asm_instr in instrs {
            // Exhaustive match for refactoring purposes.
            match asm_instr {
                Instruction::BaseAlu(BaseAluInstr {
                    mult,
                    addrs: BaseAluIo { out: ref addr, .. },
                    ..
                }) => backfill((mult, addr)),
                Instruction::ExtAlu(ExtAluInstr {
                    mult,
                    addrs: ExtAluIo { out: ref addr, .. },
                    ..
                }) => backfill((mult, addr)),
                Instruction::Mem(MemInstr {
                    addrs: MemIo { inner: ref addr },
                    mult,
                    kind: MemAccessKind::Write,
                    ..
                }) => backfill((mult, addr)),
                Instruction::Poseidon2(instr) => {
                    let Poseidon2WideInstr { addrs: Poseidon2Io { output: ref addrs, .. }, mults } =
                        instr.as_mut();
                    mults.iter_mut().zip(addrs).for_each(&mut backfill);
                }
                Instruction::Poseidon2Skinny(instr) => {
                    let Poseidon2SkinnyInstr {
                        addrs: Poseidon2Io { output: ref addrs, .. },
                        mults,
                        scratch_addrs: _,
                    } = instr.as_mut();
                    mults.iter_mut().zip(addrs).for_each(&mut backfill);
                }
                Instruction::Select(SelectInstr {
                    addrs: SelectIo { out1: ref addr1, out2: ref addr2, .. },
                    mult1,
                    mult2,
                }) => {
                    backfill((mult1, addr1));
                    backfill((mult2, addr2));
                }
                Instruction::HintBits(HintBitsInstr { output_addrs_mults, .. }) |
                Instruction::Hint(HintInstr { output_addrs_mults, .. }) => {
                    output_addrs_mults.iter_mut().for_each(|(addr, mult)| backfill((mult, addr)));
                }
                Instruction::HintExt2Felts(HintExt2FeltsInstr { output_addrs_mults, .. }) => {
                    output_addrs_mults.iter_mut().for_each(|(addr, mult)| backfill((mult, addr)));
                }
                Instruction::HintAddCurve(instr) => {
                    let HintAddCurveInstr { output_x_addrs_mults, output_y_addrs_mults, .. } =
                        instr.as_mut();
                    output_x_addrs_mults.iter_mut().for_each(|(addr, mult)| backfill((mult, addr)));
                    output_y_addrs_mults.iter_mut().for_each(|(addr, mult)| backfill((mult, addr)));
                }
                // Instructions that do not write to memory.
                Instruction::Mem(MemInstr { kind: MemAccessKind::Read, .. }) |
                Instruction::CommitPublicValues(_) |
                Instruction::Print(_) => (),
                #[cfg(feature = "debug")]
                Instruction::DebugBacktrace(_) => (),

                Instruction::PolyEval(PolyEvalInstr {
                    addrs: PolyEvalIo { out: ref addr, .. },
                    mult,
                    ..
                }) => backfill((mult, addr)),
                Instruction::ExtExpReverseBits(ref mut instr) => {
                    backfill((&mut instr.mult, instr.addrs.acc_vec.last().unwrap()));
                }
                Instruction::SumcheckRound(ref mut instr) => {
                    backfill((&mut instr.mult, &instr.addrs.new_claim));
                }
                Instruction::PrefixSumChecks(ref mut instr) => {
                    backfill((&mut instr.mult, instr.addrs.acc_vec.last().unwrap()));
                }
            }
        }

        debug_assert!(self.addr_to_mult.is_empty());
    }

    /// Compile a `DslIrProgram` that is definitionally assumed to be well-formed.
    ///
    /// Returns a well-formed program.
    pub fn compile(&mut self, program: DslIrProgram<C>) -> RecursionProgram<C::F> {
        // SAFETY: The compiler produces well-formed programs given a well-formed DSL input.
        // This is also a cryptographic requirement.
        unsafe { RecursionProgram::new_unchecked(self.compile_inner(program.into_inner())) }
    }

    /// Compile a root `DslIrBlock` that has not necessarily been validated.
    ///
    /// Returns a program that may be ill-formed.
    pub fn compile_inner(&mut self, root_block: DslIrBlock<C>) -> RootProgram<C::F> {
        // Prefix an empty basic block to be later filled in by constants.
        let mut program = tracing::debug_span!("compile raw program").in_scope(|| {
            self.compile_raw_program(root_block, vec![SeqBlock::Basic(BasicBlock::default())])
        });
        let total_memory = self.next_addr.as_canonical_u64() as usize;
        tracing::debug_span!("backfill mult").in_scope(|| self.backfill_all(program.iter_mut()));

        // NOTE: DCE (dead code elimination) was removed because it causes a cumulative sums
        // imbalance in the permutation argument. When DCE removes a dead instruction, the
        // instruction's input reads have already been counted in the writer's multiplicity
        // (via backfill), but the removed instruction's receive interactions no longer exist
        // to cancel them out. Keeping dead instructions (with mult=0 output sends) preserves
        // the balance: their input receives still match the writer's send multiplicities.

        // Put in the constants.
        tracing::debug_span!("prepend constants").in_scope(|| {
            let Some(SeqBlock::Basic(BasicBlock { instrs: instrs_consts })) =
                program.seq_blocks.first_mut()
            else {
                unreachable!()
            };
            instrs_consts.extend(
                self.consts
                    .drain()
                    .filter(|(_, (_, mult))| !mult.is_zero())
                    .sorted_by_key(|x| x.1 .0 .0)
                    .map(|(imm, (addr, mult))| {
                        Instruction::Mem(MemInstr {
                            addrs: MemIo { inner: addr },
                            vals: MemIo { inner: imm.as_block() },
                            mult,
                            kind: MemAccessKind::Write,
                        })
                    }),
            );
        });

        RootProgram { inner: program, total_memory, shape: None }
    }
}

#[derive(Debug, Clone)]
pub enum CompileOneErr<C: Config> {
    Unsupported(DslIr<C>),
    CycleTrackerEnter(Cow<'static, str>),
    CycleTrackerExit,
}

/// Immediate (i.e. constant) field element.
///
/// Required to distinguish a base and extension field element at the type level,
/// since the IR's instructions do not provide this information.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Imm<F, EF> {
    /// Element of the base field `F`.
    F(F),
    /// Element of the extension field `EF`.
    EF(EF),
}

impl<F, EF> Imm<F, EF>
where
    F: AbstractField + Copy,
    EF: AbstractExtensionField<F>,
{
    // Get a `Block` of memory representing this immediate.
    pub fn as_block(&self) -> Block<F> {
        match self {
            Imm::F(f) => Block::from(*f),
            Imm::EF(ef) => ef.as_base_slice().into(),
        }
    }
}

/// Utility functions for various register types.
trait Reg<C: Config> {
    /// Mark the register as to be read from, returning the "physical" address.
    fn read(&self, compiler: &mut AsmCompiler<C>) -> Address<C::F>;

    /// Get the "physical" address of the register, assigning a new address if necessary.
    fn read_ghost(&self, compiler: &mut AsmCompiler<C>) -> Address<C::F>;

    /// Mark the register as to be written to, returning the "physical" address.
    fn write(&self, compiler: &mut AsmCompiler<C>) -> Address<C::F>;

    fn write_many(&self, compiler: &mut AsmCompiler<C>, len: usize) -> Vec<Address<C::F>>;
}

macro_rules! impl_reg_borrowed {
    ($a:ty) => {
        impl<C, T> Reg<C> for $a
        where
            C: Config,
            T: Reg<C> + ?Sized,
        {
            fn read(&self, compiler: &mut AsmCompiler<C>) -> Address<C::F> {
                (**self).read(compiler)
            }

            fn read_ghost(&self, compiler: &mut AsmCompiler<C>) -> Address<C::F> {
                (**self).read_ghost(compiler)
            }

            fn write(&self, compiler: &mut AsmCompiler<C>) -> Address<C::F> {
                (**self).write(compiler)
            }

            fn write_many(&self, compiler: &mut AsmCompiler<C>, len: usize) -> Vec<Address<C::F>> {
                (**self).write_many(compiler, len)
            }
        }
    };
}

// Allow for more flexibility in arguments.
impl_reg_borrowed!(&T);
impl_reg_borrowed!(&mut T);
impl_reg_borrowed!(Box<T>);

macro_rules! impl_reg_vaddr {
    ($a:ty) => {
        impl<C: Config<F: PrimeField64>> Reg<C> for $a {
            fn read(&self, compiler: &mut AsmCompiler<C>) -> Address<C::F> {
                compiler.read_vaddr(self.idx as usize)
            }
            fn read_ghost(&self, compiler: &mut AsmCompiler<C>) -> Address<C::F> {
                compiler.read_ghost_vaddr(self.idx as usize)
            }
            fn write(&self, compiler: &mut AsmCompiler<C>) -> Address<C::F> {
                compiler.write_fp(self.idx as usize)
            }

            fn write_many(&self, compiler: &mut AsmCompiler<C>, len: usize) -> Vec<Address<C::F>> {
                (0..len).map(|i| compiler.write_fp((self.idx + i as u32) as usize)).collect()
            }
        }
    };
}

// These three types wrap a `u32` but they don't share a trait.
impl_reg_vaddr!(Var<C::F>);
impl_reg_vaddr!(Felt<C::F>);
impl_reg_vaddr!(Ext<C::F, C::EF>);

impl<C: Config<F: PrimeField64>> Reg<C> for Imm<C::F, C::EF> {
    fn read(&self, compiler: &mut AsmCompiler<C>) -> Address<C::F> {
        compiler.read_const(*self)
    }

    fn read_ghost(&self, compiler: &mut AsmCompiler<C>) -> Address<C::F> {
        compiler.read_ghost_const(*self)
    }

    fn write(&self, _compiler: &mut AsmCompiler<C>) -> Address<C::F> {
        panic!("cannot write to immediate in register: {self:?}")
    }

    fn write_many(&self, _compiler: &mut AsmCompiler<C>, _len: usize) -> Vec<Address<C::F>> {
        panic!("cannot write to immediate in register: {self:?}")
    }
}

impl<C: Config<F: PrimeField64>> Reg<C> for Address<C::F> {
    fn read(&self, compiler: &mut AsmCompiler<C>) -> Address<C::F> {
        compiler.read_addr(*self);
        *self
    }

    fn read_ghost(&self, compiler: &mut AsmCompiler<C>) -> Address<C::F> {
        compiler.read_ghost_addr(*self);
        *self
    }

    fn write(&self, compiler: &mut AsmCompiler<C>) -> Address<C::F> {
        compiler.write_addr(*self);
        *self
    }

    fn write_many(&self, _compiler: &mut AsmCompiler<C>, _len: usize) -> Vec<Address<C::F>> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::print_stdout)]

    use std::{collections::VecDeque, io::BufRead, iter::zip, sync::Arc};

    use p3_baby_bear::DiffusionMatrixBabyBear;
    use p3_field::{Field, PrimeField32};
    use p3_symmetric::{CryptographicHasher, Permutation};
    use rand::{rngs::StdRng, Rng, SeedableRng};

    use dt_core_machine::utils::{run_test_machine, setup_logger};
    use dt_recursion_core::{machine::RecursionAir, Runtime};
    use dt_stark::{
        baby_bear_poseidon2::BabyBearPoseidon2, inner_perm, BabyBearPoseidon2Inner, InnerChallenge,
        InnerHash, InnerVal, StarkGenericConfig,
    };

    use crate::circuit::{AsmBuilder, AsmConfig, CircuitV2Builder};

    use super::*;

    type SC = BabyBearPoseidon2;
    type F = InnerVal;
    type EF = InnerChallenge;
    fn test_block(block: DslIrBlock<AsmConfig<F, EF>>) {
        test_block_with_runner(block, |program| {
            let mut runtime = Runtime::<F, EF, DiffusionMatrixBabyBear>::new(
                program,
                BabyBearPoseidon2Inner::new().perm,
            );
            runtime.run().unwrap();
            runtime.record
        });
    }

    fn test_block_with_runner(
        block: DslIrBlock<AsmConfig<F, EF>>,
        run: impl FnOnce(Arc<RecursionProgram<F>>) -> ExecutionRecord<F>,
    ) {
        let mut compiler = super::AsmCompiler::<AsmConfig<F, EF>>::default();
        let program = Arc::new(compiler.compile_inner(block).validate().unwrap());
        let record = run(program.clone());

        // Run with the poseidon2 wide chip.
        let wide_machine =
            RecursionAir::<_, 3>::machine_wide_with_all_chips(BabyBearPoseidon2::default());
        let (pk, vk) = wide_machine.setup(&program);
        let result = run_test_machine(vec![record.clone()], wide_machine, pk, vk);
        if let Err(e) = result {
            panic!("Verification failed: {:?}", e);
        }

        // Run with the poseidon2 skinny chip.
        let skinny_machine = RecursionAir::<_, 9>::machine_skinny_with_all_chips(
            BabyBearPoseidon2::ultra_compressed(),
        );
        let (pk, vk) = skinny_machine.setup(&program);
        let result = run_test_machine(vec![record.clone()], skinny_machine, pk, vk);
        if let Err(e) = result {
            panic!("Verification failed: {:?}", e);
        }
    }

    #[test]
    fn test_poseidon2() {
        setup_logger();

        let mut builder = AsmBuilder::<F, EF>::default();
        let mut rng = StdRng::seed_from_u64(0xCAFEDA7E)
            .sample_iter::<[F; WIDTH], _>(rand::distributions::Standard);
        for _ in 0..100 {
            let input_1: [F; WIDTH] = rng.next().unwrap();
            let output_1 = inner_perm().permute(input_1);

            let input_1_felts = input_1.map(|x| builder.eval(x));
            let output_1_felts = builder.poseidon2_permute_v2(input_1_felts);
            let expected: [Felt<_>; WIDTH] = output_1.map(|x| builder.eval(x));
            for (lhs, rhs) in output_1_felts.into_iter().zip(expected) {
                builder.assert_felt_eq(lhs, rhs);
            }
        }

        test_block(builder.into_root_block());
    }

    #[test]
    fn test_poseidon2_hash() {
        let perm = inner_perm();
        let hasher = InnerHash::new(perm.clone());

        let input: [F; 26] = [
            F::from_canonical_u32(0),
            F::from_canonical_u32(1),
            F::from_canonical_u32(2),
            F::from_canonical_u32(2),
            F::from_canonical_u32(2),
            F::from_canonical_u32(2),
            F::from_canonical_u32(2),
            F::from_canonical_u32(2),
            F::from_canonical_u32(2),
            F::from_canonical_u32(2),
            F::from_canonical_u32(2),
            F::from_canonical_u32(2),
            F::from_canonical_u32(2),
            F::from_canonical_u32(2),
            F::from_canonical_u32(2),
            F::from_canonical_u32(3),
            F::from_canonical_u32(3),
            F::from_canonical_u32(3),
            F::from_canonical_u32(3),
            F::from_canonical_u32(3),
            F::from_canonical_u32(3),
            F::from_canonical_u32(3),
            F::from_canonical_u32(3),
            F::from_canonical_u32(3),
            F::from_canonical_u32(3),
            F::from_canonical_u32(3),
        ];
        let expected = hasher.hash_iter(input);
        println!("{:?}", expected);

        let mut builder = AsmBuilder::<F, EF>::default();
        let input_felts: [Felt<_>; 26] = input.map(|x| builder.eval(x));
        let result = builder.poseidon2_hash_v2(&input_felts);

        for (actual_f, expected_f) in zip(result, expected) {
            builder.assert_felt_eq(actual_f, expected_f);
        }
    }

    #[test]
    fn test_exp_reverse_bits() {
        setup_logger();

        let mut builder = AsmBuilder::<F, EF>::default();
        let mut rng =
            StdRng::seed_from_u64(0xEC0BEEF).sample_iter::<F, _>(rand::distributions::Standard);
        for _ in 0..100 {
            let power_f = rng.next().unwrap();
            let power = power_f.as_canonical_u32();
            let power_bits = (0..NUM_BITS).map(|i| (power >> i) & 1).collect::<Vec<_>>();

            let input_felt = builder.eval(power_f);
            let power_bits_felt = builder.num2bits_v2_f(input_felt, NUM_BITS);

            let base = rng.next().unwrap();
            let base_felt = builder.eval(base);
            let result_felt = builder.exp_reverse_bits_v2(base_felt, power_bits_felt);

            let expected = power_bits
                .into_iter()
                .rev()
                .zip(std::iter::successors(Some(base), |x| Some(x.square())))
                .map(|(bit, base_pow)| match bit {
                    0 => F::one(),
                    1 => base_pow,
                    _ => panic!("not a bit: {bit}"),
                })
                .product::<F>();
            let expected_felt: Felt<_> = builder.eval(expected);
            builder.assert_felt_eq(result_felt, expected_felt);
        }
        test_block(builder.into_root_block());
    }

    #[test]
    fn test_hint_bit_decomposition() {
        setup_logger();

        let mut builder = AsmBuilder::<F, EF>::default();
        let mut rng =
            StdRng::seed_from_u64(0xC0FFEE7AB1E).sample_iter::<F, _>(rand::distributions::Standard);
        for _ in 0..100 {
            let input_f = rng.next().unwrap();
            let input = input_f.as_canonical_u32();
            let output = (0..NUM_BITS).map(|i| (input >> i) & 1).collect::<Vec<_>>();

            let input_felt = builder.eval(input_f);
            let output_felts = builder.num2bits_v2_f(input_felt, NUM_BITS);
            let expected: Vec<Felt<_>> =
                output.into_iter().map(|x| builder.eval(F::from_canonical_u32(x))).collect();
            for (lhs, rhs) in output_felts.into_iter().zip(expected) {
                builder.assert_felt_eq(lhs, rhs);
            }
        }
        test_block(builder.into_root_block());
    }

    #[test]
    fn test_print_and_cycle_tracker() {
        const ITERS: usize = 5;

        setup_logger();

        let mut builder = AsmBuilder::<F, EF>::default();

        let input_fs = StdRng::seed_from_u64(0xC0FFEE7AB1E)
            .sample_iter::<F, _>(rand::distributions::Standard)
            .take(ITERS)
            .collect::<Vec<_>>();

        let input_efs = StdRng::seed_from_u64(0x7EA7AB1E)
            .sample_iter::<[F; 4], _>(rand::distributions::Standard)
            .take(ITERS)
            .collect::<Vec<_>>();

        let mut buf = VecDeque::<u8>::new();

        builder.cycle_tracker_v2_enter("printing felts");
        for (i, &input_f) in input_fs.iter().enumerate() {
            builder.cycle_tracker_v2_enter(format!("printing felt {i}"));
            let input_felt = builder.eval(input_f);
            builder.print_f(input_felt);
            builder.cycle_tracker_v2_exit();
        }
        builder.cycle_tracker_v2_exit();

        builder.cycle_tracker_v2_enter("printing exts");
        for (i, input_block) in input_efs.iter().enumerate() {
            builder.cycle_tracker_v2_enter(format!("printing ext {i}"));
            let input_ext = builder.eval(EF::from_base_slice(input_block).cons());
            builder.print_e(input_ext);
            builder.cycle_tracker_v2_exit();
        }
        builder.cycle_tracker_v2_exit();

        test_block_with_runner(builder.into_root_block(), |program| {
            let mut runtime = Runtime::<F, EF, DiffusionMatrixBabyBear>::new(
                program,
                BabyBearPoseidon2Inner::new().perm,
            );
            runtime.debug_stdout = Box::new(&mut buf);
            runtime.run().unwrap();
            runtime.record
        });

        let input_str_fs = input_fs.into_iter().map(|elt| format!("{}", elt));
        let input_str_efs = input_efs.into_iter().map(|elt| format!("{:?}", elt));
        let input_strs = input_str_fs.chain(input_str_efs);

        for (input_str, line) in zip(input_strs, buf.lines()) {
            let line = line.unwrap();
            assert!(line.contains(&input_str));
        }
    }

    #[test]
    fn test_ext2felts() {
        setup_logger();

        let mut builder = AsmBuilder::<F, EF>::default();
        let mut rng =
            StdRng::seed_from_u64(0x3264).sample_iter::<[F; 4], _>(rand::distributions::Standard);
        let mut random_ext = move || EF::from_base_slice(&rng.next().unwrap());
        for _ in 0..100 {
            let input = random_ext();
            let output: &[F] = input.as_base_slice();

            let input_ext = builder.eval(input.cons());
            let output_felts = builder.ext2felt_v2(input_ext);
            let expected: Vec<Felt<_>> = output.iter().map(|&x| builder.eval(x)).collect();
            for (lhs, rhs) in output_felts.into_iter().zip(expected) {
                builder.assert_felt_eq(lhs, rhs);
            }
        }
        test_block(builder.into_root_block());
    }

    macro_rules! test_assert_fixture {
        ($assert_felt:ident, $assert_ext:ident, $should_offset:literal) => {
            {
                use std::convert::identity;
                let mut builder = AsmBuilder::<F, EF>::default();
                test_assert_fixture!(builder, identity, F, Felt<_>, 0xDEADBEEF, $assert_felt, $should_offset);
                test_assert_fixture!(builder, EF::cons, EF, Ext<_, _>, 0xABADCAFE, $assert_ext, $should_offset);
                test_block(builder.into_root_block());
            }
        };
        ($builder:ident, $wrap:path, $t:ty, $u:ty, $seed:expr, $assert:ident, $should_offset:expr) => {
            {
                let mut elts = StdRng::seed_from_u64($seed)
                    .sample_iter::<$t, _>(rand::distributions::Standard);
                for _ in 0..100 {
                    let a = elts.next().unwrap();
                    let b = elts.next().unwrap();
                    let c = a + b;
                    let ar: $u = $builder.eval($wrap(a));
                    let br: $u = $builder.eval($wrap(b));
                    let cr: $u = $builder.eval(ar + br);
                    let cm = if $should_offset {
                        c + elts.find(|x| !x.is_zero()).unwrap()
                    } else {
                        c
                    };
                    $builder.$assert(cr, $wrap(cm));
                }
            }
        };
    }

    #[test]
    fn test_assert_eq_noop() {
        test_assert_fixture!(assert_felt_eq, assert_ext_eq, false);
    }

    #[test]
    #[should_panic]
    fn test_assert_eq_panics() {
        test_assert_fixture!(assert_felt_eq, assert_ext_eq, true);
    }

    #[test]
    fn test_assert_ne_noop() {
        test_assert_fixture!(assert_felt_ne, assert_ext_ne, true);
    }

    #[test]
    #[should_panic]
    fn test_assert_ne_panics() {
        test_assert_fixture!(assert_felt_ne, assert_ext_ne, false);
    }
}
