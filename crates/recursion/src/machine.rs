use std::ops::{Add, AddAssign};

use dt_stark::{
    air::{InteractionScope, MachineAir},
    shape::OrderedShape,
    sumcheck::config::SCStarkGenericConfig,
    Chip, SCStarkMachine, StarkGenericConfig, StarkMachine, PROOF_MAX_NUM_PVS,
};
use hashbrown::HashMap;
use p3_field::Field;
use p3_field::{
    extension::{BinomialExtensionField, BinomiallyExtendable},
    PrimeField32,
};

use crate::chips::ext_exp_reverse_bits::ExtExpReverseBitsChip;
use crate::chips::poly::PolyEvalChip;
#[cfg(feature = "koalabear")]
use crate::chips::poseidon2_skinny_kb::Poseidon2SkinnyKbChip;
#[cfg(feature = "koalabear")]
use crate::chips::poseidon2_wide_kb::Poseidon2WideKbChip;
use crate::chips::prefix_sum_checks::PrefixSumChecksChip;
use crate::{
    chips::{
        alu_base::{BaseAluChip, NUM_BASE_ALU_ENTRIES_PER_ROW, NUM_BASE_ALU_SHRINK_ENTRIES_PER_ROW},
        alu_ext::{ExtAluChip, NUM_EXT_ALU_ENTRIES_PER_ROW, NUM_EXT_ALU_SHRINK_ENTRIES_PER_ROW},
        mem::{
            constant::NUM_CONST_MEM_ENTRIES_PER_ROW, variable::NUM_VAR_MEM_ENTRIES_PER_ROW,
            MemoryConstChip, MemoryVarChip,
        },
        poseidon2_skinny::Poseidon2SkinnyChip,
        poseidon2_wide::Poseidon2WideChip,
        public_values::{PublicValuesChip, PUB_VALUES_LOG_HEIGHT},
        select::SelectChip,
    },
    instruction::{HintBitsInstr, HintExt2FeltsInstr, HintInstr},
    shape::RecursionShape,
    ExtExpReverseBitsInstr, Instruction, RecursionProgram, D,
};

#[derive(dt_derive::MachineAir)]
#[dt_core_path = "crate"]
#[execution_record_path = "crate::ExecutionRecord<F>"]
#[program_path = "crate::RecursionProgram<F>"]
#[builder_path = "crate::builder::DTRecursionAirBuilder<F = F>"]
#[eval_trait_bound = "AB::Var: 'static"]
pub enum RecursionAir<F: Field, const DEGREE: usize> {
    MemoryConst(MemoryConstChip<F>),
    MemoryVar(MemoryVarChip<F>),
    BaseAlu(BaseAluChip),
    BaseAluShrink(BaseAluChip<NUM_BASE_ALU_SHRINK_ENTRIES_PER_ROW>),
    ExtAlu(ExtAluChip),
    ExtAluShrink(ExtAluChip<NUM_EXT_ALU_SHRINK_ENTRIES_PER_ROW>),
    Poseidon2Skinny(Poseidon2SkinnyChip<DEGREE>),
    #[cfg(feature = "koalabear")]
    Poseidon2SkinnyKb(Poseidon2SkinnyKbChip<DEGREE>),
    Poseidon2Wide(Poseidon2WideChip<DEGREE>),
    #[cfg(feature = "koalabear")]
    Poseidon2WideKb(Poseidon2WideKbChip<DEGREE>),
    Select(SelectChip),
    PublicValues(PublicValuesChip),
    PolyEval(PolyEvalChip<DEGREE>),
    ExtExpReverseBits(ExtExpReverseBitsChip<DEGREE>),
    PrefixSumChecks(PrefixSumChecksChip),
}

#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct RecursionAirEventCount {
    pub mem_const_events: usize,
    pub mem_var_events: usize,
    pub base_alu_events: usize,
    pub ext_alu_events: usize,
    pub poseidon2_wide_events: usize,
    /// Number of permutations bound for the skinny Poseidon2 chip
    /// (BabyBear `Poseidon2SkinnyChip` or KoalaBear `Poseidon2SkinnyKbChip`).
    pub poseidon2_skinny_events: usize,
    pub select_events: usize,
    pub poly_eval_events: usize,
    pub ext_exp_reverse_bits_events: usize,
    pub prefix_sum_checks_events: usize,
    pub commit_pv_hash_events: usize,
}

impl<F: PrimeField32 + BinomiallyExtendable<D>, const DEGREE: usize> RecursionAir<F, DEGREE> {
    pub fn poseidon2_wide_chip() -> Self {
        #[cfg(feature = "babybear")]
        {
            RecursionAir::Poseidon2Wide(Poseidon2WideChip::<DEGREE>)
        }
        #[cfg(feature = "koalabear")]
        {
            RecursionAir::Poseidon2WideKb(Poseidon2WideKbChip::<DEGREE>)
        }
    }

    pub fn poseidon2_skinny_chip() -> Self {
        #[cfg(feature = "babybear")]
        {
            RecursionAir::Poseidon2Skinny(Poseidon2SkinnyChip::<DEGREE>::default())
        }
        #[cfg(feature = "koalabear")]
        {
            RecursionAir::Poseidon2SkinnyKb(Poseidon2SkinnyKbChip::<DEGREE>::default())
        }
    }

    /// Number of rows produced per Poseidon2 permutation by the active skinny chip.
    ///
    /// Differs by field:
    ///   * BabyBear  : 21 rows (one round per row, 8 + 13).
    ///   * KoalaBear : 9 rows (4 external + 1 folded-internal + 4 external).
    ///
    /// Selected via cargo feature gating to match the active chip's
    /// `trace::ROWS_PER_PERMUTE` constant.
    #[inline]
    pub fn skinny_rows_per_permute() -> usize {
        #[cfg(feature = "babybear")]
        {
            crate::chips::poseidon2_skinny::trace::ROWS_PER_PERMUTE
        }
        #[cfg(feature = "koalabear")]
        {
            crate::chips::poseidon2_skinny_kb::trace::ROWS_PER_PERMUTE
        }
    }

    /// Get a machine with all chips, except the dummy chip.
    pub fn machine_wide_with_all_chips<SC: StarkGenericConfig<Val = F>>(
        config: SC,
    ) -> StarkMachine<SC, Self> {
        let chips = vec![
            RecursionAir::MemoryConst(MemoryConstChip::default()),
            RecursionAir::MemoryVar(MemoryVarChip::default()),
            RecursionAir::BaseAlu(BaseAluChip),
            RecursionAir::ExtAlu(ExtAluChip),
            Self::poseidon2_wide_chip(),
            RecursionAir::Select(SelectChip),
            RecursionAir::PublicValues(PublicValuesChip),
        ]
        .into_iter()
        .map(Chip::new)
        .collect::<Vec<_>>();
        StarkMachine::new(config, chips, PROOF_MAX_NUM_PVS, false)
    }

    /// Get a machine with all chips, except the dummy chip.
    pub fn machine_skinny_with_all_chips<SC: StarkGenericConfig<Val = F>>(
        config: SC,
    ) -> StarkMachine<SC, Self> {
        let chips = vec![
            RecursionAir::MemoryConst(MemoryConstChip::default()),
            RecursionAir::MemoryVar(MemoryVarChip::default()),
            RecursionAir::BaseAluShrink(BaseAluChip),
            RecursionAir::ExtAluShrink(ExtAluChip),
            Self::poseidon2_skinny_chip(),
            RecursionAir::Select(SelectChip),
            RecursionAir::PublicValues(PublicValuesChip),
        ]
        .into_iter()
        .map(Chip::new)
        .collect::<Vec<_>>();
        StarkMachine::new(config, chips, PROOF_MAX_NUM_PVS, false)
    }

    pub fn shrink_shape() -> RecursionShape {
        // Fixed shrink-stage shape: each chip's log_height is the minimum power-of-two
        // that accommodates worst-case event counts for the default compress FRI config.
        // Bottleneck: Poseidon2SkinnyKb at 2^19.
        // Capacity is verified at runtime via the `shrink height overflow` panic.
        let shape = HashMap::from(
            [
                (Self::MemoryVar(MemoryVarChip::default()), 18),
                (Self::Select(SelectChip), 18),
                (Self::MemoryConst(MemoryConstChip::default()), 17),
                (Self::BaseAluShrink(BaseAluChip), 17),
                (Self::ExtAluShrink(ExtAluChip), 18),
                (Self::poseidon2_skinny_chip(), 19),
                (Self::PublicValues(PublicValuesChip), PUB_VALUES_LOG_HEIGHT),
            ]
            .map(|(chip, log_height)| (chip.name(), log_height)),
        );
        RecursionShape { inner: shape }
    }

    pub fn heights(program: &RecursionProgram<F>) -> Vec<(String, usize)> {
        let heights = program
            .inner
            .iter()
            .fold(RecursionAirEventCount::default(), |heights, instruction| heights + instruction);

        [
            (
                Self::MemoryConst(MemoryConstChip::default()),
                heights.mem_const_events.div_ceil(NUM_CONST_MEM_ENTRIES_PER_ROW),
            ),
            (
                Self::MemoryVar(MemoryVarChip::default()),
                heights.mem_var_events.div_ceil(NUM_VAR_MEM_ENTRIES_PER_ROW),
            ),
            (
                Self::BaseAlu(BaseAluChip),
                heights.base_alu_events.div_ceil(NUM_BASE_ALU_ENTRIES_PER_ROW),
            ),
            (
                Self::ExtAlu(ExtAluChip),
                heights.ext_alu_events.div_ceil(NUM_EXT_ALU_ENTRIES_PER_ROW),
            ),
            // Compress path routes Poseidon2 through the wide chip (the shrink stage still
            // uses the skinny chip via `sc_shrink_machine`). The wide chip fits one full
            // permutation per row, so trace height equals the number of permutation events
            // directly.
            (Self::poseidon2_wide_chip(), heights.poseidon2_wide_events),
            (Self::Select(SelectChip), heights.select_events),
            (Self::PublicValues(PublicValuesChip), PUB_VALUES_LOG_HEIGHT),
            (Self::PrefixSumChecks(PrefixSumChecksChip), heights.prefix_sum_checks_events),
            (
                Self::ExtExpReverseBits(ExtExpReverseBitsChip::<DEGREE>),
                heights.ext_exp_reverse_bits_events,
            ),
        ]
        .map(|(chip, log_height)| (chip.name(), log_height))
        .to_vec()
    }

    /// Compute actual trace heights for the **shrink** stage.
    ///
    /// Unlike [`heights`], this method uses the skinny Poseidon2 chip
    /// (height = events × `skinny_rows_per_permute`) instead of the wide
    /// chip, matching what `shrink_shape()` and `sc_shrink_machine` expect.
    pub fn shrink_heights(program: &RecursionProgram<F>) -> Vec<(String, usize)> {
        let counts = program
            .inner
            .iter()
            .fold(RecursionAirEventCount::default(), |acc, instr| acc + instr);

        #[cfg(feature = "koalabear")]
        let skinny_rows = counts.poseidon2_skinny_events * Self::skinny_rows_per_permute();

        let mut result = vec![
            (
                Self::MemoryConst(MemoryConstChip::default()).name(),
                counts.mem_const_events.div_ceil(NUM_CONST_MEM_ENTRIES_PER_ROW),
            ),
            (
                Self::MemoryVar(MemoryVarChip::default()).name(),
                counts.mem_var_events.div_ceil(NUM_VAR_MEM_ENTRIES_PER_ROW),
            ),
            (
                Self::BaseAluShrink(BaseAluChip).name(),
                counts.base_alu_events.div_ceil(NUM_BASE_ALU_SHRINK_ENTRIES_PER_ROW),
            ),
            (
                Self::ExtAluShrink(ExtAluChip).name(),
                counts.ext_alu_events.div_ceil(NUM_EXT_ALU_SHRINK_ENTRIES_PER_ROW),
            ),
        ];

        #[cfg(feature = "koalabear")]
        result.push((Self::poseidon2_skinny_chip().name(), skinny_rows));

        result.extend(vec![
            (Self::Select(SelectChip).name(), counts.select_events),
            (Self::PublicValues(PublicValuesChip).name(), 1 << PUB_VALUES_LOG_HEIGHT),
        ]);

        result
    }
}
macro_rules! poseidon2_wide_chip_for {
    ($degree:expr) => {{
        #[cfg(feature = "babybear")]
        {
            RecursionAir::Poseidon2Wide(Poseidon2WideChip::<$degree>)
        }
        #[cfg(feature = "koalabear")]
        {
            RecursionAir::Poseidon2WideKb(
                crate::chips::poseidon2_wide_kb::Poseidon2WideKbChip::<$degree>,
            )
        }
    }};
}

macro_rules! poseidon2_skinny_chip_for {
    ($degree:expr) => {{
        #[cfg(feature = "babybear")]
        {
            RecursionAir::Poseidon2Skinny(Poseidon2SkinnyChip::<$degree>::default())
        }
        #[cfg(feature = "koalabear")]
        {
            RecursionAir::Poseidon2SkinnyKb(
                crate::chips::poseidon2_skinny_kb::Poseidon2SkinnyKbChip::<$degree>::default(),
            )
        }
    }};
}

impl<F: PrimeField32 + BinomiallyExtendable<D>, const DEGREE: usize> RecursionAir<F, DEGREE> {
    /// Get a machine with all chips, except the dummy chip.
    pub fn sc_machine_wide_with_all_chips<SC: SCStarkGenericConfig<Val = F>>(
        config: SC,
    ) -> SCStarkMachine<SC, Self, RecursionAir<BinomialExtensionField<F, D>, DEGREE>> {
        let chips = vec![
            RecursionAir::MemoryConst(MemoryConstChip::default()),
            RecursionAir::MemoryVar(MemoryVarChip::default()),
            RecursionAir::BaseAlu(BaseAluChip),
            RecursionAir::ExtAlu(ExtAluChip),
            Self::poseidon2_wide_chip(),
            RecursionAir::Select(SelectChip),
            RecursionAir::PublicValues(PublicValuesChip),
        ]
        .into_iter()
        .map(Chip::new)
        .collect::<Vec<_>>();
        let chips_ext = vec![
            RecursionAir::MemoryConst(MemoryConstChip::default()),
            RecursionAir::MemoryVar(MemoryVarChip::default()),
            RecursionAir::BaseAlu(BaseAluChip),
            RecursionAir::ExtAlu(ExtAluChip),
            poseidon2_wide_chip_for!(DEGREE),
            RecursionAir::Select(SelectChip),
            RecursionAir::PublicValues(PublicValuesChip),
        ]
        .into_iter()
        .map(Chip::new)
        .collect::<Vec<_>>();
        SCStarkMachine::new(config, chips, chips_ext, PROOF_MAX_NUM_PVS, false)
    }

    /// Get a machine with all chips, except the dummy chip.
    pub fn sc_machine_skinny_with_all_chips<SC: SCStarkGenericConfig<Val = F>>(
        config: SC,
    ) -> SCStarkMachine<SC, Self, RecursionAir<BinomialExtensionField<F, D>, DEGREE>> {
        let chips = vec![
            RecursionAir::MemoryConst(MemoryConstChip::default()),
            RecursionAir::MemoryVar(MemoryVarChip::default()),
            RecursionAir::BaseAluShrink(BaseAluChip),
            RecursionAir::ExtAluShrink(ExtAluChip),
            Self::poseidon2_skinny_chip(),
            RecursionAir::Select(SelectChip),
            RecursionAir::PublicValues(PublicValuesChip),
        ]
        .into_iter()
        .map(Chip::new)
        .collect::<Vec<_>>();
        let chips_ext = vec![
            RecursionAir::MemoryConst(MemoryConstChip::default()),
            RecursionAir::MemoryVar(MemoryVarChip::default()),
            RecursionAir::BaseAluShrink(BaseAluChip),
            RecursionAir::ExtAluShrink(ExtAluChip),
            poseidon2_skinny_chip_for!(DEGREE),
            RecursionAir::Select(SelectChip),
            RecursionAir::PublicValues(PublicValuesChip),
        ]
        .into_iter()
        .map(Chip::new)
        .collect::<Vec<_>>();
        SCStarkMachine::new(config, chips, chips_ext, PROOF_MAX_NUM_PVS, false)
    }

    /// A machine with dynamic chip sizes that includes the wide variant of the Poseidon2 chip
    /// for the compress stage. `chips` and `chips_ext` must both register the wide chip so that
    /// main / extension traces share the same chip-name ordering and the global LogUp argument
    /// stays balanced.
    pub fn sc_compress_machine<SC: SCStarkGenericConfig<Val = F>>(
        config: SC,
    ) -> SCStarkMachine<SC, Self, RecursionAir<BinomialExtensionField<F, D>, DEGREE>> {
        let chips = vec![
            RecursionAir::MemoryConst(MemoryConstChip::default()),
            RecursionAir::MemoryVar(MemoryVarChip::default()),
            RecursionAir::BaseAlu(BaseAluChip),
            RecursionAir::ExtAlu(ExtAluChip),
            Self::poseidon2_wide_chip(),
            RecursionAir::Select(SelectChip),
            RecursionAir::PublicValues(PublicValuesChip),
            RecursionAir::PrefixSumChecks(PrefixSumChecksChip),
            RecursionAir::ExtExpReverseBits(ExtExpReverseBitsChip::<DEGREE>),
        ]
        .into_iter()
        .map(Chip::new)
        .collect::<Vec<_>>();
        let chips_ext = vec![
            RecursionAir::MemoryConst(MemoryConstChip::default()),
            RecursionAir::MemoryVar(MemoryVarChip::default()),
            RecursionAir::BaseAlu(BaseAluChip),
            RecursionAir::ExtAlu(ExtAluChip),
            poseidon2_wide_chip_for!(DEGREE),
            RecursionAir::Select(SelectChip),
            RecursionAir::PublicValues(PublicValuesChip),
            RecursionAir::PrefixSumChecks(PrefixSumChecksChip),
            RecursionAir::ExtExpReverseBits(ExtExpReverseBitsChip::<DEGREE>),
        ]
        .into_iter()
        .map(Chip::new)
        .collect::<Vec<_>>();
        SCStarkMachine::new(config, chips, chips_ext, PROOF_MAX_NUM_PVS, false)
    }

    /// A machine with dynamic chip sizes that includes the skinny variant of the Poseidon2 chip
    /// for the shrink stage. The shrink stage requires the skinny chip layout, so `chips` and
    /// `chips_ext` both register the skinny chip to keep main / extension trace orderings
    /// consistent.
    ///
    /// `ExtExpReverseBits` and `PrefixSumChecks` chips are intentionally excluded:
    /// `ShrinkConfig` inlines both `exp_reverse_bits_ext` and `eq_poly` into pure ExtAlu
    /// ops, so neither instruction is emitted and registering those chips would cause a
    /// runtime shape-lookup mismatch.
    pub fn sc_shrink_machine<SC: SCStarkGenericConfig<Val = F>>(
        config: SC,
    ) -> SCStarkMachine<SC, Self, RecursionAir<BinomialExtensionField<F, D>, DEGREE>> {
        let chips = vec![
            RecursionAir::MemoryConst(MemoryConstChip::default()),
            RecursionAir::MemoryVar(MemoryVarChip::default()),
            RecursionAir::BaseAluShrink(BaseAluChip),
            RecursionAir::ExtAluShrink(ExtAluChip),
            Self::poseidon2_skinny_chip(),
            RecursionAir::Select(SelectChip),
            RecursionAir::PublicValues(PublicValuesChip),
        ]
        .into_iter()
        .map(Chip::new)
        .collect::<Vec<_>>();
        let chips_ext = vec![
            RecursionAir::MemoryConst(MemoryConstChip::default()),
            RecursionAir::MemoryVar(MemoryVarChip::default()),
            RecursionAir::BaseAluShrink(BaseAluChip),
            RecursionAir::ExtAluShrink(ExtAluChip),
            poseidon2_skinny_chip_for!(DEGREE),
            RecursionAir::Select(SelectChip),
            RecursionAir::PublicValues(PublicValuesChip),
        ]
        .into_iter()
        .map(Chip::new)
        .collect::<Vec<_>>();
        SCStarkMachine::new(config, chips, chips_ext, PROOF_MAX_NUM_PVS, false)
    }

    /// A machine with dynamic chip sizes that includes the wide variant of the Poseidon2 chip
    /// for the wrap stage.
    pub fn sc_wrap_machine<SC: SCStarkGenericConfig<Val = F>>(
        config: SC,
    ) -> SCStarkMachine<SC, Self, RecursionAir<BinomialExtensionField<F, D>, DEGREE>> {
        let chips = vec![
            RecursionAir::MemoryConst(MemoryConstChip::default()),
            RecursionAir::MemoryVar(MemoryVarChip::default()),
            RecursionAir::BaseAlu(BaseAluChip),
            RecursionAir::ExtAlu(ExtAluChip),
            Self::poseidon2_wide_chip(),
            RecursionAir::Select(SelectChip),
            RecursionAir::PublicValues(PublicValuesChip),
            RecursionAir::PolyEval(PolyEvalChip::<DEGREE>),
            RecursionAir::PrefixSumChecks(PrefixSumChecksChip),
            RecursionAir::ExtExpReverseBits(ExtExpReverseBitsChip::<DEGREE>),
        ]
        .into_iter()
        .map(Chip::new)
        .collect::<Vec<_>>();
        let chips_ext = vec![
            RecursionAir::MemoryConst(MemoryConstChip::default()),
            RecursionAir::MemoryVar(MemoryVarChip::default()),
            RecursionAir::BaseAlu(BaseAluChip),
            RecursionAir::ExtAlu(ExtAluChip),
            poseidon2_wide_chip_for!(DEGREE),
            RecursionAir::Select(SelectChip),
            RecursionAir::PublicValues(PublicValuesChip),
            RecursionAir::PolyEval(PolyEvalChip::<DEGREE>),
            RecursionAir::PrefixSumChecks(PrefixSumChecksChip),
            RecursionAir::ExtExpReverseBits(ExtExpReverseBitsChip::<DEGREE>),
        ]
        .into_iter()
        .map(Chip::new)
        .collect::<Vec<_>>();
        SCStarkMachine::new(config, chips, chips_ext, PROOF_MAX_NUM_PVS, false)
    }
}
impl<F> AddAssign<&Instruction<F>> for RecursionAirEventCount {
    #[inline]
    fn add_assign(&mut self, rhs: &Instruction<F>) {
        match rhs {
            Instruction::BaseAlu(_) => self.base_alu_events += 1,
            Instruction::ExtAlu(_) => self.ext_alu_events += 1,
            Instruction::Mem(_) => self.mem_const_events += 1,
            Instruction::Poseidon2(_) => self.poseidon2_wide_events += 1,
            Instruction::Poseidon2Skinny(_) => self.poseidon2_skinny_events += 1,
            Instruction::Select(_) => self.select_events += 1,
            Instruction::Hint(HintInstr { output_addrs_mults })
            | Instruction::HintBits(HintBitsInstr {
                output_addrs_mults,
                input_addr: _, // No receive interaction for the hint operation
            }) => self.mem_var_events += output_addrs_mults.len(),
            Instruction::HintExt2Felts(HintExt2FeltsInstr {
                output_addrs_mults,
                input_addr: _, // No receive interaction for the hint operation
            }) => self.mem_var_events += output_addrs_mults.len(),
            Instruction::HintAddCurve(instr) => {
                self.mem_var_events += instr.output_x_addrs_mults.len();
                self.mem_var_events += instr.output_y_addrs_mults.len();
            }
            Instruction::CommitPublicValues(_) => self.commit_pv_hash_events += 1,
            Instruction::Print(_) => {}
            #[cfg(feature = "debug")]
            Instruction::DebugBacktrace(_) => {}
            Instruction::PolyEval(_) => self.poly_eval_events += 1,
            Instruction::ExtExpReverseBits(ExtExpReverseBitsInstr { addrs, .. }) => {
                self.ext_exp_reverse_bits_events += addrs.exp.len()
            }
            Instruction::PrefixSumChecks(ref instr) => {
                self.prefix_sum_checks_events += instr.addrs.x1_vec.len()
            }
        }
    }
}

impl<F> Add<&Instruction<F>> for RecursionAirEventCount {
    type Output = Self;

    #[inline]
    fn add(mut self, rhs: &Instruction<F>) -> Self::Output {
        self += rhs;
        self
    }
}

impl std::ops::Sub for RecursionAirEventCount {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            mem_const_events: self.mem_const_events - rhs.mem_const_events,
            mem_var_events: self.mem_var_events - rhs.mem_var_events,
            base_alu_events: self.base_alu_events - rhs.base_alu_events,
            ext_alu_events: self.ext_alu_events - rhs.ext_alu_events,
            poseidon2_wide_events: self.poseidon2_wide_events - rhs.poseidon2_wide_events,
            poseidon2_skinny_events: self.poseidon2_skinny_events - rhs.poseidon2_skinny_events,
            select_events: self.select_events - rhs.select_events,
            poly_eval_events: self.poly_eval_events - rhs.poly_eval_events,
            ext_exp_reverse_bits_events: self.ext_exp_reverse_bits_events - rhs.ext_exp_reverse_bits_events,
            prefix_sum_checks_events: self.prefix_sum_checks_events - rhs.prefix_sum_checks_events,
            commit_pv_hash_events: self.commit_pv_hash_events - rhs.commit_pv_hash_events,
        }
    }
}

impl std::ops::Mul<usize> for RecursionAirEventCount {
    type Output = Self;
    fn mul(self, rhs: usize) -> Self {
        Self {
            mem_const_events: self.mem_const_events * rhs,
            mem_var_events: self.mem_var_events * rhs,
            base_alu_events: self.base_alu_events * rhs,
            ext_alu_events: self.ext_alu_events * rhs,
            poseidon2_wide_events: self.poseidon2_wide_events * rhs,
            poseidon2_skinny_events: self.poseidon2_skinny_events * rhs,
            select_events: self.select_events * rhs,
            poly_eval_events: self.poly_eval_events * rhs,
            ext_exp_reverse_bits_events: self.ext_exp_reverse_bits_events * rhs,
            prefix_sum_checks_events: self.prefix_sum_checks_events * rhs,
            commit_pv_hash_events: self.commit_pv_hash_events * rhs,
        }
    }
}

impl std::ops::Add for RecursionAirEventCount {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            mem_const_events: self.mem_const_events + rhs.mem_const_events,
            mem_var_events: self.mem_var_events + rhs.mem_var_events,
            base_alu_events: self.base_alu_events + rhs.base_alu_events,
            ext_alu_events: self.ext_alu_events + rhs.ext_alu_events,
            poseidon2_wide_events: self.poseidon2_wide_events + rhs.poseidon2_wide_events,
            poseidon2_skinny_events: self.poseidon2_skinny_events + rhs.poseidon2_skinny_events,
            select_events: self.select_events + rhs.select_events,
            poly_eval_events: self.poly_eval_events + rhs.poly_eval_events,
            ext_exp_reverse_bits_events: self.ext_exp_reverse_bits_events + rhs.ext_exp_reverse_bits_events,
            prefix_sum_checks_events: self.prefix_sum_checks_events + rhs.prefix_sum_checks_events,
            commit_pv_hash_events: self.commit_pv_hash_events + rhs.commit_pv_hash_events,
        }
    }
}

impl From<RecursionShape> for OrderedShape {
    fn from(value: RecursionShape) -> Self {
        value.inner.into_iter().collect()
    }
}

#[cfg(test)]
pub mod tests {
    use std::{iter::once, sync::Arc};

    use dt_stark::{baby_bear_poseidon2::BabyBearPoseidon2, StarkGenericConfig};
    use machine::RecursionAir;
    use p3_baby_bear::DiffusionMatrixBabyBear;
    use p3_field::{
        extension::{BinomialExtensionField, HasFrobenius},
        AbstractExtensionField, AbstractField, Field,
    };
    use rand::prelude::*;

    // TODO expand glob import
    use crate::{runtime::instruction as instr, *};

    type SC = BabyBearPoseidon2;
    type F = <SC as StarkGenericConfig>::Val;
    type EF = BinomialExtensionField<F, 4>;
    type A = RecursionAir<F, 3>;
    type B = RecursionAir<F, 9>;

    /// Runs the given program on machines that use the wide and skinny Poseidon2 chips.
    pub fn run_recursion_test_machines(program: RecursionProgram<F>) {
        let program = Arc::new(program);
        let mut runtime =
            Runtime::<F, EF, DiffusionMatrixBabyBear>::new(program.clone(), SC::new().perm);
        runtime.run().unwrap();
        // println!("poly eval events: {:?}, total num is: {:?}", runtime.record.poly_eval_events, runtime.record.poly_eval_events.len());

        // Run with the poseidon2 wide chip.
        let machine = A::machine_wide_with_all_chips(BabyBearPoseidon2::compressed());
        let (pk, vk) = machine.setup(&program);
        run_test_machine(vec![runtime.record.clone()], machine, pk, vk)
            .expect("Verification failed");

        // Run with the poseidon2 skinny chip.
        let skinny_machine =
            B::machine_skinny_with_all_chips(BabyBearPoseidon2::ultra_compressed());
        let (pk, vk) = skinny_machine.setup(&program);
        run_test_machine(vec![runtime.record], skinny_machine, pk, vk)
            .expect("Verification failed");
    }

    /// Constructs a linear program and runs it on machines that use the wide and skinny Poseidon2
    /// chips.
    pub fn test_recursion_linear_program(instrs: Vec<Instruction<F>>) {
        run_recursion_test_machines(linear_program(instrs).unwrap());
    }

    #[test]
    pub fn fibonacci() {
        let n = 10;

        let instructions = once(instr::mem(MemAccessKind::Write, 1, 0, 0))
            .chain(once(instr::mem(MemAccessKind::Write, 2, 1, 1)))
            .chain((2..=n).map(|i| instr::base_alu(BaseAluOpcode::AddF, 2, i, i - 2, i - 1)))
            .chain(once(instr::mem(MemAccessKind::Read, 1, n - 1, 34)))
            .chain(once(instr::mem(MemAccessKind::Read, 2, n, 55)))
            .collect::<Vec<_>>();

        test_recursion_linear_program(instructions);
    }

    #[test]
    #[should_panic]
    pub fn div_nonzero_by_zero() {
        let instructions = vec![
            instr::mem(MemAccessKind::Write, 1, 0, 0),
            instr::mem(MemAccessKind::Write, 1, 1, 1),
            instr::base_alu(BaseAluOpcode::DivF, 1, 2, 1, 0),
            instr::mem(MemAccessKind::Read, 1, 2, 1),
        ];

        test_recursion_linear_program(instructions);
    }

    #[test]
    pub fn div_zero_by_zero() {
        let instructions = vec![
            instr::mem(MemAccessKind::Write, 1, 0, 0),
            instr::mem(MemAccessKind::Write, 1, 1, 0),
            instr::base_alu(BaseAluOpcode::DivF, 1, 2, 1, 0),
            instr::mem(MemAccessKind::Read, 1, 2, 1),
        ];

        test_recursion_linear_program(instructions);
    }

    #[test]
    pub fn field_norm() {
        let mut instructions = Vec::new();

        let mut rng = StdRng::seed_from_u64(0xDEADBEEF);
        let mut addr = 0;
        for _ in 0..100 {
            let inner: [F; 4] = std::iter::repeat_with(|| {
                core::array::from_fn(|_| rng.sample(rand::distributions::Standard))
            })
            .find(|xs| !xs.iter().all(F::is_zero))
            .unwrap();
            let x = BinomialExtensionField::<F, D>::from_base_slice(&inner);
            let gal = x.galois_group();

            let mut acc = BinomialExtensionField::one();

            instructions.push(instr::mem_ext(MemAccessKind::Write, 1, addr, acc));
            for conj in gal {
                instructions.push(instr::mem_ext(MemAccessKind::Write, 1, addr + 1, conj));
                instructions.push(instr::ext_alu(ExtAluOpcode::MulE, 1, addr + 2, addr, addr + 1));

                addr += 2;
                acc *= conj;
            }
            let base_cmp: F = acc.as_base_slice()[0];
            instructions.push(instr::mem_single(MemAccessKind::Read, 1, addr, base_cmp));
            addr += 1;
        }

        test_recursion_linear_program(instructions);
    }
}
