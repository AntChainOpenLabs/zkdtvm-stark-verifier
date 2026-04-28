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
use crate::chips::sumcheck_round::SumcheckRoundChip;
use crate::{
    chips::{
        alu_base::{BaseAluChip, NUM_BASE_ALU_ENTRIES_PER_ROW},
        alu_ext::{ExtAluChip, NUM_EXT_ALU_ENTRIES_PER_ROW},
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

#[execution_record_path = "crate::ExecutionRecord<F>"]
#[program_path = "crate::RecursionProgram<F>"]
#[builder_path = "crate::builder::DTRecursionAirBuilder<F = F>"]
#[eval_trait_bound = "AB::Var: 'static"]
pub enum RecursionAir<F: Field, const DEGREE: usize> {
    MemoryConst(MemoryConstChip<F>),
    MemoryVar(MemoryVarChip<F>),
    BaseAlu(BaseAluChip),
    ExtAlu(ExtAluChip),
    Poseidon2Skinny(Poseidon2SkinnyChip<DEGREE>),
    #[cfg(feature = "koalabear")]
    Poseidon2SkinnyKb(Poseidon2SkinnyKbChip<DEGREE>),
    Poseidon2Wide(Poseidon2WideChip<DEGREE>),
    #[cfg(feature = "koalabear")]
    Poseidon2WideKb(Poseidon2WideKbChip<DEGREE>),
    Select(SelectChip),
    PublicValues(PublicValuesChip),
    // PolyEval(PolyEvalChip<DEGREE>),
    ExtExpReverseBits(ExtExpReverseBitsChip<DEGREE>),
    // SumcheckRound(SumcheckRoundChip),
    PrefixSumChecks(PrefixSumChecksChip),
}

#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct RecursionAirEventCount {
    pub mem_const_events: usize,
    pub mem_var_events: usize,
    pub base_alu_events: usize,
    pub ext_alu_events: usize,
    pub poseidon2_wide_events: usize,
    pub select_events: usize,
    pub poly_eval_events: usize,
    pub ext_exp_reverse_bits_num: usize,
    pub ext_exp_reverse_bits_events: usize,
    pub sumcheck_round_num: usize,
    pub sumcheck_round_events: usize,
    pub prefix_sum_checks_num: usize,
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
            RecursionAir::BaseAlu(BaseAluChip),
            RecursionAir::ExtAlu(ExtAluChip),
            Self::poseidon2_skinny_chip(),
            RecursionAir::Select(SelectChip),
            RecursionAir::PublicValues(PublicValuesChip),
        ]
        .into_iter()
        .map(Chip::new)
        .collect::<Vec<_>>();
        StarkMachine::new(config, chips, PROOF_MAX_NUM_PVS, false)
    }

    /// A machine with dynamic chip sizes that includes the wide variant of the Poseidon2 chip.
    pub fn compress_machine<SC: StarkGenericConfig<Val = F>>(config: SC) -> StarkMachine<SC, Self> {
        let chips = vec![
            RecursionAir::MemoryConst(MemoryConstChip::default()),
            RecursionAir::MemoryVar(MemoryVarChip::default()),
            RecursionAir::BaseAlu(BaseAluChip),
            RecursionAir::ExtAlu(ExtAluChip),
            Self::poseidon2_wide_chip(),
            RecursionAir::Select(SelectChip),
            RecursionAir::PublicValues(PublicValuesChip),
            // RecursionAir::SumcheckRound(SumcheckRoundChip),
            RecursionAir::PrefixSumChecks(PrefixSumChecksChip),
            RecursionAir::ExtExpReverseBits(ExtExpReverseBitsChip::<DEGREE>),
        ]
        .into_iter()
        .map(Chip::new)
        .collect::<Vec<_>>();
        StarkMachine::new(config, chips, PROOF_MAX_NUM_PVS, false)
    }

    pub fn shrink_machine<SC: StarkGenericConfig<Val = F>>(config: SC) -> StarkMachine<SC, Self> {
        Self::compress_machine(config)
    }

    /// A machine with dynamic chip sizes that includes the skinny variant of the Poseidon2 chip.
    ///
    /// This machine assumes that the `shrink` stage has a fixed shape, so there is no need to
    /// fix the trace sizes.
    pub fn wrap_machine<SC: StarkGenericConfig<Val = F>>(config: SC) -> StarkMachine<SC, Self> {
        let chips = vec![
            RecursionAir::MemoryConst(MemoryConstChip::default()),
            RecursionAir::MemoryVar(MemoryVarChip::default()),
            RecursionAir::BaseAlu(BaseAluChip),
            RecursionAir::ExtAlu(ExtAluChip),
            Self::poseidon2_wide_chip(),
            RecursionAir::Select(SelectChip),
            RecursionAir::PublicValues(PublicValuesChip),
            // RecursionAir::PolyEval(PolyEvalChip::<DEGREE>),
            // RecursionAir::SumcheckRound(SumcheckRoundChip),
            RecursionAir::PrefixSumChecks(PrefixSumChecksChip),
            RecursionAir::ExtExpReverseBits(ExtExpReverseBitsChip::<DEGREE>),
        ]
        .into_iter()
        .map(Chip::new)
        .collect::<Vec<_>>();
        StarkMachine::new(config, chips, PROOF_MAX_NUM_PVS, false)
    }

    pub fn shrink_shape() -> RecursionShape {
        let shape = HashMap::from(
            [
                (Self::MemoryVar(MemoryVarChip::default()), 18),
                (Self::Select(SelectChip), 18),
                (Self::MemoryConst(MemoryConstChip::default()), 17),
                (Self::BaseAlu(BaseAluChip), 17),
                (Self::ExtAlu(ExtAluChip), 18),
                (Self::poseidon2_wide_chip(), 16),
                (Self::PublicValues(PublicValuesChip), PUB_VALUES_LOG_HEIGHT),
                // (Self::SumcheckRound(SumcheckRoundChip), 14),
                (Self::ExtExpReverseBits(ExtExpReverseBitsChip::<DEGREE>), 15),
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
            (Self::poseidon2_wide_chip(), heights.poseidon2_wide_events),
            (Self::Select(SelectChip), heights.select_events),
            (Self::PublicValues(PublicValuesChip), PUB_VALUES_LOG_HEIGHT),
            // (Self::SumcheckRound(SumcheckRoundChip), heights.sumcheck_round_events),
            (Self::PrefixSumChecks(PrefixSumChecksChip), heights.prefix_sum_checks_events),
            (
                Self::ExtExpReverseBits(ExtExpReverseBitsChip::<DEGREE>),
                heights.ext_exp_reverse_bits_events,
            ),
        ]
        .map(|(chip, log_height)| (chip.name(), log_height))
        .to_vec()
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
            RecursionAir::BaseAlu(BaseAluChip),
            RecursionAir::ExtAlu(ExtAluChip),
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
            RecursionAir::BaseAlu(BaseAluChip),
            RecursionAir::ExtAlu(ExtAluChip),
            poseidon2_skinny_chip_for!(DEGREE),
            RecursionAir::Select(SelectChip),
            RecursionAir::PublicValues(PublicValuesChip),
        ]
        .into_iter()
        .map(Chip::new)
        .collect::<Vec<_>>();
        SCStarkMachine::new(config, chips, chips_ext, PROOF_MAX_NUM_PVS, false)
    }

    /// A machine with dynamic chip sizes that includes the wide variant of the Poseidon2 chip.
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
            // RecursionAir::SumcheckRound(SumcheckRoundChip),
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
            // RecursionAir::SumcheckRound(SumcheckRoundChip),
            RecursionAir::PrefixSumChecks(PrefixSumChecksChip),
            RecursionAir::ExtExpReverseBits(ExtExpReverseBitsChip::<DEGREE>),
        ]
        .into_iter()
        .map(Chip::new)
        .collect::<Vec<_>>();
        SCStarkMachine::new(config, chips, chips_ext, PROOF_MAX_NUM_PVS, false)
    }

    pub fn sc_shrink_machine<SC: SCStarkGenericConfig<Val = F>>(
        config: SC,
    ) -> SCStarkMachine<SC, Self, RecursionAir<BinomialExtensionField<F, D>, DEGREE>> {
        Self::sc_compress_machine(config)
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
            // RecursionAir::PolyEval(PolyEvalChip::<DEGREE>),
            // RecursionAir::SumcheckRound(SumcheckRoundChip),
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
            // RecursionAir::PolyEval(PolyEvalChip::<DEGREE>),
            // RecursionAir::SumcheckRound(SumcheckRoundChip),
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
                self.ext_exp_reverse_bits_num += 1;
                self.ext_exp_reverse_bits_events += addrs.exp.len()
            }
            Instruction::SumcheckRound(ref instr) => {
                self.sumcheck_round_num += 1;
                self.sumcheck_round_events += instr.addrs.coeffs.len()
            }
            Instruction::PrefixSumChecks(ref instr) => {
                self.prefix_sum_checks_num += 1;
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

impl From<RecursionShape> for OrderedShape {
    fn from(value: RecursionShape) -> Self {
        value.inner.into_iter().collect()
    }
}
