use std::marker::PhantomData;

use dt_stark::{
    air::{FullAir, FullAirBuilder, MachineAir, PairCol, PolyAirExtendable},
    sumcheck::{config::SCStarkGenericConfig, trace::CompressedMatrix},
    PROOF_MAX_NUM_PVS,
};
use p3_air::BaseAir;
use p3_field::Field;

use crate::chips::{
    alu_base::NUM_BASE_ALU_SHRINK_ENTRIES_PER_ROW, alu_ext::NUM_EXT_ALU_SHRINK_ENTRIES_PER_ROW,
};

use super::{
    BaseAluChipPolyAir, ExtAluChipPolyAir, ExtExpReverseBitsChipPolyAir, MemoryConstChipPolyAir,
    MemoryVarChipPolyAir, PrefixSumChecksChipPolyAir, PublicValuesChipPolyAir, SelectChipPolyAir,
    SumcheckRoundChipPolyAir,
};

#[cfg(feature = "babybear")]
pub type Poseidon2WideChipPolyAir = super::Poseidon2WideBbChipPolyAir;
#[cfg(feature = "koalabear")]
pub type Poseidon2WideChipPolyAir = super::Poseidon2WideKbChipPolyAir;

#[cfg(feature = "babybear")]
pub type Poseidon2SkinnyChipPolyAir = super::Poseidon2SkinnyBbChipPolyAir;
#[cfg(feature = "koalabear")]
pub type Poseidon2SkinnyChipPolyAir = super::Poseidon2SkinnyKbChipPolyAir;

#[derive(dt_derive::MachinePolyAir)]
#[execution_record_path = "crate::ExecutionRecord<F>"]
#[program_path = "crate::RecursionProgram<F>"]
pub enum RecursionPolyAir<F: Field> {
    MemoryConst(MemoryConstChipPolyAir),
    MemoryVar(MemoryVarChipPolyAir),
    BaseAlu(BaseAluChipPolyAir),
    BaseAluShrink(BaseAluChipPolyAir<NUM_BASE_ALU_SHRINK_ENTRIES_PER_ROW>),
    ExtAlu(ExtAluChipPolyAir),
    ExtAluShrink(ExtAluChipPolyAir<NUM_EXT_ALU_SHRINK_ENTRIES_PER_ROW>),
    Poseidon2Wide(Poseidon2WideChipPolyAir),
    Poseidon2Skinny(Poseidon2SkinnyChipPolyAir),
    Select(SelectChipPolyAir),
    PublicValues(PublicValuesChipPolyAir),
    SumcheckRound(SumcheckRoundChipPolyAir),
    PrefixSumChecks(PrefixSumChecksChipPolyAir),
    ExtExpReverseBits(ExtExpReverseBitsChipPolyAir),
    #[allow(dead_code)]
    Phantom(PhantomChip<F>),
}

#[derive(Default, Clone, Copy)]
pub struct PhantomChip<F>(PhantomData<F>);

impl<F: Field> BaseAir<F> for PhantomChip<F> {
    fn width(&self) -> usize {
        0
    }
}

impl<F: Field> MachineAir<F> for PhantomChip<F> {
    type Record = crate::ExecutionRecord<F>;
    type Program = crate::RecursionProgram<F>;
    fn name(&self) -> String {
        "_phantom".to_string()
    }
    fn preprocessed_width(&self) -> usize {
        0
    }
    fn generate_preprocessed_trace(&self, _: &Self::Program) -> Option<CompressedMatrix<F>> {
        None
    }
    fn generate_dependencies(&self, _: &Self::Record, _: &mut Self::Record) {}
    fn num_rows(&self, _: &Self::Record) -> Option<usize> {
        None
    }
    fn generate_trace(&self, _: &Self::Record, _: &mut Self::Record) -> CompressedMatrix<F> {
        unreachable!()
    }
    fn included(&self, _: &Self::Record) -> bool {
        false
    }
    fn local_only(&self) -> bool {
        true
    }
}

impl<AB: FullAirBuilder> FullAir<AB> for PhantomChip<AB::F> {
    fn width(&self) -> usize {
        0
    }
    fn reserved_poly(&self) -> Vec<PairCol> {
        vec![]
    }
    fn precompute_lc(&self, _: &mut AB) {}
    fn eval(&self, _: &mut AB) {}
    fn lookup(&self, _: &mut AB) {}
}

impl<F: Field> RecursionPolyAir<F> {
    pub fn compress_chips() -> Vec<Self> {
        vec![
            Self::MemoryConst(MemoryConstChipPolyAir),
            Self::MemoryVar(MemoryVarChipPolyAir),
            Self::BaseAlu(BaseAluChipPolyAir::default()),
            Self::ExtAlu(ExtAluChipPolyAir::default()),
            Self::poseidon2_wide_chip(),
            Self::Select(SelectChipPolyAir),
            Self::PublicValues(PublicValuesChipPolyAir),
            Self::SumcheckRound(SumcheckRoundChipPolyAir),
            Self::PrefixSumChecks(PrefixSumChecksChipPolyAir),
            Self::ExtExpReverseBits(ExtExpReverseBitsChipPolyAir),
        ]
    }

    pub fn sc_compress_machine<SC: SCStarkGenericConfig<Val = F>, const D: usize>(
        config: SC,
    ) -> polyair::SCStarkMachine<SC, Self, D>
    where
        F: PolyAirExtendable<D>,
    {
        let chips: Vec<polyair::Chip<Self, F, D>> =
            Self::compress_chips().into_iter().map(|air| polyair::Chip::new(air)).collect();
        polyair::SCStarkMachine::new(config, chips, PROOF_MAX_NUM_PVS, false)
    }

    /// Shrink stage: skinny Poseidon2, shrink ALU sizes, no
    /// SumcheckRound/PrefixSumChecks/ExtExpReverseBits.
    pub fn shrink_chips() -> Vec<Self> {
        vec![
            Self::MemoryConst(MemoryConstChipPolyAir),
            Self::MemoryVar(MemoryVarChipPolyAir),
            Self::BaseAluShrink(BaseAluChipPolyAir::default()),
            Self::ExtAluShrink(ExtAluChipPolyAir::default()),
            Self::poseidon2_skinny_chip(),
            Self::Select(SelectChipPolyAir),
            Self::PublicValues(PublicValuesChipPolyAir),
        ]
    }

    pub fn sc_shrink_machine<SC: SCStarkGenericConfig<Val = F>, const D: usize>(
        config: SC,
    ) -> polyair::SCStarkMachine<SC, Self, D>
    where
        F: PolyAirExtendable<D>,
    {
        let chips: Vec<polyair::Chip<Self, F, D>> =
            Self::shrink_chips().into_iter().map(|air| polyair::Chip::new(air)).collect();
        polyair::SCStarkMachine::new(config, chips, PROOF_MAX_NUM_PVS, false)
    }

    fn poseidon2_wide_chip() -> Self {
        Self::Poseidon2Wide(Poseidon2WideChipPolyAir::default())
    }

    fn poseidon2_skinny_chip() -> Self {
        Self::Poseidon2Skinny(Poseidon2SkinnyChipPolyAir::default())
    }
}
