use dt_stark::{
    air::{FullAir, InteractionScope, MachineAir, PairCol},
    sumcheck::trace::CompressedMatrix,
};
use p3_air::BaseAir;

use crate::{
    config::F,
    primitives_dt::{bus::RangeCheckerBus, range::RangeCheckerAir},
    proof_shape_dt::ProofHeightSetAir,
    system_dt::{RecursionNativeProgram, RecursionRecord},
    transcript_dt::{
        bus::Poseidon2PermuteBus, merkle_path::MerklePathAir, poseidon2::Poseidon2PermuteAir,
        sponge::TranscriptSpongeAir,
    },
    whir_dt::{
        WhirLeafExtStreamAir, WhirLeafStreamAir, WhirQueryFoldAir, WhirSampleBandAir,
        WhirTwiddleTableAir,
    },
};

use super::NativeAirFamily;

/// The 11 mathematical AIR families whose logical identity is independent of
/// the native recursion layer.
#[derive(Debug, Clone)]
pub enum NativeSharedAir {
    TranscriptSponge(TranscriptSpongeAir),
    MerklePath(MerklePathAir),
    Poseidon2Permute(Poseidon2PermuteAir),
    ProofHeightSet(ProofHeightSetAir),
    WhirTwiddleTable(WhirTwiddleTableAir),
    WhirSampleBand(WhirSampleBandAir),
    WhirQueryFold(WhirQueryFoldAir),
    WhirLeafStream(WhirLeafStreamAir),
    WhirLeafExtStream(WhirLeafExtStreamAir),
    Range8(RangeCheckerAir<8>),
    Range21(RangeCheckerAir<21>),
}

impl NativeSharedAir {
    pub fn all() -> Vec<Self> {
        vec![
            Self::TranscriptSponge(TranscriptSpongeAir::default()),
            Self::MerklePath(MerklePathAir::default()),
            Self::Poseidon2Permute(Poseidon2PermuteAir::new(Poseidon2PermuteBus::new())),
            Self::ProofHeightSet(ProofHeightSetAir::default()),
            Self::WhirTwiddleTable(WhirTwiddleTableAir::default()),
            Self::WhirSampleBand(WhirSampleBandAir::default()),
            Self::WhirQueryFold(WhirQueryFoldAir::default()),
            Self::WhirLeafStream(WhirLeafStreamAir::default()),
            Self::WhirLeafExtStream(WhirLeafExtStreamAir::default()),
            Self::Range8(RangeCheckerAir::<8>::new(RangeCheckerBus::new())),
            Self::Range21(RangeCheckerAir::<21>::new(RangeCheckerBus::new())),
        ]
    }

    pub fn family(&self) -> NativeAirFamily {
        match self {
            Self::TranscriptSponge(_) => NativeAirFamily::TranscriptSponge,
            Self::MerklePath(_) => NativeAirFamily::MerklePath,
            Self::Poseidon2Permute(_) => NativeAirFamily::Poseidon2Permute,
            Self::ProofHeightSet(_) => NativeAirFamily::ProofHeightSet,
            Self::WhirTwiddleTable(_) => NativeAirFamily::WhirTwiddleTable,
            Self::WhirSampleBand(_) => NativeAirFamily::WhirSampleBand,
            Self::WhirQueryFold(_) => NativeAirFamily::WhirQueryFold,
            Self::WhirLeafStream(_) => NativeAirFamily::WhirLeafStream,
            Self::WhirLeafExtStream(_) => NativeAirFamily::WhirLeafExtStream,
            Self::Range8(_) => NativeAirFamily::Range8,
            Self::Range21(_) => NativeAirFamily::Range21,
        }
    }
}

macro_rules! dispatch_shared_air {
    ($self:expr, $air:ident => $body:expr) => {
        match $self {
            NativeSharedAir::TranscriptSponge($air) => $body,
            NativeSharedAir::MerklePath($air) => $body,
            NativeSharedAir::Poseidon2Permute($air) => $body,
            NativeSharedAir::ProofHeightSet($air) => $body,
            NativeSharedAir::WhirTwiddleTable($air) => $body,
            NativeSharedAir::WhirSampleBand($air) => $body,
            NativeSharedAir::WhirQueryFold($air) => $body,
            NativeSharedAir::WhirLeafStream($air) => $body,
            NativeSharedAir::WhirLeafExtStream($air) => $body,
            NativeSharedAir::Range8($air) => $body,
            NativeSharedAir::Range21($air) => $body,
        }
    };
}

impl BaseAir<F> for NativeSharedAir {
    fn width(&self) -> usize {
        dispatch_shared_air!(self, air => BaseAir::<F>::width(air))
    }
}

impl<AB> FullAir<AB> for NativeSharedAir
where
    AB: dt_stark::air::FullAirBuilder<F = F>,
{
    fn width(&self) -> usize {
        dispatch_shared_air!(self, air => FullAir::<AB>::width(air))
    }

    fn num_public_values(&self) -> usize {
        dispatch_shared_air!(self, air => FullAir::<AB>::num_public_values(air))
    }

    fn required_max_beta_power(&self) -> usize {
        dispatch_shared_air!(self, air => FullAir::<AB>::required_max_beta_power(air))
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        dispatch_shared_air!(self, air => FullAir::<AB>::reserved_poly(air))
    }

    fn precompute_lc(&self, builder: &mut AB) {
        dispatch_shared_air!(self, air => FullAir::<AB>::precompute_lc(air, builder))
    }

    fn eval(&self, builder: &mut AB) {
        dispatch_shared_air!(self, air => FullAir::<AB>::eval(air, builder))
    }

    fn lookup(&self, builder: &mut AB) {
        dispatch_shared_air!(self, air => FullAir::<AB>::lookup(air, builder))
    }

    fn global(&self) -> bool {
        dispatch_shared_air!(self, air => FullAir::<AB>::global(air))
    }
}

impl MachineAir<F> for NativeSharedAir {
    type Record = RecursionRecord;
    type Program = RecursionNativeProgram<F>;

    fn name(&self) -> String {
        dispatch_shared_air!(self, air => MachineAir::<F>::name(air))
    }

    fn preprocessed_width(&self) -> usize {
        dispatch_shared_air!(self, air => MachineAir::<F>::preprocessed_width(air))
    }

    fn preprocessed_num_rows(&self, program: &Self::Program, instrs_len: usize) -> Option<usize> {
        dispatch_shared_air!(self, air =>
            MachineAir::<F>::preprocessed_num_rows(air, program, instrs_len)
        )
    }

    fn generate_preprocessed_trace(&self, program: &Self::Program) -> Option<CompressedMatrix<F>> {
        dispatch_shared_air!(self, air =>
            MachineAir::<F>::generate_preprocessed_trace(air, program)
        )
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        dispatch_shared_air!(self, air => MachineAir::<F>::num_rows(air, input))
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        dispatch_shared_air!(self, air => MachineAir::<F>::generate_trace(air, input, output))
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        dispatch_shared_air!(self, air =>
            MachineAir::<F>::generate_dependencies(air, input, output)
        )
    }

    fn included(&self, record: &Self::Record) -> bool {
        dispatch_shared_air!(self, air => MachineAir::<F>::included(air, record))
    }

    fn commit_scope(&self) -> InteractionScope {
        dispatch_shared_air!(self, air => MachineAir::<F>::commit_scope(air))
    }

    fn local_only(&self) -> bool {
        dispatch_shared_air!(self, air => MachineAir::<F>::local_only(air))
    }

    fn padding_row(&self) -> Vec<F> {
        dispatch_shared_air!(self, air => MachineAir::<F>::padding_row(air))
    }
}
