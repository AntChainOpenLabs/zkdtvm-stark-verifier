use crate::StarkGenericConfig;
use native_recursion::machine_dt::{CpuNativeProver, NativeProverProvider};
use dt_core_machine::riscv::riscv_polyair::RiscvPolyAir;
use dt_stark::{
    sumcheck::{
        keys::SCStarkProvingKey,
        prover::{SCMachineProver as StarkMachineProver, SumcheckProver},
    },
    Challenge,
};
use polyair::prover::{
    SCMachineProver as PolyAirMachineProver, SumcheckProver as PolyAirSumcheckProver,
};

use crate::{
    CompressPolyAir, CoreSC, InnerSC, OuterSC, RootSC, ShrinkPolyAir, WrapAir, POLYAIR_EXT_DEGREE,
};

pub trait DTProverComponents: Send + Sync {
    type NativeProvider: NativeProverProvider;

    /// The prover for making zkDTVM core proofs.
    type CoreProver: PolyAirMachineProver<
            CoreSC,
            RiscvPolyAir<<CoreSC as StarkGenericConfig>::Val>,
            POLYAIR_EXT_DEGREE,
        > + Send
        + Sync;

    /// The prover for making zkDTVM recursive proofs.
    type CompressProver: PolyAirMachineProver<
            InnerSC,
            CompressPolyAir<<InnerSC as StarkGenericConfig>::Val>,
            POLYAIR_EXT_DEGREE,
        > + Send
        + Sync;

    /// The prover for shrinking compressed proofs.
    type ShrinkProver: PolyAirMachineProver<
            InnerSC,
            ShrinkPolyAir<<InnerSC as StarkGenericConfig>::Val>,
            POLYAIR_EXT_DEGREE,
        > + Send
        + Sync;

    /// The prover for the final root_shrink proof. This stage is not verified
    /// recursively, so its PCS may use a native-verifier-friendly hash.
    type RootShrinkProver: PolyAirMachineProver<
            RootSC,
            ShrinkPolyAir<<RootSC as StarkGenericConfig>::Val>,
            POLYAIR_EXT_DEGREE,
        > + Send
        + Sync;

    /// The prover for wrapping compressed proofs into SNARK-friendly field elements.
    type WrapProver: StarkMachineProver<
            OuterSC,
            WrapAir<<OuterSC as StarkGenericConfig>::Val>,
            WrapAir<Challenge<OuterSC>>,
        > + Send
        + Sync;
}

pub struct SCCpuProverComponents;

impl DTProverComponents for SCCpuProverComponents {
    type NativeProvider = CpuNativeProver;

    type CoreProver = PolyAirSumcheckProver<
        CoreSC,
        RiscvPolyAir<<CoreSC as StarkGenericConfig>::Val>,
        POLYAIR_EXT_DEGREE,
    >;
    type CompressProver = PolyAirSumcheckProver<
        InnerSC,
        CompressPolyAir<<InnerSC as StarkGenericConfig>::Val>,
        POLYAIR_EXT_DEGREE,
    >;
    type ShrinkProver = PolyAirSumcheckProver<
        InnerSC,
        ShrinkPolyAir<<InnerSC as StarkGenericConfig>::Val>,
        POLYAIR_EXT_DEGREE,
    >;
    type RootShrinkProver = PolyAirSumcheckProver<
        RootSC,
        ShrinkPolyAir<<RootSC as StarkGenericConfig>::Val>,
        POLYAIR_EXT_DEGREE,
    >;
    type WrapProver = SumcheckProver<
        OuterSC,
        WrapAir<<OuterSC as StarkGenericConfig>::Val>,
        WrapAir<Challenge<OuterSC>>,
    >;
}
