use core::fmt;

use dt_core_executor::{ExecutionRecord, Program, RiscvAirId};
use dt_stark::{
    air::{MachineAir, PolyAirExtendable, DT_PROOF_NUM_PV_ELTS},
    sumcheck::config::SCStarkGenericConfig,
};
use hashbrown::HashMap;
use p3_air::BaseAir;
use p3_field::Field;
use polyair::{Chip, SCStarkMachine};
use strum_macros::{EnumDiscriminants, EnumIter};

use crate::{
    alu::{
        AddChipPolyAir, AddiChipPolyAir, BitwiseChipPolyAir, DivRemChipPolyAir, LtChipPolyAir,
        MulChipPolyAir, ShiftLeftPolyAir, ShiftRightPolyAir, SubChipPolyAir,
    },
    bytes::{byte_polyair::ByteChipPolyAir, trace::NUM_ROWS as BYTE_CHIP_NUM_ROWS},
    control_flow::{AuipcChipPolyAir, BranchChipPolyAir, JalChipPolyAir, JalrChipPolyAir},
    global::{GlobalChip, GlobalTileReducerChip},
    memory::{
        global_polyair::MemoryGlobalChipPolyAir, local_polyair::MemoryLocalChipPolyAir,
        LoadByteChipPolyAir, LoadHalfChipPolyAir, LoadWordChipPolyAir, MemoryChipType,
        StoreByteChipPolyAir, StoreHalfChipPolyAir, StoreWordChipPolyAir,
    },
    program::program_polyair::ProgramChipPolyAir,
    riscv::{
        riscv_chips::{
            Bls12381Parameters, Bn254Parameters, Ed25519Parameters, EdwardsCurve,
            Secp256k1Parameters, Secp256r1Parameters, SwCurve,
        },
        RiscvAir,
    },
    shape::Shapeable,
    syscall::{
        chip::SyscallShardKind,
        instructions::syscall_instrs_polyair::SyscallInstrsChipPolyAir,
        precompiles::{
            edwards::{EdAddAssignPolyAir, EdDecompressPolyAir},
            fptower::{Fp2AddSubPolyAir, Fp2MulPolyAir, FpOpPolyAir},
            keccak_dt::{
                keccak_controller_polyair::KeccakControllerPolyAir,
                keccak_polyair::KeccakPermutePolyAir,
            },
            poseidon_permute::poseidon_permute_polyair::Poseidon2PermutePolyAir,
            sha256::{
                compress_dt::{
                    compress_controller_polyair::ShaCompressControllerPolyAir,
                    compress_polyair::ShaCompressPolyAir,
                },
                extend_dt::{
                    extend_controller_polyair::ShaExtendControllerPolyAir,
                    extend_polyair::ShaExtendPolyAir,
                },
            },
            u256x2048_mul::U256x2048MulPolyAir,
            uint256::Uint256MulPolyAir,
            weierstrass::{
                SignChoiceRule, WeierstrassAddPolyAir, WeierstrassDecompressPolyAir,
                WeierstrassDoublePolyAir,
            },
        },
        syscall_polyair::SyscallChipPolyAir,
    },
};
use dt_curves::weierstrass::{bls12_381::Bls12381BaseField, bn254::Bn254BaseField};

#[derive(dt_derive::MachinePolyAir, EnumDiscriminants)]
#[strum_discriminants(derive(Hash, EnumIter))]
pub enum RiscvPolyAir<F: Field> {
    /// An AIR that contains a preprocessed program table and a lookup for the instructions.
    Program(ProgramChipPolyAir),
    /// An AIR for RISC-V Bitwise instructions.
    Bitwise(BitwiseChipPolyAir),
    /// An AIR for RISC-V Mul instruction.
    Mul(MulChipPolyAir),
    /// An AIR for RISC-V Div and Rem instructions.
    DivRem(DivRemChipPolyAir),
    /// An AIR for RISC-V Lt instruction.
    Lt(LtChipPolyAir),
    /// An AIR for RISC-V SLL instruction.
    ShiftLeft(ShiftLeftPolyAir),
    /// An AIR for RISC-V SRL and SRA instruction.
    ShiftRight(ShiftRightPolyAir),
    /// An AIR for RISC-V AUIPC instruction.
    AUIPC(AuipcChipPolyAir),
    /// An AIR for RISC-V branch instructions.
    Branch(BranchChipPolyAir),
    /// An AIR for RISC-V ecall instructions.
    SyscallInstrs(SyscallInstrsChipPolyAir),
    /// A lookup table for byte operations.
    ByteLookup(ByteChipPolyAir),
    /// A table for initializing the global memory state.
    MemoryGlobalInit(MemoryGlobalChipPolyAir),
    /// A table for finalizing the global memory state.
    MemoryGlobalFinal(MemoryGlobalChipPolyAir),
    /// A table for the local memory state.
    MemoryLocal(MemoryLocalChipPolyAir),
    /// A table for all the syscall invocations.
    SyscallCore(SyscallChipPolyAir),
    /// A table for all the precompile invocations.
    SyscallPrecompile(SyscallChipPolyAir),
    /// A table for all the global interactions.
    Global(GlobalChip),
    /// A table canonicalizing raw Global tile terminals.
    GlobalTileReducer(GlobalTileReducerChip),
    /// An AIR for the RISC-V ADD instruction.
    Add(AddChipPolyAir),
    /// An AIR for the RISC-V ADDI instruction.
    Addi(AddiChipPolyAir),
    /// An AIR for the RISC-V SUB instruction.
    Sub(SubChipPolyAir),
    /// An AIR for RISC-V load byte instructions.
    LoadByte(LoadByteChipPolyAir),
    /// An AIR for RISC-V load half instructions.
    LoadHalf(LoadHalfChipPolyAir),
    /// An AIR for RISC-V load word instructions.
    LoadWord(LoadWordChipPolyAir),
    /// An AIR for RISC-V store byte instructions.
    StoreByte(StoreByteChipPolyAir),
    /// An AIR for RISC-V store half instructions.
    StoreHalf(StoreHalfChipPolyAir),
    /// An AIR for RISC-V store word instructions.
    StoreWord(StoreWordChipPolyAir),
    /// An AIR for the RISC-V JAL instruction.
    Jal(JalChipPolyAir),
    /// An AIR for the RISC-V JALR instruction.
    Jalr(JalrChipPolyAir),
    /// A precompile for sha256 extend.
    Sha256Extend(ShaExtendPolyAir),
    /// A precompile for sha256 compress.
    Sha256Compress(ShaCompressPolyAir),
    /// A precompile for addition on the Elliptic curve ed25519.
    Ed25519Add(EdAddAssignPolyAir<EdwardsCurve<Ed25519Parameters>>),
    /// A precompile for decompressing a point on the Edwards curve ed25519.
    Ed25519Decompress(EdDecompressPolyAir<Ed25519Parameters>),
    /// A precompile for decompressing a point on the K256 curve.
    K256Decompress(WeierstrassDecompressPolyAir<SwCurve<Secp256k1Parameters>>),
    /// A precompile for decompressing a point on the P256 curve.
    P256Decompress(WeierstrassDecompressPolyAir<SwCurve<Secp256r1Parameters>>),
    /// A precompile for addition on the Elliptic curve secp256k1.
    Secp256k1Add(WeierstrassAddPolyAir<SwCurve<Secp256k1Parameters>>),
    /// A precompile for doubling a point on the Elliptic curve secp256k1.
    Secp256k1Double(WeierstrassDoublePolyAir<SwCurve<Secp256k1Parameters>>),
    /// A precompile for addition on the Elliptic curve secp256r1.
    Secp256r1Add(WeierstrassAddPolyAir<SwCurve<Secp256r1Parameters>>),
    /// A precompile for doubling a point on the Elliptic curve secp256r1.
    Secp256r1Double(WeierstrassDoublePolyAir<SwCurve<Secp256r1Parameters>>),
    /// A precompile for the Keccak permutation.
    KeccakP(KeccakPermutePolyAir),
    /// A precompile for addition on the Elliptic curve bn254.
    Bn254Add(WeierstrassAddPolyAir<SwCurve<Bn254Parameters>>),
    /// A precompile for doubling a point on the Elliptic curve bn254.
    Bn254Double(WeierstrassDoublePolyAir<SwCurve<Bn254Parameters>>),
    /// A precompile for addition on the Elliptic curve bls12_381.
    Bls12381Add(WeierstrassAddPolyAir<SwCurve<Bls12381Parameters>>),
    /// A precompile for doubling a point on the Elliptic curve bls12_381.
    Bls12381Double(WeierstrassDoublePolyAir<SwCurve<Bls12381Parameters>>),
    /// A precompile for uint256 mul.
    Uint256Mul(Uint256MulPolyAir),
    /// A precompile for u256x2048 mul.
    U256x2048Mul(U256x2048MulPolyAir),
    /// A precompile for decompressing a point on the BLS12-381 curve.
    Bls12381Decompress(WeierstrassDecompressPolyAir<SwCurve<Bls12381Parameters>>),
    /// A precompile for BLS12-381 fp operation.
    Bls12381Fp(FpOpPolyAir<Bls12381BaseField>),
    /// A precompile for BLS12-381 fp2 multiplication.
    Bls12381Fp2Mul(Fp2MulPolyAir<Bls12381BaseField>),
    /// A precompile for BLS12-381 fp2 addition/subtraction.
    Bls12381Fp2AddSub(Fp2AddSubPolyAir<Bls12381BaseField>),
    /// A precompile for BN-254 fp operation.
    Bn254Fp(FpOpPolyAir<Bn254BaseField>),
    /// A precompile for BN-254 fp2 multiplication.
    Bn254Fp2Mul(Fp2MulPolyAir<Bn254BaseField>),
    /// A precompile for BN-254 fp2 addition/subtraction.
    Bn254Fp2AddSub(Fp2AddSubPolyAir<Bn254BaseField>),
    ShaExtendController(ShaExtendControllerPolyAir),
    ShaCompressController(ShaCompressControllerPolyAir),
    KeccakController(KeccakControllerPolyAir),
    /// A precompile for Poseidon2 permutation.
    Poseidon2Permute(Poseidon2PermutePolyAir<F>),
}

impl<F: Field> RiscvPolyAir<F> {
    pub fn id(&self) -> RiscvAirId {
        RiscvAirId::from(RiscvPolyAirDiscriminants::from(self))
    }

    pub fn costs() -> HashMap<String, u64> {
        let mut costs = HashMap::new();
        for air in Self::all_air() {
            let cost = (air.preprocessed_width() + air.width()) as u64;
            costs.insert(air.name(), cost);
        }
        costs
    }

    pub fn sc_machine<SC: SCStarkGenericConfig<Val = F>, const D: usize>(
        config: SC,
    ) -> SCStarkMachine<SC, Self, D>
    where
        F: PolyAirExtendable<D>,
    {
        let chips: Vec<_> = Self::all_air().into_iter().map(|air| Chip::new(air)).collect();
        SCStarkMachine::new(config, chips, DT_PROOF_NUM_PV_ELTS, true)
    }

    /// Returns all RISC-V polyairs in the same order as `RiscvAir::chips()`.
    pub fn all_air() -> Vec<Self> {
        vec![
            Self::Program(ProgramChipPolyAir::default()),
            Self::Sha256Extend(ShaExtendPolyAir::default()),
            Self::Sha256Compress(ShaCompressPolyAir::default()),
            Self::Ed25519Add(EdAddAssignPolyAir::<EdwardsCurve<Ed25519Parameters>>::new()),
            Self::Ed25519Decompress(EdDecompressPolyAir::<Ed25519Parameters>::new()),
            Self::K256Decompress(
                WeierstrassDecompressPolyAir::<SwCurve<Secp256k1Parameters>>::new(
                    SignChoiceRule::LeastSignificantBit,
                ),
            ),
            Self::Secp256k1Add(WeierstrassAddPolyAir::<SwCurve<Secp256k1Parameters>>::new()),
            Self::Secp256k1Double(WeierstrassDoublePolyAir::<SwCurve<Secp256k1Parameters>>::new()),
            Self::P256Decompress(
                WeierstrassDecompressPolyAir::<SwCurve<Secp256r1Parameters>>::new(
                    SignChoiceRule::LeastSignificantBit,
                ),
            ),
            Self::Secp256r1Add(WeierstrassAddPolyAir::<SwCurve<Secp256r1Parameters>>::new()),
            Self::Secp256r1Double(WeierstrassDoublePolyAir::<SwCurve<Secp256r1Parameters>>::new()),
            Self::KeccakP(KeccakPermutePolyAir::new()),
            Self::Bn254Add(WeierstrassAddPolyAir::<SwCurve<Bn254Parameters>>::new()),
            Self::Bn254Double(WeierstrassDoublePolyAir::<SwCurve<Bn254Parameters>>::new()),
            Self::Bls12381Add(WeierstrassAddPolyAir::<SwCurve<Bls12381Parameters>>::new()),
            Self::Bls12381Double(WeierstrassDoublePolyAir::<SwCurve<Bls12381Parameters>>::new()),
            Self::Uint256Mul(Uint256MulPolyAir::default()),
            Self::U256x2048Mul(U256x2048MulPolyAir::default()),
            Self::Bls12381Fp(FpOpPolyAir::<Bls12381BaseField>::new()),
            Self::Bls12381Fp2AddSub(Fp2AddSubPolyAir::<Bls12381BaseField>::new()),
            Self::Bls12381Fp2Mul(Fp2MulPolyAir::<Bls12381BaseField>::new()),
            Self::Bn254Fp(FpOpPolyAir::<Bn254BaseField>::new()),
            Self::Bn254Fp2AddSub(Fp2AddSubPolyAir::<Bn254BaseField>::new()),
            Self::Bn254Fp2Mul(Fp2MulPolyAir::<Bn254BaseField>::new()),
            Self::Bls12381Decompress(
                WeierstrassDecompressPolyAir::<SwCurve<Bls12381Parameters>>::new(
                    SignChoiceRule::Lexicographic,
                ),
            ),
            Self::SyscallCore(SyscallChipPolyAir::new(SyscallShardKind::Core)),
            Self::SyscallPrecompile(SyscallChipPolyAir::new(SyscallShardKind::Precompile)),
            Self::DivRem(DivRemChipPolyAir::default()),
            Self::Bitwise(BitwiseChipPolyAir::default()),
            Self::Mul(MulChipPolyAir::default()),
            Self::ShiftRight(ShiftRightPolyAir::default()),
            Self::ShiftLeft(ShiftLeftPolyAir::default()),
            Self::Lt(LtChipPolyAir::default()),
            Self::AUIPC(AuipcChipPolyAir::default()),
            Self::Branch(BranchChipPolyAir::default()),
            Self::SyscallInstrs(SyscallInstrsChipPolyAir::default()),
            Self::MemoryGlobalInit(MemoryGlobalChipPolyAir::new(MemoryChipType::Initialize)),
            Self::MemoryGlobalFinal(MemoryGlobalChipPolyAir::new(MemoryChipType::Finalize)),
            Self::MemoryLocal(MemoryLocalChipPolyAir),
            Self::ShaExtendController(ShaExtendControllerPolyAir::new()),
            Self::ShaCompressController(ShaCompressControllerPolyAir::new()),
            Self::KeccakController(KeccakControllerPolyAir::new()),
            Self::Global(GlobalChip),
            Self::GlobalTileReducer(GlobalTileReducerChip),
            Self::ByteLookup(ByteChipPolyAir::default()),
            Self::Poseidon2Permute(Poseidon2PermutePolyAir::default()),
            Self::Add(AddChipPolyAir::default()),
            Self::Addi(AddiChipPolyAir::default()),
            Self::Sub(SubChipPolyAir::default()),
            Self::LoadByte(LoadByteChipPolyAir::default()),
            Self::LoadHalf(LoadHalfChipPolyAir::default()),
            Self::LoadWord(LoadWordChipPolyAir::default()),
            Self::StoreByte(StoreByteChipPolyAir::default()),
            Self::StoreHalf(StoreHalfChipPolyAir::default()),
            Self::StoreWord(StoreWordChipPolyAir::default()),
            Self::Jal(JalChipPolyAir::default()),
            Self::Jalr(JalrChipPolyAir::default()),
        ]
    }

    /// Get the heights of the preprocessed airs for a given program.
    pub fn preprocessed_heights(program: &Program) -> Vec<(RiscvAirId, usize)> {
        vec![
            (RiscvAirId::Program, program.instructions.len()),
            (RiscvAirId::Byte, BYTE_CHIP_NUM_ROWS),
        ]
    }

    /// Get the heights of the airs for a given execution record.
    pub fn core_heights(record: &ExecutionRecord) -> Vec<(RiscvAirId, usize)> {
        record.core_heights()
    }

    pub fn get_all_core_airs() -> Vec<Self> {
        vec![
            Self::Add(AddChipPolyAir::default()),
            Self::Addi(AddiChipPolyAir::default()),
            Self::Sub(SubChipPolyAir::default()),
            Self::Bitwise(BitwiseChipPolyAir::default()),
            Self::Mul(MulChipPolyAir::default()),
            Self::DivRem(DivRemChipPolyAir::default()),
            Self::Lt(LtChipPolyAir::default()),
            Self::ShiftLeft(ShiftLeftPolyAir::default()),
            Self::ShiftRight(ShiftRightPolyAir::default()),
            Self::LoadByte(LoadByteChipPolyAir::default()),
            Self::LoadHalf(LoadHalfChipPolyAir::default()),
            Self::LoadWord(LoadWordChipPolyAir::default()),
            Self::StoreByte(StoreByteChipPolyAir::default()),
            Self::StoreHalf(StoreHalfChipPolyAir::default()),
            Self::StoreWord(StoreWordChipPolyAir::default()),
            Self::AUIPC(AuipcChipPolyAir::default()),
            Self::Branch(BranchChipPolyAir::default()),
            Self::Jal(JalChipPolyAir::default()),
            Self::Jalr(JalrChipPolyAir::default()),
            Self::SyscallInstrs(SyscallInstrsChipPolyAir::default()),
            Self::MemoryLocal(MemoryLocalChipPolyAir),
            Self::Global(GlobalChip),
            Self::GlobalTileReducer(GlobalTileReducerChip),
            Self::SyscallCore(SyscallChipPolyAir::new(SyscallShardKind::Core)),
        ]
    }

    pub fn memory_init_final_airs() -> Vec<Self> {
        vec![
            Self::MemoryGlobalInit(MemoryGlobalChipPolyAir::new(MemoryChipType::Initialize)),
            Self::MemoryGlobalFinal(MemoryGlobalChipPolyAir::new(MemoryChipType::Finalize)),
            Self::Global(GlobalChip),
            Self::GlobalTileReducer(GlobalTileReducerChip),
        ]
    }

    pub fn precompile_airs_with_memory_events_per_row() -> impl Iterator<Item = (RiscvAirId, usize)>
    {
        RiscvAir::<F>::precompile_airs_with_memory_events_per_row()
    }
}

impl<F: Field> PartialEq for RiscvPolyAir<F> {
    fn eq(&self, other: &Self) -> bool {
        self.name() == other.name()
    }
}

impl<F: Field> Eq for RiscvPolyAir<F> {}

impl<F: Field> core::hash::Hash for RiscvPolyAir<F> {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.name().hash(state);
    }
}

impl<F: Field> fmt::Debug for RiscvPolyAir<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl From<RiscvPolyAirDiscriminants> for RiscvAirId {
    fn from(value: RiscvPolyAirDiscriminants) -> Self {
        match value {
            RiscvPolyAirDiscriminants::Program => RiscvAirId::Program,
            RiscvPolyAirDiscriminants::Add => RiscvAirId::Add,
            RiscvPolyAirDiscriminants::Addi => RiscvAirId::Addi,
            RiscvPolyAirDiscriminants::Sub => RiscvAirId::Sub,
            RiscvPolyAirDiscriminants::Bitwise => RiscvAirId::Bitwise,
            RiscvPolyAirDiscriminants::Mul => RiscvAirId::Mul,
            RiscvPolyAirDiscriminants::DivRem => RiscvAirId::DivRem,
            RiscvPolyAirDiscriminants::Lt => RiscvAirId::Lt,
            RiscvPolyAirDiscriminants::ShiftLeft => RiscvAirId::ShiftLeft,
            RiscvPolyAirDiscriminants::ShiftRight => RiscvAirId::ShiftRight,
            RiscvPolyAirDiscriminants::LoadByte => RiscvAirId::LoadByte,
            RiscvPolyAirDiscriminants::LoadHalf => RiscvAirId::LoadHalf,
            RiscvPolyAirDiscriminants::LoadWord => RiscvAirId::LoadWord,
            RiscvPolyAirDiscriminants::StoreByte => RiscvAirId::StoreByte,
            RiscvPolyAirDiscriminants::StoreHalf => RiscvAirId::StoreHalf,
            RiscvPolyAirDiscriminants::StoreWord => RiscvAirId::StoreWord,
            RiscvPolyAirDiscriminants::AUIPC => RiscvAirId::Auipc,
            RiscvPolyAirDiscriminants::Branch => RiscvAirId::Branch,
            RiscvPolyAirDiscriminants::Jal => RiscvAirId::Jal,
            RiscvPolyAirDiscriminants::Jalr => RiscvAirId::Jalr,
            RiscvPolyAirDiscriminants::SyscallInstrs => RiscvAirId::SyscallInstrs,
            RiscvPolyAirDiscriminants::ByteLookup => RiscvAirId::Byte,
            RiscvPolyAirDiscriminants::MemoryGlobalInit => RiscvAirId::MemoryGlobalInit,
            RiscvPolyAirDiscriminants::MemoryGlobalFinal => RiscvAirId::MemoryGlobalFinalize,
            RiscvPolyAirDiscriminants::MemoryLocal => RiscvAirId::MemoryLocal,
            RiscvPolyAirDiscriminants::SyscallCore => RiscvAirId::SyscallCore,
            RiscvPolyAirDiscriminants::SyscallPrecompile => RiscvAirId::SyscallPrecompile,
            RiscvPolyAirDiscriminants::Global => RiscvAirId::Global,
            RiscvPolyAirDiscriminants::GlobalTileReducer => RiscvAirId::GlobalTileReducer,
            RiscvPolyAirDiscriminants::Sha256Extend => RiscvAirId::ShaExtend,
            RiscvPolyAirDiscriminants::Sha256Compress => RiscvAirId::ShaCompress,
            RiscvPolyAirDiscriminants::Ed25519Add => RiscvAirId::EdAddAssign,
            RiscvPolyAirDiscriminants::Ed25519Decompress => RiscvAirId::EdDecompress,
            RiscvPolyAirDiscriminants::K256Decompress => RiscvAirId::Secp256k1Decompress,
            RiscvPolyAirDiscriminants::P256Decompress => RiscvAirId::Secp256r1Decompress,
            RiscvPolyAirDiscriminants::Secp256k1Add => RiscvAirId::Secp256k1AddAssign,
            RiscvPolyAirDiscriminants::Secp256k1Double => RiscvAirId::Secp256k1DoubleAssign,
            RiscvPolyAirDiscriminants::Secp256r1Add => RiscvAirId::Secp256r1AddAssign,
            RiscvPolyAirDiscriminants::Secp256r1Double => RiscvAirId::Secp256r1DoubleAssign,
            RiscvPolyAirDiscriminants::KeccakP => RiscvAirId::KeccakPermute,
            RiscvPolyAirDiscriminants::Bn254Add => RiscvAirId::Bn254AddAssign,
            RiscvPolyAirDiscriminants::Bn254Double => RiscvAirId::Bn254DoubleAssign,
            RiscvPolyAirDiscriminants::Bls12381Add => RiscvAirId::Bls12381AddAssign,
            RiscvPolyAirDiscriminants::Bls12381Double => RiscvAirId::Bls12381DoubleAssign,
            RiscvPolyAirDiscriminants::Uint256Mul => RiscvAirId::Uint256MulMod,
            RiscvPolyAirDiscriminants::U256x2048Mul => RiscvAirId::U256XU2048Mul,
            RiscvPolyAirDiscriminants::Bls12381Decompress => RiscvAirId::Bls12381Decompress,
            RiscvPolyAirDiscriminants::Bls12381Fp => RiscvAirId::Bls12381FpOpAssign,
            RiscvPolyAirDiscriminants::Bls12381Fp2Mul => RiscvAirId::Bls12381Fp2MulAssign,
            RiscvPolyAirDiscriminants::Bls12381Fp2AddSub => RiscvAirId::Bls12381Fp2AddSubAssign,
            RiscvPolyAirDiscriminants::Bn254Fp => RiscvAirId::Bn254FpOpAssign,
            RiscvPolyAirDiscriminants::Bn254Fp2Mul => RiscvAirId::Bn254Fp2MulAssign,
            RiscvPolyAirDiscriminants::Bn254Fp2AddSub => RiscvAirId::Bn254Fp2AddSubAssign,
            RiscvPolyAirDiscriminants::ShaExtendController => RiscvAirId::ShaExtendController,
            RiscvPolyAirDiscriminants::ShaCompressController => RiscvAirId::ShaCompressController,
            RiscvPolyAirDiscriminants::KeccakController => RiscvAirId::KeccakController,
            RiscvPolyAirDiscriminants::Poseidon2Permute => RiscvAirId::Poseidon2Permute,
        }
    }
}
