use std::{
    fmt::{Display, Formatter, Result as FmtResult},
    str::FromStr,
};

use dt_stark::shape::Shape;
use enum_map::{Enum, EnumMap};
use serde::{Deserialize, Serialize};
use strum::{EnumIter, IntoEnumIterator, IntoStaticStr};
use subenum::subenum;

/// RV32IM AIR Identifiers.
///
/// These identifiers are for the various chips in the rv32im prover. We need them in the
/// executor to compute the memory cost of the current shard of execution.
///
/// The [`CoreAirId`]s are the AIRs that are not part of precompile shards and not the program or
/// byte AIR.
#[subenum(CoreAirId)]
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    Serialize,
    Deserialize,
    EnumIter,
    IntoStaticStr,
    PartialOrd,
    Ord,
    Enum,
)]
pub enum RiscvAirId {
    /// The program chip.
    Program = 1,
    /// The SHA-256 extend chip.
    ShaExtend = 2,
    /// The SHA-256 compress chip.
    ShaCompress = 3,
    /// The Edwards add assign chip.
    EdAddAssign = 4,
    /// The Edwards decompress chip.
    EdDecompress = 5,
    /// The secp256k1 decompress chip.
    Secp256k1Decompress = 6,
    /// The secp256k1 add assign chip.
    Secp256k1AddAssign = 7,
    /// The secp256k1 double assign chip.
    Secp256k1DoubleAssign = 8,
    /// The secp256r1 decompress chip.
    Secp256r1Decompress = 9,
    /// The secp256r1 add assign chip.
    Secp256r1AddAssign = 10,
    /// The secp256r1 double assign chip.
    Secp256r1DoubleAssign = 11,
    /// The Keccak permute chip.
    KeccakPermute = 12,
    /// The bn254 add assign chip.
    Bn254AddAssign = 13,
    /// The bn254 double assign chip.
    Bn254DoubleAssign = 14,
    /// The bls12-381 add assign chip.
    Bls12381AddAssign = 15,
    /// The bls12-381 double assign chip.
    Bls12381DoubleAssign = 16,
    /// The uint256 mul mod chip.
    Uint256MulMod = 17,
    /// The u256 xu2048 mul chip.
    U256XU2048Mul = 18,
    /// The bls12-381 fp op assign chip.
    Bls12381FpOpAssign = 19,
    /// The bls12-831 fp2 add sub assign chip.
    Bls12381Fp2AddSubAssign = 20,
    /// The bls12-831 fp2 mul assign chip.
    Bls12381Fp2MulAssign = 21,
    /// The bn254 fp2 add sub assign chip.
    Bn254FpOpAssign = 22,
    /// The bn254 fp op assign chip.
    Bn254Fp2AddSubAssign = 23,
    /// The bn254 fp2 mul assign chip.
    Bn254Fp2MulAssign = 24,
    /// The bls12-381 decompress chip.
    Bls12381Decompress = 25,
    /// The syscall core chip.
    #[subenum(CoreAirId)]
    SyscallCore = 26,
    /// The syscall precompile chip.
    SyscallPrecompile = 27,
    /// The div rem chip.
    #[subenum(CoreAirId)]
    DivRem = 28,
    /// The bitwise chip.
    #[subenum(CoreAirId)]
    Bitwise = 30,
    /// The mul chip.
    #[subenum(CoreAirId)]
    Mul = 31,
    /// The shift right chip.
    #[subenum(CoreAirId)]
    ShiftRight = 32,
    /// The shift left chip.
    #[subenum(CoreAirId)]
    ShiftLeft = 33,
    /// The lt chip.
    #[subenum(CoreAirId)]
    Lt = 34,
    /// The auipc chip.
    #[subenum(CoreAirId)]
    Auipc = 36,
    /// The branch chip.
    #[subenum(CoreAirId)]
    Branch = 37,
    /// The syscall instructions chip.
    #[subenum(CoreAirId)]
    SyscallInstrs = 39,
    /// The memory global init chip.
    MemoryGlobalInit = 40,
    /// The memory global finalize chip.
    MemoryGlobalFinalize = 41,
    /// The memory local chip.
    #[subenum(CoreAirId)]
    MemoryLocal = 42,
    /// The global chip.
    #[subenum(CoreAirId)]
    Global = 43,
    /// The byte chip.
    Byte = 44,
    ShaExtendController = 45,
    ShaCompressController = 46,
    KeccakController = 47,
    /// The add chip (split from `AddSub`).
    #[subenum(CoreAirId)]
    Add = 48,
    /// The addi chip (split from `AddSub`).
    #[subenum(CoreAirId)]
    Addi = 49,
    /// The sub chip (split from `AddSub`).
    #[subenum(CoreAirId)]
    Sub = 50,
    /// The load byte chip (split from `MemoryInstrs`).
    #[subenum(CoreAirId)]
    LoadByte = 51,
    /// The load half chip (split from `MemoryInstrs`).
    #[subenum(CoreAirId)]
    LoadHalf = 52,
    /// The load word chip (split from `MemoryInstrs`).
    #[subenum(CoreAirId)]
    LoadWord = 53,
    /// The store byte chip (split from `MemoryInstrs`).
    #[subenum(CoreAirId)]
    StoreByte = 54,
    /// The store half chip (split from `MemoryInstrs`).
    #[subenum(CoreAirId)]
    StoreHalf = 55,
    /// The store word chip (split from `MemoryInstrs`).
    #[subenum(CoreAirId)]
    StoreWord = 56,
    /// The jal chip (split from Jump).
    #[subenum(CoreAirId)]
    Jal = 57,
    /// The jalr chip (split from Jump).
    #[subenum(CoreAirId)]
    Jalr = 58,
    /// The Poseidon2 permute precompile chip.
    Poseidon2Permute = 59,
    /// The canonical tile-boundary bridge for Global.
    #[subenum(CoreAirId)]
    GlobalTileReducer = 60,
}

impl RiscvAirId {
    /// Returns the AIRs that are not part of precompile shards and not the program or byte AIR.
    #[must_use]
    pub fn core() -> Vec<RiscvAirId> {
        vec![
            RiscvAirId::Add,
            RiscvAirId::Addi,
            RiscvAirId::Sub,
            RiscvAirId::Mul,
            RiscvAirId::Bitwise,
            RiscvAirId::ShiftLeft,
            RiscvAirId::ShiftRight,
            RiscvAirId::DivRem,
            RiscvAirId::Lt,
            RiscvAirId::LoadByte,
            RiscvAirId::LoadHalf,
            RiscvAirId::LoadWord,
            RiscvAirId::StoreByte,
            RiscvAirId::StoreHalf,
            RiscvAirId::StoreWord,
            RiscvAirId::Auipc,
            RiscvAirId::Branch,
            RiscvAirId::Jal,
            RiscvAirId::Jalr,
            RiscvAirId::MemoryLocal,
            RiscvAirId::SyscallCore,
            RiscvAirId::SyscallInstrs,
            RiscvAirId::Global,
            RiscvAirId::GlobalTileReducer,
        ]
    }

    #[must_use]
    pub fn cpu() -> Vec<RiscvAirId> {
        vec![
            RiscvAirId::Add,
            RiscvAirId::Addi,
            RiscvAirId::Sub,
            RiscvAirId::Mul,
            RiscvAirId::Bitwise,
            RiscvAirId::ShiftLeft,
            RiscvAirId::ShiftRight,
            RiscvAirId::DivRem,
            RiscvAirId::Lt,
            RiscvAirId::LoadByte,
            RiscvAirId::LoadHalf,
            RiscvAirId::LoadWord,
            RiscvAirId::StoreByte,
            RiscvAirId::StoreHalf,
            RiscvAirId::StoreWord,
            RiscvAirId::Auipc,
            RiscvAirId::Branch,
            RiscvAirId::Jal,
            RiscvAirId::Jalr,
            RiscvAirId::SyscallCore,
            RiscvAirId::SyscallInstrs,
        ]
    }

    /// TODO replace these three with subenums or something
    /// Whether the ID represents a core AIR.
    #[must_use]
    pub fn is_core(self) -> bool {
        CoreAirId::try_from(self).is_ok()
    }

    /// Whether the ID represents a memory AIR.
    #[must_use]
    pub fn is_memory(self) -> bool {
        matches!(
            self,
            RiscvAirId::MemoryGlobalInit |
                RiscvAirId::MemoryGlobalFinalize |
                RiscvAirId::Global |
                RiscvAirId::GlobalTileReducer
        )
    }

    /// Whether the ID represents a precompile AIR.
    #[must_use]
    pub fn is_precompile(self) -> bool {
        matches!(
            self,
            RiscvAirId::ShaExtend |
                RiscvAirId::ShaCompress |
                RiscvAirId::EdAddAssign |
                RiscvAirId::EdDecompress |
                RiscvAirId::Secp256k1Decompress |
                RiscvAirId::Secp256k1AddAssign |
                RiscvAirId::Secp256k1DoubleAssign |
                RiscvAirId::Secp256r1Decompress |
                RiscvAirId::Secp256r1AddAssign |
                RiscvAirId::Secp256r1DoubleAssign |
                RiscvAirId::KeccakPermute |
                RiscvAirId::Bn254AddAssign |
                RiscvAirId::Bn254DoubleAssign |
                RiscvAirId::Bls12381AddAssign |
                RiscvAirId::Bls12381DoubleAssign |
                RiscvAirId::Uint256MulMod |
                RiscvAirId::U256XU2048Mul |
                RiscvAirId::Bls12381FpOpAssign |
                RiscvAirId::Bls12381Fp2AddSubAssign |
                RiscvAirId::Bls12381Fp2MulAssign |
                RiscvAirId::Bn254FpOpAssign |
                RiscvAirId::Bn254Fp2AddSubAssign |
                RiscvAirId::Bn254Fp2MulAssign |
                RiscvAirId::Bls12381Decompress
        )
    }

    /// The number of rows in the AIR produced by each event.
    #[must_use]
    pub fn rows_per_event(&self) -> usize {
        match self {
            Self::ShaCompress => 64,
            Self::ShaExtend => 48,
            Self::KeccakPermute => 24,
            _ => 1,
        }
    }

    /// The estimated number of local memory events (`MemoryLocalEvent`) generated per precompile
    /// event. Returns 0 for non-precompile AIRs.
    ///
    /// Each `MemoryLocalEvent` corresponds to one unique memory address touched by
    /// `mr()`/`mw()`/`mr_slice()`/`mw_slice()` in the syscall's `execute()` function.
    /// For add operations where p and q may overlap, we use the worst case (p != q).
    #[must_use]
    pub fn local_mem_events_per_event(&self) -> usize {
        match self {
            // SHA: 8 h reads + 64 w reads + 8 h writes = 72 unique addresses
            Self::ShaCompress => 72,
            // SHA extend: 48 rounds over w[0..64] → 64 unique addresses
            Self::ShaExtend => 64,
            // Keccak: 50 u32 words read + 50 u32 words write = 50 unique addresses
            Self::KeccakPermute => 50,
            // Ed25519/secp256k1/secp256r1/bn254 add: worst case p!=q → 16 reads + 16 writes = 32
            Self::EdAddAssign |
            Self::Secp256k1AddAssign |
            Self::Secp256r1AddAssign |
            Self::Bn254AddAssign => 32,
            // Ed25519 decompress: 8 reads (y) + 8 writes (x) = 16
            // secp256k1/secp256r1/bn254 double: 16 writes = 16
            // secp256k1/secp256r1 decompress: 8 reads + 8 writes = 16
            Self::EdDecompress |
            Self::Secp256k1DoubleAssign |
            Self::Secp256r1DoubleAssign |
            Self::Bn254DoubleAssign |
            Self::Secp256k1Decompress |
            Self::Secp256r1Decompress => 16,
            // bls12381 add: 24 reads + 24 writes = 48 (worst case p!=q)
            Self::Bls12381AddAssign => 48,
            // bls12381 double: 24 writes = 24
            Self::Bls12381DoubleAssign => 24,
            // bls12381 decompress: 12 reads + 12 writes = 24
            Self::Bls12381Decompress => 24,
            // uint256 mul: 8 reads (y) + 8 reads (mod) + 8 writes (x) = 24
            Self::Uint256MulMod => 24,
            // u256x2048 mul: 2 reg reads + 8 reads (a) + 64 reads (b) + 64 writes (lo) + 8 writes
            // (hi) = 146
            Self::U256XU2048Mul => 146,
            // bn254 fp: 8 reads + 8 writes = 16
            Self::Bn254FpOpAssign => 16,
            // bls12381 fp: 12 reads + 12 writes = 24
            Self::Bls12381FpOpAssign => 24,
            // bn254 fp2 add/sub/mul: 16 reads + 16 writes = 32
            Self::Bn254Fp2AddSubAssign | Self::Bn254Fp2MulAssign => 32,
            // bls12381 fp2 add/sub/mul: 24 reads + 24 writes = 48
            Self::Bls12381Fp2AddSubAssign | Self::Bls12381Fp2MulAssign => 48,
            // Poseidon2: 24 reads + 24 writes = 48, but state is in-place so 24 unique addresses
            Self::Poseidon2Permute => 24,
            // Non-precompile AIRs.
            _ => 0,
        }
    }

    /// Static maximum number of precompile events per precompile shard.
    ///
    /// Computed from `PRECOMPILE_SHARD_CELLS_THRESHOLD` using raw (unpadded) cells:
    /// `N = min(area_limit, height_limit)` where
    /// `area_limit = (budget - byte_overhead - program_overhead) / raw_cells_per_event`
    /// and `height_limit = SHARD_HEIGHT_THRESHOLD / max_height_per_event`.
    ///
    /// Returns 0 for non-precompile AIRs.
    #[must_use]
    pub fn max_precompile_shard_events(&self) -> usize {
        (match self {
            Self::ShaCompress => 12_380,
            Self::ShaExtend => 14_832,
            Self::KeccakPermute => 8_124,
            Self::EdAddAssign => 30_372,
            Self::EdDecompress => 57_934,
            Self::Secp256k1AddAssign | Self::Secp256r1AddAssign | Self::Bn254AddAssign => 30_510,
            Self::Secp256k1DoubleAssign | Self::Secp256r1DoubleAssign | Self::Bn254DoubleAssign => {
                57_660
            }
            Self::Secp256k1Decompress | Self::Secp256r1Decompress => 58_998,
            Self::Bls12381AddAssign => 20_439,
            Self::Bls12381DoubleAssign => 38_796,
            Self::Bls12381Decompress => 39_605,
            Self::Uint256MulMod => 41_675,
            Self::U256XU2048Mul => 6_961,
            Self::Bls12381FpOpAssign => 41_471,
            Self::Bls12381Fp2AddSubAssign => 21_051,
            Self::Bls12381Fp2MulAssign => 20_641,
            Self::Bn254FpOpAssign => 61_584,
            Self::Bn254Fp2AddSubAssign => 31_415,
            Self::Bn254Fp2MulAssign => 30_809,
            Self::Poseidon2Permute => 40_292,
            _ => 0,
        })
    }

    /// Returns the string representation of the AIR.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        self.into()
    }
}

impl FromStr for RiscvAirId {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let air = Self::iter().find(|chip| chip.as_str() == s);
        match air {
            Some(air) => Ok(air),
            None => Err(format!("Invalid RV32IMAir: {s}")),
        }
    }
}

impl Display for RiscvAirId {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "{}", self.as_str())
    }
}

/// Defines a set of maximal shapes for generating core proofs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaximalShapes {
    inner: Vec<EnumMap<CoreAirId, u32>>,
}

impl FromIterator<Shape<RiscvAirId>> for MaximalShapes {
    fn from_iter<T: IntoIterator<Item = Shape<RiscvAirId>>>(iter: T) -> Self {
        let mut maximal_shapes = Vec::new();
        for shape in iter {
            let mut maximal_shape = EnumMap::<CoreAirId, u32>::default();
            for (air, height) in shape {
                if let Ok(core_air) = CoreAirId::try_from(air) {
                    maximal_shape[core_air] = height as u32;
                } else if air != RiscvAirId::Program && air != RiscvAirId::Byte {
                    tracing::warn!("Invalid core air: {air}");
                }
            }
            maximal_shapes.push(maximal_shape);
        }
        Self { inner: maximal_shapes }
    }
}

impl MaximalShapes {
    /// Returns an iterator over the maximal shapes.
    pub fn iter(&self) -> impl Iterator<Item = &EnumMap<CoreAirId, u32>> {
        self.inner.iter()
    }
}
