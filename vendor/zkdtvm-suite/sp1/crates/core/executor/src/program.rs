//! Programs that can be executed by the zkDTVM.

use std::{
    fs::File,
    io::Read,
    str::FromStr,
    sync::{Arc, RwLock},
};

use crate::{
    disassembler::{transpile, Elf},
    instruction::Instruction,
    RiscvAirId,
};
#[cfg(all(feature = "koalabear", test))]
use dt_stark::global_d11::construct_map_reference;
#[cfg(feature = "koalabear")]
pub use dt_stark::global_d11::ProgramImageBoundaryV1;
#[cfg(feature = "koalabear")]
use dt_stark::global_d11::{
    construct_map, pack_unsigned, D11AffinePointV1, D11ProjectivePointV1, GlobalMapErrorV1,
    GlobalPackInputV1, ProjectivePointError, D11_PROJECTIVE_228_QDELTA_WIRE_ID,
    PARAMETER_MANIFEST_SHA256,
};
use dt_stark::{
    air::{MachineAir, MachineProgram},
    shape::Shape,
    InteractionKind,
};
use hashbrown::HashMap;
#[cfg(feature = "babybear")]
use p3_field::AbstractExtensionField;
use p3_field::Field;
#[cfg(feature = "koalabear")]
use p3_koala_bear::KoalaBear;
use p3_maybe_rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
#[cfg(feature = "koalabear")]
use tiny_keccak::{Hasher, Keccak};

/// A program that can be executed by the zkDTVM.
///
/// Contains a series of instructions along with the initial memory image. It also contains the
/// start address and base address of the program.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Program {
    /// The instructions of the program.
    pub instructions: Vec<Instruction>,
    /// The start address of the program.
    pub pc_start: u32,
    /// The base address of the program.
    pub pc_base: u32,
    /// The initial memory image, useful for global constants.
    memory_image: HashMap<u32, u32>,
    /// The shape for the preprocessed tables.
    pub preprocessed_shape: Option<Shape<RiscvAirId>>,
    /// Monotonic identity generation for the mutable construction phase.
    #[cfg(feature = "koalabear")]
    #[serde(skip, default)]
    memory_image_generation: u64,
    /// The immutable, versioned Global program-image authority.
    #[cfg(feature = "koalabear")]
    #[serde(skip, default)]
    prepared_global_program: RwLock<Option<Arc<PreparedGlobalProgram>>>,
}

/// Immutable, identity-checked program-image authority used by Global preparation.
#[cfg(feature = "koalabear")]
#[derive(Debug, Eq, PartialEq)]
pub struct PreparedGlobalProgram {
    generation: u64,
    image_identity: ProgramImageIdentity,
    entries: Vec<ProgramMapWitness>,
    initial_boundary: ProgramImageBoundaryV1<u32>,
    boundary_digest: [u8; 32],
}

#[cfg(feature = "koalabear")]
const PROGRAM_IMAGE_IDENTITY_DOMAIN_V1: &[u8] = b"dt-global-program-image-v1\0";
#[cfg(feature = "koalabear")]
const PROGRAM_BOUNDARY_DOMAIN_V1: &[u8] = b"dt-global-program-boundary-v1\0";

#[cfg(feature = "koalabear")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProgramMapWitness {
    pub addr: u32,
    pub word: u32,
    pub tweak: u16,
    pub canonical_y: [u32; 11],
}

#[cfg(feature = "koalabear")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProgramImageIdentity {
    pub parameter_manifest_digest: [u8; 32],
    pub word_count: usize,
    pub ordered_image_digest: [u8; 32],
}

#[cfg(feature = "koalabear")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PrepareGlobalProgramError {
    Map { addr: u32, word: u32, cause: GlobalMapErrorV1 },
    InvalidBoundary(ProjectivePointError),
}

#[cfg(feature = "koalabear")]
impl PreparedGlobalProgram {
    fn matches_generation(&self, generation: u64) -> bool {
        self.generation == generation &&
            self.image_identity.parameter_manifest_digest == PARAMETER_MANIFEST_SHA256
    }

    pub const fn image_identity(&self) -> &ProgramImageIdentity {
        &self.image_identity
    }

    pub fn entries(&self) -> &[ProgramMapWitness] {
        &self.entries
    }

    #[must_use]
    pub const fn initial_boundary(&self) -> &ProgramImageBoundaryV1<u32> {
        &self.initial_boundary
    }

    #[must_use]
    pub const fn boundary_digest(&self) -> [u8; 32] {
        self.boundary_digest
    }

    #[must_use]
    pub fn entry(&self, addr: u32) -> Option<ProgramMapWitness> {
        self.entries
            .binary_search_by_key(&addr, |entry| entry.addr)
            .ok()
            .map(|index| self.entries[index])
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(feature = "koalabear")]
impl ProgramMapWitness {
    #[must_use]
    pub fn unsigned_point<F: p3_field::PrimeField32>(&self) -> D11AffinePointV1<F> {
        let input = GlobalPackInputV1 {
            message: [
                0,
                0,
                self.addr,
                self.word & 255,
                (self.word >> 8) & 255,
                (self.word >> 16) & 255,
                (self.word >> 24) & 255,
            ],
            kind: InteractionKind::Memory as u8,
        };
        let x = pack_unsigned::<F>(input, self.tweak)
            .expect("admitted Global program witness must remain in PackV1 domain")
            .to_d11();
        let point = D11AffinePointV1 {
            x,
            y: dt_stark::global_d11::D11::from_canonical_u32(self.canonical_y),
        };
        debug_assert!(point.is_on_curve());
        point
    }
}

impl Clone for Program {
    fn clone(&self) -> Self {
        #[cfg(feature = "koalabear")]
        let prepared_global_program = self
            .prepared_global_program
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        Self {
            instructions: self.instructions.clone(),
            pc_start: self.pc_start,
            pc_base: self.pc_base,
            memory_image: self.memory_image.clone(),
            preprocessed_shape: self.preprocessed_shape.clone(),
            #[cfg(feature = "koalabear")]
            memory_image_generation: self.memory_image_generation,
            #[cfg(feature = "koalabear")]
            prepared_global_program: RwLock::new(prepared_global_program),
        }
    }
}

impl Program {
    /// Create a new [Program].
    #[must_use]
    pub fn new(instructions: Vec<Instruction>, pc_start: u32, pc_base: u32) -> Self {
        Self {
            instructions,
            pc_start,
            pc_base,
            memory_image: HashMap::new(),
            preprocessed_shape: None,
            #[cfg(feature = "koalabear")]
            memory_image_generation: 0,
            #[cfg(feature = "koalabear")]
            prepared_global_program: RwLock::default(),
        }
    }

    /// Disassemble a RV32IM ELF to a program that be executed by the VM.
    ///
    /// # Errors
    ///
    /// This function may return an error if the ELF is not valid.
    pub fn from(input: &[u8]) -> eyre::Result<Self> {
        // Decode the bytes as an ELF.
        let elf = Elf::decode(input)?;

        // Transpile the RV32IM instructions.
        let instructions = transpile(&elf.instructions);

        // Return the program.
        Ok(Program {
            instructions,
            pc_start: elf.pc_start,
            pc_base: elf.pc_base,
            memory_image: elf.memory_image,
            preprocessed_shape: None,
            #[cfg(feature = "koalabear")]
            memory_image_generation: 0,
            #[cfg(feature = "koalabear")]
            prepared_global_program: RwLock::default(),
        })
    }

    /// Disassemble a RV32IM ELF to a program that be executed by the VM from a file path.
    ///
    /// # Errors
    ///
    /// This function will return an error if the file cannot be opened or read.
    pub fn from_elf(path: &str) -> eyre::Result<Self> {
        let mut elf_code = Vec::new();
        File::open(path)?.read_to_end(&mut elf_code)?;
        Program::from(&elf_code)
    }

    /// Read the immutable program memory image.
    #[must_use]
    pub const fn memory_image(&self) -> &HashMap<u32, u32> {
        &self.memory_image
    }

    /// Mutate the program memory image during construction and invalidate the prepared authority.
    pub fn memory_image_mut(&mut self) -> &mut HashMap<u32, u32> {
        #[cfg(feature = "koalabear")]
        {
            self.memory_image_generation = self
                .memory_image_generation
                .checked_add(1)
                .expect("program memory-image generation exhausted");
            *self
                .prepared_global_program
                .get_mut()
                .unwrap_or_else(|poisoned| poisoned.into_inner()) = None;
        }
        &mut self.memory_image
    }

    /// Custom logic for padding the trace to a power of two according to the proof shape.
    pub fn fixed_log2_rows<F: Field, A: MachineAir<F>>(&self, air: &A) -> Option<usize> {
        let id = RiscvAirId::from_str(&air.name()).unwrap();
        self.preprocessed_shape.as_ref().map(|shape| {
            shape
                .log2_height(&id)
                .unwrap_or_else(|| panic!("Chip {} not found in specified shape", air.name()))
        })
    }

    #[must_use]
    /// Fetch the instruction at the given program counter.
    pub fn fetch(&self, pc: u32) -> &Instruction {
        let idx = ((pc - self.pc_base) / 4) as usize;
        &self.instructions[idx]
    }

    /// Build a fresh address-sorted program-image authority without publishing it to the cache.
    #[cfg(feature = "koalabear")]
    pub fn build_prepared_global_program(
        &self,
    ) -> Result<PreparedGlobalProgram, PrepareGlobalProgramError> {
        let mut memory_words =
            self.memory_image.iter().map(|(&addr, &word)| (addr, word)).collect::<Vec<_>>();
        memory_words.sort_unstable_by_key(|&(addr, _)| addr);
        let entries = memory_words
            .par_iter()
            .map(|&(addr, word)| {
                let input = GlobalPackInputV1 {
                    message: [
                        0,
                        0,
                        addr,
                        word & 255,
                        (word >> 8) & 255,
                        (word >> 16) & 255,
                        (word >> 24) & 255,
                    ],
                    kind: InteractionKind::Memory as u8,
                };
                construct_map::<KoalaBear>(input, true)
                    .map(|mapped| ProgramMapWitness {
                        addr,
                        word,
                        tweak: mapped.witness.tweak,
                        canonical_y: mapped.witness.canonical_y,
                    })
                    .map_err(|cause| PrepareGlobalProgramError::Map { addr, word, cause })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut initial = D11ProjectivePointV1::<KoalaBear>::identity();
        for entry in &entries {
            initial = initial
                .add_complete(&entry.unsigned_point::<KoalaBear>().negated().to_projective());
        }
        let initial_boundary = canonical_program_boundary(initial)
            .map_err(PrepareGlobalProgramError::InvalidBoundary)?;
        let image_identity = ProgramImageIdentity {
            parameter_manifest_digest: PARAMETER_MANIFEST_SHA256,
            word_count: entries.len(),
            ordered_image_digest: ordered_program_image_digest(&entries),
        };
        let boundary_digest = program_boundary_digest(&image_identity, &initial_boundary);
        Ok(PreparedGlobalProgram {
            generation: self.memory_image_generation,
            image_identity,
            entries,
            initial_boundary,
            boundary_digest,
        })
    }

    /// Return the immutable session-local Global program authority.
    #[cfg(feature = "koalabear")]
    pub fn prepared_global_program(
        &self,
    ) -> Result<Arc<PreparedGlobalProgram>, PrepareGlobalProgramError> {
        if let Some(prepared) = self
            .prepared_global_program
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_ref()
            .filter(|prepared| prepared.matches_generation(self.memory_image_generation))
        {
            return Ok(Arc::clone(prepared));
        }

        let prepared = Arc::new(self.build_prepared_global_program()?);
        let mut cache =
            self.prepared_global_program.write().unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(existing) = cache
            .as_ref()
            .filter(|existing| existing.matches_generation(self.memory_image_generation))
        {
            return Ok(Arc::clone(existing));
        }
        *cache = Some(Arc::clone(&prepared));
        Ok(prepared)
    }
}

#[cfg(feature = "koalabear")]
fn canonical_program_boundary(
    point: D11ProjectivePointV1<KoalaBear>,
) -> Result<ProgramImageBoundaryV1<u32>, ProjectivePointError> {
    Ok(match point.to_affine()? {
        None => ProgramImageBoundaryV1::Infinity,
        Some(affine) => ProgramImageBoundaryV1::Affine {
            x: affine.x.to_canonical_u32(),
            y: affine.y.to_canonical_u32(),
        },
    })
}

#[cfg(feature = "koalabear")]
fn ordered_program_image_digest(entries: &[ProgramMapWitness]) -> [u8; 32] {
    let mut hasher = Keccak::v256();
    hasher.update(PROGRAM_IMAGE_IDENTITY_DOMAIN_V1);
    hasher.update(&[D11_PROJECTIVE_228_QDELTA_WIRE_ID]);
    hasher.update(&PARAMETER_MANIFEST_SHA256);
    hasher.update(&(entries.len() as u64).to_le_bytes());
    for entry in entries {
        hasher.update(&entry.addr.to_le_bytes());
        hasher.update(&entry.word.to_le_bytes());
    }
    let mut digest = [0u8; 32];
    hasher.finalize(&mut digest);
    digest
}

#[cfg(feature = "koalabear")]
fn program_boundary_digest(
    identity: &ProgramImageIdentity,
    boundary: &ProgramImageBoundaryV1<u32>,
) -> [u8; 32] {
    let mut hasher = Keccak::v256();
    hasher.update(PROGRAM_BOUNDARY_DOMAIN_V1);
    hasher.update(&[D11_PROJECTIVE_228_QDELTA_WIRE_ID]);
    hasher.update(&identity.parameter_manifest_digest);
    hasher.update(&(identity.word_count as u64).to_le_bytes());
    hasher.update(&identity.ordered_image_digest);
    match boundary {
        ProgramImageBoundaryV1::Infinity => hasher.update(&[0]),
        ProgramImageBoundaryV1::Affine { x, y } => {
            hasher.update(&[1]);
            for coefficient in x.iter().chain(y) {
                hasher.update(&coefficient.to_le_bytes());
            }
        }
    }
    let mut digest = [0u8; 32];
    hasher.finalize(&mut digest);
    digest
}

impl<F: Field> MachineProgram<F> for Program {
    fn pc_start(&self) -> F {
        F::from_canonical_u32(self.pc_start)
    }

    #[cfg(feature = "koalabear")]
    fn initial_global_boundary(&self) -> Option<ProgramImageBoundaryV1<u32>> {
        self.prepared_global_program().ok().map(|prepared| *prepared.initial_boundary())
    }

}

#[cfg(all(test, feature = "koalabear"))]
mod tests {
    use super::*;

    #[test]
    fn program_artifact_is_sorted_shared_and_recomputable() {
        let mut program = Program::default();
        for (addr, word) in [(0x140, 0x1122_3344), (0x100, 0x5566_7788), (0x120, 7)] {
            program.memory_image_mut().insert(addr, word);
        }
        let prepared = program.prepared_global_program().unwrap();
        assert!(prepared.entries.windows(2).all(|pair| pair[0].addr < pair[1].addr));
        assert_eq!(prepared.len(), 3);
        assert_eq!(prepared.image_identity().parameter_manifest_digest, PARAMETER_MANIFEST_SHA256);
        assert!(Arc::ptr_eq(&prepared, &program.prepared_global_program().unwrap()));

        let shared_clone = program.clone();
        assert!(Arc::ptr_eq(&prepared, &shared_clone.prepared_global_program().unwrap()));
        let recomputed = program.build_prepared_global_program().unwrap();
        assert_eq!(prepared.as_ref(), &recomputed);
        assert_eq!(std::mem::size_of::<ProgramMapWitness>(), 56);
    }

    #[test]
    fn program_boundary_cancels_all_unchanged_final_points() {
        let mut program = Program::default();
        for (addr, word) in [(0x100, 0x1122_3344), (0x104, 0), (0x108, u32::MAX)] {
            program.memory_image_mut().insert(addr, word);
        }
        let prepared = program.prepared_global_program().unwrap();
        let mut sum = match prepared.initial_boundary() {
            ProgramImageBoundaryV1::Infinity => D11ProjectivePointV1::<KoalaBear>::identity(),
            ProgramImageBoundaryV1::Affine { x, y } => D11AffinePointV1 {
                x: dt_stark::global_d11::D11::from_canonical_u32(*x),
                y: dt_stark::global_d11::D11::from_canonical_u32(*y),
            }
            .to_projective(),
        };
        for entry in prepared.entries() {
            let fresh = construct_map_reference::<KoalaBear>(
                GlobalPackInputV1 {
                    message: [
                        0,
                        0,
                        entry.addr,
                        entry.word & 255,
                        (entry.word >> 8) & 255,
                        (entry.word >> 16) & 255,
                        (entry.word >> 24) & 255,
                    ],
                    kind: InteractionKind::Memory as u8,
                },
                true,
            )
            .unwrap();
            assert_eq!(entry.tweak, fresh.witness.tweak);
            assert_eq!(entry.canonical_y, fresh.witness.canonical_y);
            sum = sum.add_complete(&entry.unsigned_point().to_projective());
        }
        assert!(sum.is_identity());
        assert!(!sum.is_zero_triple());
    }

    #[test]
    fn program_generation_invalidates_without_scanning_warm_hits() {
        let mut program = Program::default();
        program.memory_image_mut().insert(0x100, 7);
        let first = program.prepared_global_program().unwrap();
        assert!(Arc::ptr_eq(&first, &program.prepared_global_program().unwrap()));

        program.memory_image_mut().insert(0x104, 13);
        let second = program.prepared_global_program().unwrap();
        assert!(!Arc::ptr_eq(&first, &second));
        assert_eq!(second.entry(0x104).unwrap().word, 13);
    }

    #[test]
    fn empty_program_has_canonical_infinity_boundary() {
        let prepared = Program::default().prepared_global_program().unwrap();
        assert!(prepared.is_empty());
        assert_eq!(prepared.initial_boundary(), &ProgramImageBoundaryV1::Infinity);
        assert_ne!(prepared.boundary_digest(), [0; 32]);
    }
}
