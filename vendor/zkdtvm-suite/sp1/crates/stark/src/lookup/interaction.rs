use core::fmt::{Debug, Display};

use p3_air::VirtualPairCol;
use p3_field::{ExtensionField, Field};

use crate::air::InteractionScope;

/// How a lookup payload is interpreted before beta compression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InteractionValueEncoding {
    /// Each entry in `values` is one base-field lookup value.
    Base,
    /// Consecutive base limbs are coefficients of extension-field blocks.
    ExtensionBlocks { degree: usize },
}

impl InteractionValueEncoding {
    /// Number of beta-weighted values after applying this encoding.
    #[must_use]
    pub const fn encoded_len(self, base_values: usize) -> usize {
        match self {
            Self::Base => base_values,
            Self::ExtensionBlocks { degree } => {
                assert!(degree > 1, "extension-block encoding degree must exceed one");
                assert!(base_values > 0, "extension-block payload must not be empty");
                base_values.div_ceil(degree)
            }
        }
    }

    /// Validate encoding metadata at the framework boundary.
    pub fn validate(self, base_values: usize) {
        if let Self::ExtensionBlocks { degree } = self {
            assert!(degree > 1, "extension-block encoding degree must exceed one");
            assert!(base_values > 0, "extension-block payload must not be empty");
        }
    }
}

/// An interaction for a lookup or a permutation argument.
#[derive(Clone)]
pub struct Interaction<F: Field> {
    /// The values of the interaction.
    pub values: Vec<VirtualPairCol<F>>,
    /// Typed interpretation of `values` before beta compression.
    pub value_encoding: InteractionValueEncoding,
    /// The multiplicity of the interaction.
    pub multiplicity: VirtualPairCol<F>,
    /// The kind of interaction.
    pub kind: InteractionKind,
    /// The scope of the interaction.
    pub scope: InteractionScope,
}

impl<F: Field> Interaction<F> {
    /// Converts this interaction to an interaction over an extension field.
    pub fn field_ext<EF: ExtensionField<F>>(&self) -> Interaction<EF> {
        Interaction {
            values: self
                .values
                .iter()
                .map(|v| VirtualPairCol {
                    column_weights: v
                        .column_weights
                        .iter()
                        .map(|(c, w)| (*c, (*w).into()))
                        .collect::<Vec<_>>(),
                    constant: v.constant.into(),
                })
                .collect(),
            value_encoding: self.value_encoding,
            multiplicity: VirtualPairCol {
                column_weights: self
                    .multiplicity
                    .column_weights
                    .iter()
                    .map(|(c, w)| (*c, (*w).into()))
                    .collect::<Vec<_>>(),
                constant: self.multiplicity.constant.into(),
            },
            kind: self.kind,
            scope: self.scope,
        }
    }
}

/// The type of interaction for a lookup argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum InteractionKind {
    /// Interaction with the memory table, such as read and write.
    Memory = 1,

    /// Interaction with the program table, loading an instruction at a given pc address.
    Program = 2,

    /// Interaction with instruction oracle.
    Instruction = 3,

    /// Interaction with the ALU operations.
    Alu = 4,

    /// Interaction with the byte lookup table for byte operations.
    Byte = 5,

    /// Requesting a range check for a given value and range.
    Range = 6,

    /// Interaction with the field op table for field operations.
    Field = 7,

    /// Interaction with a syscall.
    Syscall = 8,

    /// Interaction with the global table.
    Global = 9,

    /// Interaction with the `ShaExtend` chip.
    ShaExtend = 10,

    /// Interaction with the `ShaCompress` chip.
    ShaCompress = 11,

    /// Interaction with the `Keccak` chip.
    Keccak = 12,

    /// Interaction with the cpu state
    State = 13,

    /// Interaction for linking consecutive addresses in `MemoryGlobal` chips.
    MemoryGlobalAddr = 14,

    /// Interaction with the bit vector lookup table for bit decomposition.
    BitVec = 16,

    /// Frozen domain for the indexed simple-projective Global chain.
    /// Its 34 base values are packed into seven quintic-extension blocks.
    GlobalProjectiveChainV2 = 17,

    /// Interaction domain for zkDTVM recursion interactions.
    ///
    /// This is intentionally excluded from [`InteractionKind::all_kinds`] so
    /// existing core machines keep using the legacy interaction set. Native
    /// recursion verifier/prover paths must opt into
    /// [`InteractionKind::all_recursion_kinds`] or another explicit active set.
    Recursion = 64,
}

/// Active interaction kinds for native-recursion machines.
pub static RECURSION_INTERACTION_KINDS: [InteractionKind; 1] = [InteractionKind::Recursion];

impl InteractionKind {
    /// Decode a stable protocol interaction ID without reusing retired IDs.
    #[must_use]
    pub const fn from_protocol_id(id: u32) -> Option<Self> {
        match id {
            1 => Some(Self::Memory),
            2 => Some(Self::Program),
            3 => Some(Self::Instruction),
            4 => Some(Self::Alu),
            5 => Some(Self::Byte),
            6 => Some(Self::Range),
            7 => Some(Self::Field),
            8 => Some(Self::Syscall),
            9 => Some(Self::Global),
            10 => Some(Self::ShaExtend),
            11 => Some(Self::ShaCompress),
            12 => Some(Self::Keccak),
            13 => Some(Self::State),
            14 => Some(Self::MemoryGlobalAddr),
            // 15 is permanently retired.
            16 => Some(Self::BitVec),
            17 => Some(Self::GlobalProjectiveChainV2),
            64 => Some(Self::Recursion),
            _ => None,
        }
    }

    /// Returns all legacy/core kinds of interactions.
    #[must_use]
    pub fn all_kinds() -> Vec<InteractionKind> {
        vec![
            InteractionKind::Memory,
            InteractionKind::Program,
            InteractionKind::Instruction,
            InteractionKind::Alu,
            InteractionKind::Byte,
            InteractionKind::Range,
            InteractionKind::Field,
            InteractionKind::Syscall,
            InteractionKind::Global,
            InteractionKind::ShaExtend,
            InteractionKind::ShaCompress,
            InteractionKind::Keccak,
            InteractionKind::State,
            InteractionKind::MemoryGlobalAddr,
            InteractionKind::BitVec,
            InteractionKind::GlobalProjectiveChainV2,
        ]
    }

    /// Returns native-recursion-only interaction kinds.
    #[must_use]
    pub fn all_recursion_kinds() -> Vec<InteractionKind> {
        Self::recursion_kinds().to_vec()
    }

    /// Returns interaction kinds owned by the canonical projective Global AIR.
    #[must_use]
    pub const fn global_projective_kinds() -> &'static [InteractionKind] {
        &[InteractionKind::GlobalProjectiveChainV2]
    }

    /// Returns native-recursion-only interaction kinds without allocation.
    #[must_use]
    pub fn recursion_kinds() -> &'static [InteractionKind] {
        &RECURSION_INTERACTION_KINDS
    }

    /// Returns every currently known interaction kind.
    ///
    /// Use this for diagnostics only. Prover/verifier soundness checks should
    /// use the active kind set for the machine being verified.
    #[must_use]
    pub fn all_known_kinds() -> Vec<InteractionKind> {
        let mut kinds = Self::all_kinds();
        kinds.extend(Self::all_recursion_kinds());
        kinds
    }
}

impl<F: Field> Interaction<F> {
    /// Create a new interaction.
    pub const fn new(
        values: Vec<VirtualPairCol<F>>,
        multiplicity: VirtualPairCol<F>,
        kind: InteractionKind,
        scope: InteractionScope,
    ) -> Self {
        Self { values, value_encoding: InteractionValueEncoding::Base, multiplicity, kind, scope }
    }

    /// Create an interaction with an explicit typed payload encoding.
    #[must_use]
    pub fn new_with_encoding(
        values: Vec<VirtualPairCol<F>>,
        value_encoding: InteractionValueEncoding,
        multiplicity: VirtualPairCol<F>,
        kind: InteractionKind,
        scope: InteractionScope,
    ) -> Self {
        value_encoding.validate(values.len());
        Self { values, value_encoding, multiplicity, kind, scope }
    }

    /// The index of the argument in the lookup table.
    pub const fn argument_index(&self) -> usize {
        self.kind as usize
    }
}

impl<F: Field> Debug for Interaction<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Interaction")
            .field("kind", &self.kind)
            .field("scope", &self.scope)
            .field("value_encoding", &self.value_encoding)
            .field("base_values", &self.values.len())
            .finish_non_exhaustive()
    }
}

impl Display for InteractionKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InteractionKind::Memory => write!(f, "Memory"),
            InteractionKind::Program => write!(f, "Program"),
            InteractionKind::Instruction => write!(f, "Instruction"),
            InteractionKind::Alu => write!(f, "Alu"),
            InteractionKind::Byte => write!(f, "Byte"),
            InteractionKind::Range => write!(f, "Range"),
            InteractionKind::Field => write!(f, "Field"),
            InteractionKind::Syscall => write!(f, "Syscall"),
            InteractionKind::Global => write!(f, "Global"),
            InteractionKind::ShaExtend => write!(f, "ShaExtend"),
            InteractionKind::ShaCompress => write!(f, "ShaCompress"),
            InteractionKind::Keccak => write!(f, "Keccak"),
            InteractionKind::State => write!(f, "State"),
            InteractionKind::MemoryGlobalAddr => write!(f, "MemoryGlobalAddr"),
            InteractionKind::BitVec => write!(f, "BitVec"),
            InteractionKind::GlobalProjectiveChainV2 => {
                write!(f, "GlobalProjectiveChainV2")
            }
            InteractionKind::Recursion => write!(f, "Recursion"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::InteractionKind;

    #[test]
    fn interaction_kind_sets_keep_recursion_separate() {
        assert!(!InteractionKind::all_kinds().contains(&InteractionKind::Recursion));
        assert_eq!(InteractionKind::all_recursion_kinds(), vec![InteractionKind::Recursion]);
        assert_eq!(InteractionKind::recursion_kinds(), &[InteractionKind::Recursion]);
        assert!(InteractionKind::all_known_kinds().contains(&InteractionKind::Recursion));
        assert!(InteractionKind::all_kinds().contains(&InteractionKind::GlobalProjectiveChainV2));
        assert_eq!(InteractionKind::from_protocol_id(15), None);
        assert_eq!(
            InteractionKind::from_protocol_id(17),
            Some(InteractionKind::GlobalProjectiveChainV2)
        );
    }

    #[test]
    fn recursion_argument_index_is_stable() {
        assert_eq!(InteractionKind::Recursion as usize, 64);
    }
}
