use crate::lookup::{InteractionKind, InteractionValueEncoding};

/// An interaction is a cross-table lookup.
pub struct AirInteraction<E> {
    /// The values of the interaction.
    pub values: Vec<E>,
    /// Typed interpretation of `values` before beta compression.
    pub value_encoding: InteractionValueEncoding,
    /// The multiplicity of the interaction.
    pub multiplicity: E,
    /// The kind of interaction.
    pub kind: InteractionKind,
}

impl<E> AirInteraction<E> {
    /// Create a new [`AirInteraction`].
    pub const fn new(values: Vec<E>, multiplicity: E, kind: InteractionKind) -> Self {
        Self { values, value_encoding: InteractionValueEncoding::Base, multiplicity, kind }
    }

    /// Create an interaction whose consecutive base values are packed into
    /// extension-field blocks before applying beta powers.
    #[must_use]
    pub fn new_extension_blocks(
        values: Vec<E>,
        degree: usize,
        multiplicity: E,
        kind: InteractionKind,
    ) -> Self {
        let value_encoding = InteractionValueEncoding::ExtensionBlocks { degree };
        value_encoding.validate(values.len());
        Self { values, value_encoding, multiplicity, kind }
    }
}
