use dt_stark::{air::FullAirBuilder, InteractionKind};
use p3_field::AbstractField;

use crate::{
    interaction::RecursionInteractionIndexSpace,
    interaction_registry_dt::RecursionInteractionSchema,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecursionFullAirBus {
    schema: RecursionInteractionSchema,
}

impl RecursionFullAirBus {
    pub const fn new(schema: RecursionInteractionSchema) -> Self {
        Self { schema }
    }

    pub const fn schema(&self) -> RecursionInteractionSchema {
        self.schema
    }

    pub const fn payload_arity(&self) -> usize {
        self.schema.payload_arity
    }

    pub const fn denominator_value_count(&self) -> usize {
        self.schema.denominator_value_count()
    }

    pub const fn required_max_beta_power_floor(&self) -> usize {
        self.schema.required_max_beta_power_floor()
    }

    pub fn denominator<AB, E>(
        &self,
        builder: &AB,
        payload: impl IntoIterator<Item = E>,
    ) -> AB::VarExt
    where
        AB: FullAirBuilder,
        E: Into<AB::VarMaybeExt>,
    {
        assert_eq!(
            self.schema.index_space,
            RecursionInteractionIndexSpace::Global,
            "global recursion interaction bus cannot include proof_idx"
        );
        let values = self.lower_values::<AB, E, AB::VarMaybeExt>(None, payload);
        builder.lookup_denominator(recursion_interaction_kind::<AB>(), values)
    }

    pub fn denominator_for_proof<AB, P, E>(
        &self,
        builder: &AB,
        proof_idx: P,
        payload: impl IntoIterator<Item = E>,
    ) -> AB::VarExt
    where
        AB: FullAirBuilder,
        P: Into<AB::VarMaybeExt>,
        E: Into<AB::VarMaybeExt>,
    {
        assert_eq!(
            self.schema.index_space,
            RecursionInteractionIndexSpace::PerProof,
            "per-proof recursion interaction bus must include proof_idx"
        );
        let values = self.lower_values::<AB, E, P>(Some(proof_idx), payload);
        builder.lookup_denominator(recursion_interaction_kind::<AB>(), values)
    }

    /// Compute a per-proof lookup denominator whose payload is already grouped into
    /// extension-field blocks.  The interaction id and proof index remain separate
    /// blocks, so this has the same domain separation as `denominator_for_proof` while
    /// advancing beta only once per packed payload block.
    pub fn denominator_ext_blocks_for_proof<AB>(
        &self,
        builder: &AB,
        proof_idx: AB::VarMaybeExt,
        payload: impl IntoIterator<Item = AB::VarExt>,
    ) -> AB::VarExt
    where
        AB: FullAirBuilder,
    {
        assert_eq!(
            self.schema.index_space,
            RecursionInteractionIndexSpace::PerProof,
            "per-proof recursion interaction bus must include proof_idx"
        );
        let payload = payload.into_iter().collect::<Vec<_>>();
        assert_eq!(
            payload.len(),
            self.schema.payload_arity,
            "recursion interaction payload arity mismatch for {}",
            self.schema.name
        );

        let lift = |value: AB::VarMaybeExt| AB::from_ef(AB::EF::zero()) + value;
        let mut blocks = Vec::with_capacity(self.denominator_value_count());
        blocks.push(lift(AB::VarMaybeExt::from(AB::F::from_canonical_usize(
            self.schema.interaction_idx.0,
        ))));
        blocks.push(lift(proof_idx));
        blocks.extend(payload);
        builder.lookup_denominator_ext_blocks(recursion_interaction_kind::<AB>(), blocks)
    }

    fn lower_values<AB, E, P>(
        &self,
        proof_idx: Option<P>,
        payload: impl IntoIterator<Item = E>,
    ) -> Vec<AB::VarMaybeExt>
    where
        AB: FullAirBuilder,
        E: Into<AB::VarMaybeExt>,
        P: Into<AB::VarMaybeExt>,
    {
        let payload = payload.into_iter().map(Into::into).collect::<Vec<_>>();
        assert_eq!(
            payload.len(),
            self.schema.payload_arity,
            "recursion interaction payload arity mismatch for {}",
            self.schema.name
        );

        let mut values = Vec::with_capacity(self.denominator_value_count());
        values.push(AB::VarMaybeExt::from(AB::F::from_canonical_usize(
            self.schema.interaction_idx.0,
        )));
        match self.schema.index_space {
            RecursionInteractionIndexSpace::Global => {
                debug_assert!(proof_idx.is_none());
            }
            RecursionInteractionIndexSpace::PerProof => {
                values.push(proof_idx.expect("proof_idx required for per-proof bus").into());
            }
        }
        values.extend(payload);
        values
    }
}

fn recursion_interaction_kind<AB: FullAirBuilder>() -> AB::VarMaybeExt {
    AB::VarMaybeExt::from(AB::F::from_canonical_usize(InteractionKind::Recursion as usize))
}
