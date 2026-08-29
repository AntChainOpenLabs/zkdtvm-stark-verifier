use dt_stark::{
    air::{AirInteraction, InteractionScope, MessageBuilder},
    InteractionKind,
};
use p3_air::AirBuilder;
use p3_field::{AbstractField, PrimeField64};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RecursionInteractionIdx(pub usize);

impl From<usize> for RecursionInteractionIdx {
    fn from(value: usize) -> Self {
        Self(value)
    }
}

impl From<u64> for RecursionInteractionIdx {
    fn from(value: u64) -> Self {
        Self(value as usize)
    }
}

impl From<u32> for RecursionInteractionIdx {
    fn from(value: u32) -> Self {
        Self(value as usize)
    }
}

impl From<u16> for RecursionInteractionIdx {
    fn from(value: u16) -> Self {
        Self(value as usize)
    }
}

impl From<u8> for RecursionInteractionIdx {
    fn from(value: u8) -> Self {
        Self(value as usize)
    }
}

impl From<i32> for RecursionInteractionIdx {
    fn from(value: i32) -> Self {
        assert!(value >= 0, "recursion interaction index cannot be negative");
        Self(value as usize)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecursionInteractionKind {
    Lookup,
    Permutation,
}

/// Whether a lowered recursion interaction is keyed by proof index.
///
/// Global/provider interactions omit proof index and per-proof interactions
/// include it as the second field. Their `RecursionInteractionIdx` allocation
/// domains must stay disjoint; otherwise a global/provider interaction can
/// collide with proof-indexed semantics under the same logical bus id.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecursionInteractionIndexSpace {
    Global,
    PerProof,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecursionInteractionLoweringSpec {
    pub interaction_idx: RecursionInteractionIdx,
    pub recursion_kind: RecursionInteractionKind,
    pub index_space: RecursionInteractionIndexSpace,
    pub stark_kind: InteractionKind,
    pub scope: InteractionScope,
    pub payload_arity: Option<usize>,
}

impl RecursionInteractionLoweringSpec {
    pub const fn new(
        interaction_idx: RecursionInteractionIdx,
        recursion_kind: RecursionInteractionKind,
    ) -> Self {
        Self::new_global(interaction_idx, recursion_kind)
    }

    pub const fn new_global(
        interaction_idx: RecursionInteractionIdx,
        recursion_kind: RecursionInteractionKind,
    ) -> Self {
        Self {
            interaction_idx,
            recursion_kind,
            index_space: RecursionInteractionIndexSpace::Global,
            stark_kind: InteractionKind::Recursion,
            scope: InteractionScope::Local,
            payload_arity: None,
        }
    }

    pub const fn new_per_proof(
        interaction_idx: RecursionInteractionIdx,
        recursion_kind: RecursionInteractionKind,
    ) -> Self {
        Self {
            interaction_idx,
            recursion_kind,
            index_space: RecursionInteractionIndexSpace::PerProof,
            stark_kind: InteractionKind::Recursion,
            scope: InteractionScope::Local,
            payload_arity: None,
        }
    }

    pub const fn with_payload_arity(mut self, payload_arity: usize) -> Self {
        self.payload_arity = Some(payload_arity);
        self
    }

    pub fn active_interaction_kinds() -> &'static [InteractionKind] {
        InteractionKind::recursion_kinds()
    }

    #[inline]
    fn lower_values<AB>(
        &self,
        proof_idx: Option<AB::Expr>,
        payload: impl IntoIterator<Item = AB::Expr>,
    ) -> Vec<AB::Expr>
    where
        AB: RecursionInteractionBuilder,
    {
        let payload = payload.into_iter().collect::<Vec<_>>();
        if let Some(payload_arity) = self.payload_arity {
            assert_eq!(
                payload.len(),
                payload_arity,
                "recursion interaction payload arity mismatch"
            );
        }
        let proof_idx_len =
            usize::from(self.index_space == RecursionInteractionIndexSpace::PerProof);
        let mut values = Vec::with_capacity(payload.len() + 1 + proof_idx_len);
        values.push(AB::Expr::from_canonical_usize(self.interaction_idx.0));
        if self.index_space == RecursionInteractionIndexSpace::PerProof {
            values.push(proof_idx.unwrap_or_else(AB::Expr::zero));
        } else {
            debug_assert!(proof_idx.is_none());
        }
        values.extend(payload);
        values
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecursionInteractionBudget {
    pub num_sends: usize,
    pub num_receives: usize,
    pub log_height: usize,
}

impl RecursionInteractionBudget {
    pub const fn new(num_sends: usize, num_receives: usize, log_height: usize) -> Self {
        Self { num_sends, num_receives, log_height }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecursionInteractionBudgetError {
    CountOverflow,
    LogHeightTooLarge,
    MultiplicityOverflow { max_lookup_mult: u64, field_order: u64 },
}

pub fn validate_recursion_interaction_budget<F>(
    chips: impl IntoIterator<Item = RecursionInteractionBudget>,
) -> Result<u64, RecursionInteractionBudgetError>
where
    F: PrimeField64,
{
    let mut max_lookup_mult = 0u64;
    for chip in chips {
        let num_sends = u64::try_from(chip.num_sends)
            .map_err(|_| RecursionInteractionBudgetError::CountOverflow)?;
        let num_receives = u64::try_from(chip.num_receives)
            .map_err(|_| RecursionInteractionBudgetError::CountOverflow)?;
        let interaction_count = num_sends
            .checked_add(num_receives)
            .ok_or(RecursionInteractionBudgetError::CountOverflow)?;
        let shift = u32::try_from(chip.log_height)
            .map_err(|_| RecursionInteractionBudgetError::LogHeightTooLarge)?;
        let rows =
            1u64.checked_shl(shift).ok_or(RecursionInteractionBudgetError::LogHeightTooLarge)?;
        let chip_mult = interaction_count
            .checked_mul(rows)
            .ok_or(RecursionInteractionBudgetError::CountOverflow)?;
        max_lookup_mult = max_lookup_mult
            .checked_add(chip_mult)
            .ok_or(RecursionInteractionBudgetError::CountOverflow)?;
    }
    if max_lookup_mult >= F::ORDER_U64 {
        return Err(RecursionInteractionBudgetError::MultiplicityOverflow {
            max_lookup_mult,
            field_order: F::ORDER_U64,
        });
    }
    Ok(max_lookup_mult)
}

pub trait RecursionInteractionBuilder:
    AirBuilder + MessageBuilder<AirInteraction<<Self as AirBuilder>::Expr>>
{
}

impl<AB> RecursionInteractionBuilder for AB where
    AB: AirBuilder + MessageBuilder<AirInteraction<<AB as AirBuilder>::Expr>>
{
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecursionLookupInteraction {
    spec: RecursionInteractionLoweringSpec,
}

impl RecursionLookupInteraction {
    #[inline]
    pub fn new(interaction_idx: impl Into<RecursionInteractionIdx>) -> Self {
        Self::new_global(interaction_idx)
    }

    #[inline]
    pub fn new_global(interaction_idx: impl Into<RecursionInteractionIdx>) -> Self {
        Self {
            spec: RecursionInteractionLoweringSpec::new_global(
                interaction_idx.into(),
                RecursionInteractionKind::Lookup,
            ),
        }
    }

    #[inline]
    pub fn new_global_with_payload_arity(
        interaction_idx: impl Into<RecursionInteractionIdx>,
        payload_arity: usize,
    ) -> Self {
        Self {
            spec: RecursionInteractionLoweringSpec::new_global(
                interaction_idx.into(),
                RecursionInteractionKind::Lookup,
            )
            .with_payload_arity(payload_arity),
        }
    }

    #[inline]
    pub fn new_per_proof(interaction_idx: impl Into<RecursionInteractionIdx>) -> Self {
        Self {
            spec: RecursionInteractionLoweringSpec::new_per_proof(
                interaction_idx.into(),
                RecursionInteractionKind::Lookup,
            ),
        }
    }

    #[inline]
    pub fn new_per_proof_with_payload_arity(
        interaction_idx: impl Into<RecursionInteractionIdx>,
        payload_arity: usize,
    ) -> Self {
        Self {
            spec: RecursionInteractionLoweringSpec::new_per_proof(
                interaction_idx.into(),
                RecursionInteractionKind::Lookup,
            )
            .with_payload_arity(payload_arity),
        }
    }

    #[inline]
    pub fn lookup_key<AB, E>(
        &self,
        builder: &mut AB,
        payload: impl IntoIterator<Item = E>,
        enabled: impl Into<AB::Expr>,
    ) where
        AB: RecursionInteractionBuilder,
        E: Into<AB::Expr>,
    {
        debug_assert_eq!(self.spec.index_space, RecursionInteractionIndexSpace::Global);
        let values = self.spec.lower_values::<AB>(None, payload.into_iter().map(Into::into));
        builder.receive(
            AirInteraction::new(values, enabled.into(), self.spec.stark_kind),
            self.spec.scope,
        );
    }

    #[inline]
    pub fn lookup_key_for_proof<AB, E>(
        &self,
        builder: &mut AB,
        proof_idx: Option<impl Into<AB::Expr>>,
        payload: impl IntoIterator<Item = E>,
        enabled: impl Into<AB::Expr>,
    ) where
        AB: RecursionInteractionBuilder,
        E: Into<AB::Expr>,
    {
        debug_assert_eq!(self.spec.index_space, RecursionInteractionIndexSpace::PerProof);
        let values = self
            .spec
            .lower_values::<AB>(proof_idx.map(Into::into), payload.into_iter().map(Into::into));
        builder.receive(
            AirInteraction::new(values, enabled.into(), self.spec.stark_kind),
            self.spec.scope,
        );
    }

    #[inline]
    pub fn add_key_with_lookups<AB, E>(
        &self,
        builder: &mut AB,
        payload: impl IntoIterator<Item = E>,
        num_lookups: impl Into<AB::Expr>,
    ) where
        AB: RecursionInteractionBuilder,
        E: Into<AB::Expr>,
    {
        debug_assert_eq!(self.spec.index_space, RecursionInteractionIndexSpace::Global);
        let values = self.spec.lower_values::<AB>(None, payload.into_iter().map(Into::into));
        builder.send(
            AirInteraction::new(values, num_lookups.into(), self.spec.stark_kind),
            self.spec.scope,
        );
    }

    #[inline]
    pub fn add_key_with_lookups_for_proof<AB, E>(
        &self,
        builder: &mut AB,
        proof_idx: Option<impl Into<AB::Expr>>,
        payload: impl IntoIterator<Item = E>,
        num_lookups: impl Into<AB::Expr>,
    ) where
        AB: RecursionInteractionBuilder,
        E: Into<AB::Expr>,
    {
        debug_assert_eq!(self.spec.index_space, RecursionInteractionIndexSpace::PerProof);
        let values = self
            .spec
            .lower_values::<AB>(proof_idx.map(Into::into), payload.into_iter().map(Into::into));
        builder.send(
            AirInteraction::new(values, num_lookups.into(), self.spec.stark_kind),
            self.spec.scope,
        );
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecursionPermutationInteraction {
    spec: RecursionInteractionLoweringSpec,
}

impl RecursionPermutationInteraction {
    #[inline]
    pub fn new(interaction_idx: impl Into<RecursionInteractionIdx>) -> Self {
        Self::new_global(interaction_idx)
    }

    #[inline]
    pub fn new_global(interaction_idx: impl Into<RecursionInteractionIdx>) -> Self {
        Self {
            spec: RecursionInteractionLoweringSpec::new_global(
                interaction_idx.into(),
                RecursionInteractionKind::Permutation,
            ),
        }
    }

    #[inline]
    pub fn new_global_with_payload_arity(
        interaction_idx: impl Into<RecursionInteractionIdx>,
        payload_arity: usize,
    ) -> Self {
        Self {
            spec: RecursionInteractionLoweringSpec::new_global(
                interaction_idx.into(),
                RecursionInteractionKind::Permutation,
            )
            .with_payload_arity(payload_arity),
        }
    }

    #[inline]
    pub fn new_per_proof(interaction_idx: impl Into<RecursionInteractionIdx>) -> Self {
        Self {
            spec: RecursionInteractionLoweringSpec::new_per_proof(
                interaction_idx.into(),
                RecursionInteractionKind::Permutation,
            ),
        }
    }

    #[inline]
    pub fn new_per_proof_with_payload_arity(
        interaction_idx: impl Into<RecursionInteractionIdx>,
        payload_arity: usize,
    ) -> Self {
        Self {
            spec: RecursionInteractionLoweringSpec::new_per_proof(
                interaction_idx.into(),
                RecursionInteractionKind::Permutation,
            )
            .with_payload_arity(payload_arity),
        }
    }

    #[inline]
    pub fn send<AB, E>(
        &self,
        builder: &mut AB,
        payload: impl IntoIterator<Item = E>,
        enabled: impl Into<AB::Expr>,
    ) where
        AB: RecursionInteractionBuilder,
        E: Into<AB::Expr>,
    {
        debug_assert_eq!(self.spec.index_space, RecursionInteractionIndexSpace::Global);
        let values = self.spec.lower_values::<AB>(None, payload.into_iter().map(Into::into));
        builder.send(
            AirInteraction::new(values, enabled.into(), self.spec.stark_kind),
            self.spec.scope,
        );
    }

    #[inline]
    pub fn send_for_proof<AB, E>(
        &self,
        builder: &mut AB,
        proof_idx: Option<impl Into<AB::Expr>>,
        payload: impl IntoIterator<Item = E>,
        enabled: impl Into<AB::Expr>,
    ) where
        AB: RecursionInteractionBuilder,
        E: Into<AB::Expr>,
    {
        debug_assert_eq!(self.spec.index_space, RecursionInteractionIndexSpace::PerProof);
        let values = self
            .spec
            .lower_values::<AB>(proof_idx.map(Into::into), payload.into_iter().map(Into::into));
        builder.send(
            AirInteraction::new(values, enabled.into(), self.spec.stark_kind),
            self.spec.scope,
        );
    }

    #[inline]
    pub fn receive<AB, E>(
        &self,
        builder: &mut AB,
        payload: impl IntoIterator<Item = E>,
        enabled: impl Into<AB::Expr>,
    ) where
        AB: RecursionInteractionBuilder,
        E: Into<AB::Expr>,
    {
        debug_assert_eq!(self.spec.index_space, RecursionInteractionIndexSpace::Global);
        let values = self.spec.lower_values::<AB>(None, payload.into_iter().map(Into::into));
        builder.receive(
            AirInteraction::new(values, enabled.into(), self.spec.stark_kind),
            self.spec.scope,
        );
    }

    #[inline]
    pub fn receive_for_proof<AB, E>(
        &self,
        builder: &mut AB,
        proof_idx: Option<impl Into<AB::Expr>>,
        payload: impl IntoIterator<Item = E>,
        enabled: impl Into<AB::Expr>,
    ) where
        AB: RecursionInteractionBuilder,
        E: Into<AB::Expr>,
    {
        debug_assert_eq!(self.spec.index_space, RecursionInteractionIndexSpace::PerProof);
        let values = self
            .spec
            .lower_values::<AB>(proof_idx.map(Into::into), payload.into_iter().map(Into::into));
        builder.receive(
            AirInteraction::new(values, enabled.into(), self.spec.stark_kind),
            self.spec.scope,
        );
    }
}
