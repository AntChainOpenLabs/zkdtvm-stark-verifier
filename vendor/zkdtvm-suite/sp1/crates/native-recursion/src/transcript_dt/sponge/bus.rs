use dt_stark::air::FullAirBuilder;

use crate::{
    interaction_full_air_dt::RecursionFullAirBus,
    interaction_registry_dt::{TRANSCRIPT_EVENT_SCHEMA, TRANSCRIPT_SPONGE_CHAIN_SCHEMA},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TranscriptSpongeChainBus {
    bus: RecursionFullAirBus,
}

impl TranscriptSpongeChainBus {
    pub const fn new() -> Self {
        Self { bus: RecursionFullAirBus::new(TRANSCRIPT_SPONGE_CHAIN_SCHEMA) }
    }

    pub const fn required_max_beta_power_floor(&self) -> usize {
        self.bus.required_max_beta_power_floor()
    }

    pub fn denominator<AB>(
        &self,
        builder: &AB,
        proof_idx: AB::VarMaybeExt,
        tidx: AB::VarMaybeExt,
        state: [AB::VarMaybeExt; 16],
        s_count: AB::VarMaybeExt,
    ) -> AB::VarExt
    where
        AB: FullAirBuilder,
    {
        let mut values = Vec::with_capacity(18);
        values.push(tidx);
        values.extend(state);
        values.push(s_count);
        self.bus.denominator_for_proof(builder, proof_idx, values)
    }
}

impl Default for TranscriptSpongeChainBus {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TranscriptEventBus {
    bus: RecursionFullAirBus,
}

impl TranscriptEventBus {
    pub const fn new() -> Self {
        Self { bus: RecursionFullAirBus::new(TRANSCRIPT_EVENT_SCHEMA) }
    }

    pub const fn required_max_beta_power_floor(&self) -> usize {
        self.bus.required_max_beta_power_floor()
    }

    pub fn denominator<AB>(
        &self,
        builder: &AB,
        proof_idx: AB::VarMaybeExt,
        tidx: AB::VarMaybeExt,
        is_sample: AB::VarMaybeExt,
        value: AB::VarMaybeExt,
    ) -> AB::VarExt
    where
        AB: FullAirBuilder,
    {
        self.bus.denominator_for_proof(builder, proof_idx, [tidx, is_sample, value])
    }
}

impl Default for TranscriptEventBus {
    fn default() -> Self {
        Self::new()
    }
}
