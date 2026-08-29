use dt_stark::air::FullAirBuilder;

use crate::{
    config::D_EF,
    interaction_full_air_dt::RecursionFullAirBus,
    interaction_registry_dt::{
        BATCH_OPENING_POINT_SCHEMA, BATCH_SUMCHECK_CLAIM_CHAIN_SCHEMA, SUMCHECK_OUT_SCHEMA,
    },
};

pub const SUMCHECK_OUT_PERM_ALPHA: usize = 0;
pub const SUMCHECK_OUT_PERM_BETA: usize = 1;
pub const SUMCHECK_OUT_ALPHA: usize = 2;
pub const SUMCHECK_OUT_EQ: usize = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchOpeningPointBus {
    bus: RecursionFullAirBus,
}

impl BatchOpeningPointBus {
    pub const fn new() -> Self {
        Self { bus: RecursionFullAirBus::new(BATCH_OPENING_POINT_SCHEMA) }
    }

    pub const fn required_max_beta_power_floor(&self) -> usize {
        self.bus.required_max_beta_power_floor()
    }

    pub fn denominator<AB>(
        &self,
        builder: &AB,
        proof_idx: AB::VarMaybeExt,
        opening_idx: AB::VarMaybeExt,
        value: [AB::VarMaybeExt; D_EF],
    ) -> AB::VarExt
    where
        AB: FullAirBuilder,
    {
        let mut values = Vec::with_capacity(1 + D_EF);
        values.push(opening_idx);
        values.extend(value);
        self.bus.denominator_for_proof(builder, proof_idx, values)
    }
}

impl Default for BatchOpeningPointBus {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchSumcheckClaimChainBus {
    bus: RecursionFullAirBus,
}

impl BatchSumcheckClaimChainBus {
    pub const fn new() -> Self {
        Self { bus: RecursionFullAirBus::new(BATCH_SUMCHECK_CLAIM_CHAIN_SCHEMA) }
    }

    pub const fn required_max_beta_power_floor(&self) -> usize {
        self.bus.required_max_beta_power_floor()
    }

    pub fn denominator<AB>(
        &self,
        builder: &AB,
        proof_idx: AB::VarMaybeExt,
        round_idx: AB::VarMaybeExt,
        r_rounds: AB::VarMaybeExt,
        c_chips: AB::VarMaybeExt,
        claim: [AB::VarMaybeExt; D_EF],
    ) -> AB::VarExt
    where
        AB: FullAirBuilder,
    {
        let mut values = Vec::with_capacity(3 + D_EF);
        values.push(round_idx);
        values.push(r_rounds);
        values.push(c_chips);
        values.extend(claim);
        self.bus.denominator_for_proof(builder, proof_idx, values)
    }
}

impl Default for BatchSumcheckClaimChainBus {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SumcheckOutBus {
    bus: RecursionFullAirBus,
}

impl SumcheckOutBus {
    pub const fn new() -> Self {
        Self { bus: RecursionFullAirBus::new(SUMCHECK_OUT_SCHEMA) }
    }

    pub const fn required_max_beta_power_floor(&self) -> usize {
        self.bus.required_max_beta_power_floor()
    }

    pub fn denominator<AB>(
        &self,
        builder: &AB,
        proof_idx: AB::VarMaybeExt,
        kind: AB::VarMaybeExt,
        idx: AB::VarMaybeExt,
        value: [AB::VarMaybeExt; D_EF],
    ) -> AB::VarExt
    where
        AB: FullAirBuilder,
    {
        let mut values = Vec::with_capacity(2 + D_EF);
        values.push(kind);
        values.push(idx);
        values.extend(value);
        self.bus.denominator_for_proof(builder, proof_idx, values)
    }
}

impl Default for SumcheckOutBus {
    fn default() -> Self {
        Self::new()
    }
}
