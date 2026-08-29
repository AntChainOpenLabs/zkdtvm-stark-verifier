use dt_stark::air::FullAirBuilder;
use native_recursion_derive::AlignedBorrow;

use crate::{
    interaction_full_air_dt::RecursionFullAirBus, interaction_registry_dt::RANGE_CHECKER_SCHEMA,
};

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct RangeCheckerBusMessage<T> {
    pub value: T,
    pub max_bits: T,
}

impl<T> RangeCheckerBusMessage<T> {
    pub fn into_payload(self) -> [T; 2] {
        [self.value, self.max_bits]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RangeCheckerBus {
    bus: RecursionFullAirBus,
}

impl RangeCheckerBus {
    pub const fn new() -> Self {
        Self { bus: RecursionFullAirBus::new(RANGE_CHECKER_SCHEMA) }
    }

    pub const fn required_max_beta_power_floor(&self) -> usize {
        self.bus.required_max_beta_power_floor()
    }

    pub fn denominator<AB>(
        &self,
        builder: &AB,
        message: RangeCheckerBusMessage<AB::VarMaybeExt>,
    ) -> AB::VarExt
    where
        AB: FullAirBuilder,
    {
        self.bus.denominator(builder, message.into_payload())
    }
}

impl Default for RangeCheckerBus {
    fn default() -> Self {
        Self::new()
    }
}
