use dt_stark::air::FullAirBuilder;

use crate::{
    config::POSEIDON2_WIDTH,
    interaction_full_air_dt::RecursionFullAirBus,
    interaction_registry_dt::POSEIDON2_PERMUTE_SCHEMA,
    transcript_dt::poseidon2::{
        columns::{poseidon2_input_from_row, poseidon2_output_from_row},
        NUM_POSEIDON2_PERMUTE_DENOMINATOR_VALUES, NUM_POSEIDON2_PERMUTE_PAYLOAD_VALUES,
    },
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Poseidon2PermuteBus {
    pub bus: RecursionFullAirBus,
}

impl Poseidon2PermuteBus {
    pub const fn new() -> Self {
        Self { bus: RecursionFullAirBus::new(POSEIDON2_PERMUTE_SCHEMA) }
    }

    pub const fn payload_arity(&self) -> usize {
        NUM_POSEIDON2_PERMUTE_PAYLOAD_VALUES
    }

    pub const fn denominator_value_count(&self) -> usize {
        NUM_POSEIDON2_PERMUTE_DENOMINATOR_VALUES
    }

    pub const fn required_max_beta_power_floor(&self) -> usize {
        self.bus.required_max_beta_power_floor()
    }

    pub fn denominator_from_main<AB>(&self, builder: &AB, main: &[AB::VarMaybeExt]) -> AB::VarExt
    where
        AB: FullAirBuilder,
    {
        self.denominator(builder, poseidon2_input_from_row(main), poseidon2_output_from_row(main))
    }

    pub fn denominator<AB>(
        &self,
        builder: &AB,
        input: [AB::VarMaybeExt; POSEIDON2_WIDTH],
        output: [AB::VarMaybeExt; POSEIDON2_WIDTH],
    ) -> AB::VarExt
    where
        AB: FullAirBuilder,
    {
        let mut values = Vec::with_capacity(self.payload_arity());
        values.extend(input);
        values.extend(output);

        self.bus.denominator(builder, values)
    }
}

impl Default for Poseidon2PermuteBus {
    fn default() -> Self {
        Self::new()
    }
}
