use core::{borrow::Borrow, ops::Deref};

use dt_stark::{
    air::{FullAir, FullAirBuilder, MachineAir, PairCol},
    sumcheck::trace::CompressedMatrix,
};
use native_recursion_derive::AlignedBorrow;
use p3_air::BaseAir;
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use crate::{
    config::F,
    primitives_dt::{
        bus::{RangeCheckerBus, RangeCheckerBusMessage},
        range::trace::RangeCheckerTraceGenerator,
    },
    system_dt::{RecursionNativeProgram, RecursionRecord},
};

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct RangeCheckerCols<T, const NUM_BITS: usize> {
    pub value: T,
    pub bits: [T; NUM_BITS],
    pub mult: T,
}

#[derive(Debug, Clone, Copy)]
pub struct RangeCheckerAir<const NUM_BITS: usize> {
    pub bus: RangeCheckerBus,
}

impl<const NUM_BITS: usize> RangeCheckerAir<NUM_BITS> {
    pub const fn new(bus: RangeCheckerBus) -> Self {
        Self { bus }
    }
}

impl<Fld: Field, const NUM_BITS: usize> BaseAir<Fld> for RangeCheckerAir<NUM_BITS> {
    fn width(&self) -> usize {
        RangeCheckerCols::<Fld, NUM_BITS>::width()
    }
}

impl<AB, const NUM_BITS: usize> FullAir<AB> for RangeCheckerAir<NUM_BITS>
where
    AB: FullAirBuilder,
{
    fn width(&self) -> usize {
        RangeCheckerCols::<AB::F, NUM_BITS>::width()
    }

    fn required_max_beta_power(&self) -> usize {
        self.bus.required_max_beta_power_floor()
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        (0..RangeCheckerCols::<AB::F, NUM_BITS>::width()).map(PairCol::Main).collect()
    }

    fn precompute_lc(&self, builder: &mut AB) {
        let value = {
            let main = builder.main();
            let local: &RangeCheckerCols<AB::VarMaybeExt, NUM_BITS> = main.borrow();
            local.value.clone()
        };
        let denominator = self.bus.denominator(
            builder,
            RangeCheckerBusMessage {
                value,
                max_bits: AB::VarMaybeExt::from(AB::F::from_canonical_usize(NUM_BITS)),
            },
        );
        builder.retain_precomputed(denominator);
    }

    fn eval(&self, builder: &mut AB) {
        assert!(NUM_BITS > 0, "range checker needs at least one bit");

        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        let local: &RangeCheckerCols<AB::VarMaybeExt, NUM_BITS> = local_binding.deref().borrow();

        let mut reconstructed = AB::zero_maybe();
        let mut power_of_two = AB::one_maybe();
        for bit in local.bits.iter() {
            builder.assert_zero(bit.clone() * (bit.clone() - AB::one_maybe()));
            reconstructed = reconstructed + bit.clone() * power_of_two.clone();
            power_of_two = power_of_two * AB::VarMaybeExt::from(AB::F::two());
        }
        builder.assert_eq(local.value.clone(), reconstructed);
    }

    fn lookup(&self, builder: &mut AB) {
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        let local: &RangeCheckerCols<AB::VarMaybeExt, NUM_BITS> = local_binding.deref().borrow();

        // Order matches precompute_lc: send RangeCheckerBus(value, max_bits).
        builder.send(local.mult.clone());
    }
}

impl<const NUM_BITS: usize> MachineAir<F> for RangeCheckerAir<NUM_BITS> {
    type Record = RecursionRecord;
    type Program = RecursionNativeProgram<F>;

    fn name(&self) -> String {
        format!("NativeRangeChecker{NUM_BITS}")
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        Some(RangeCheckerTraceGenerator::<NUM_BITS>::trace_height(&input.range))
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        _output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        RangeCheckerTraceGenerator::<NUM_BITS>::generate_trace_compressed_from_pool(&input.range)
    }

    fn generate_dependencies(&self, _input: &Self::Record, _output: &mut Self::Record) {}

    fn included(&self, record: &Self::Record) -> bool {
        record.range.requests_for_bits(NUM_BITS).next().is_some()
    }

    fn local_only(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::{D_EF, F},
        primitives_dt::bus::RangeCheckerBus,
    };

    #[test]
    fn symbolic_analysis() {
        let air = RangeCheckerAir::<7>::new(RangeCheckerBus::new());
        let chip = polyair::Chip::<RangeCheckerAir<7>, F, D_EF>::new(air);
        assert_eq!(chip.num_lookup(), 1);
        assert_eq!(chip.required_max_beta_power(), 14);
        assert_eq!(chip.degree, 3);

        let air = RangeCheckerAir::<8>::new(RangeCheckerBus::new());
        let chip = polyair::Chip::<RangeCheckerAir<8>, F, D_EF>::new(air);
        assert_eq!(chip.num_lookup(), 1);
        assert_eq!(chip.required_max_beta_power(), 14);
        assert_eq!(chip.degree, 3);
    }
}
