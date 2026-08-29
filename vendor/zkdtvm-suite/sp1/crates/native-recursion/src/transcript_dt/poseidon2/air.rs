use core::ops::Deref;

use dt_core_machine::operations::poseidon2_kb::air::eval_poseidon2_full;
use dt_stark::{
    air::{FullAir, FullAirBuilder, MachineAir, PairCol},
    sumcheck::trace::CompressedMatrix,
};
use p3_air::BaseAir;
use p3_field::Field;
use p3_matrix::Matrix;

use crate::{
    config::F,
    system_dt::{RecursionNativeProgram, RecursionRecord},
    transcript_dt::{
        bus::Poseidon2PermuteBus,
        poseidon2::{
            columns::{Poseidon2ColsView, NUM_POSEIDON2_PERMUTE_COLS, POSEIDON2_MULT_COL},
            trace::Poseidon2PermuteTraceGenerator,
        },
    },
};

#[derive(Debug, Clone, Copy)]
pub struct Poseidon2PermuteAir {
    pub bus: Poseidon2PermuteBus,
}

impl Poseidon2PermuteAir {
    pub const fn new(bus: Poseidon2PermuteBus) -> Self {
        Self { bus }
    }
}

impl<AB: FullAirBuilder> FullAir<AB> for Poseidon2PermuteAir {
    fn width(&self) -> usize {
        NUM_POSEIDON2_PERMUTE_COLS
    }

    fn required_max_beta_power(&self) -> usize {
        self.bus.required_max_beta_power_floor()
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        (0..NUM_POSEIDON2_PERMUTE_COLS).map(PairCol::Main).collect()
    }

    fn precompute_lc(&self, builder: &mut AB) {
        let denominator = {
            let main = builder.main();
            self.bus.denominator_from_main(builder, main)
        };
        builder.retain_precomputed(denominator);
    }

    fn eval(&self, builder: &mut AB) {
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        let local: &[AB::VarMaybeExt] = local_binding.deref();
        let view = Poseidon2ColsView::from_slice(local);
        eval_poseidon2_full(builder, &view);
    }

    fn lookup(&self, builder: &mut AB) {
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        let local: &[AB::VarMaybeExt] = local_binding.deref();

        // Order matches precompute_lc: send Poseidon2PermuteBus(input16, output16).
        builder.send(local[POSEIDON2_MULT_COL].clone());
    }
}

impl<Fld: Field> BaseAir<Fld> for Poseidon2PermuteAir {
    fn width(&self) -> usize {
        NUM_POSEIDON2_PERMUTE_COLS
    }
}

impl MachineAir<F> for Poseidon2PermuteAir {
    type Record = RecursionRecord;
    type Program = RecursionNativeProgram<F>;

    fn name(&self) -> String {
        "NativePoseidon2Permute".to_string()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        Some(Poseidon2PermuteTraceGenerator::trace_height(&input.poseidon2))
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        _output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        // The complete 179-column witness is generated only at this tracegen boundary.
        let start = crate::Instant::now();
        let witness_nanos_before = input.poseidon2_tracegen.generation_nanos();
        let trace = Poseidon2PermuteTraceGenerator::generate_trace_compressed(
            &input.poseidon2,
            &input.poseidon2_tracegen,
        );
        let witness_ms = u128::from(
            input.poseidon2_tracegen.generation_nanos().saturating_sub(witness_nanos_before),
        ) / 1_000_000;
        input.profile.add_record_split("tracegen.poseidon2_witness_generation", witness_ms);
        input.profile.add_record_split(
            "tracegen.poseidon2_matrix_population",
            start.elapsed().as_millis().saturating_sub(witness_ms),
        );
        input.profile.set_structural_counter(
            "full_poseidon2_witness_rows_during_tracegen",
            u64::try_from(input.poseidon2.unique_count()).expect("Poseidon2 rows exceed u64"),
        );
        trace
    }

    fn generate_dependencies(&self, _input: &Self::Record, _output: &mut Self::Record) {}

    fn included(&self, _record: &Self::Record) -> bool {
        true
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
        transcript_dt::poseidon2::NUM_POSEIDON2_PERMUTE_DENOMINATOR_VALUES,
    };

    #[test]
    fn symbolic_analysis() {
        let air = Poseidon2PermuteAir::new(Poseidon2PermuteBus::new());
        let chip = polyair::Chip::<Poseidon2PermuteAir, F, D_EF>::new(air);
        assert_eq!(chip.num_lookup(), 1);
        assert_eq!(chip.required_max_beta_power(), NUM_POSEIDON2_PERMUTE_DENOMINATOR_VALUES + 1);
        assert_eq!(chip.degree, 3);
    }
}
