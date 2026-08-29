use crate::{
    chips::select::{SelectChip, SELECT_COLS, SELECT_PREPROCESSED_COLS},
    *,
};
use dt_stark::{
    air::{FullAir, FullAirBuilder, MachineAir, PairCol},
    sumcheck::trace::CompressedMatrix,
    InteractionKind,
};
use p3_air::BaseAir;
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;
use std::ops::Deref;

#[derive(Default, Clone, Copy)]
pub struct SelectChipPolyAir;

// Main trace column indices (SelectCols layout: SelectIo { bit, out1, out2, in1, in2 })
const MAIN_BIT: usize = 0;
const MAIN_OUT1: usize = 1;
const MAIN_OUT2: usize = 2;
const MAIN_IN1: usize = 3;
const MAIN_IN2: usize = 4;

// Preprocessed column indices (SelectPreprocessedCols layout:
//   is_real, addrs: SelectIo<Address> { bit, out1, out2, in1, in2 }, mult1, mult2)
const PREP_IS_REAL: usize = 0;
const PREP_ADDR_BIT: usize = 1;
const PREP_ADDR_OUT1: usize = 2;
const PREP_ADDR_OUT2: usize = 3;
const PREP_ADDR_IN1: usize = 4;
const PREP_ADDR_IN2: usize = 5;
const PREP_MULT1: usize = 6;
const PREP_MULT2: usize = 7;

// Reserved polynomial indices (output of reserved_poly(), used in eval/lookup)
const RESERVED_BIT: usize = 0;
const RESERVED_OUT1: usize = 1;
const RESERVED_OUT2: usize = 2;
const RESERVED_IN1: usize = 3;
const RESERVED_IN2: usize = 4;
const RESERVED_IS_REAL: usize = 5;
const RESERVED_MULT1: usize = 6;
const RESERVED_MULT2: usize = 7;

impl<AB: FullAirBuilder> FullAir<AB> for SelectChipPolyAir {
    fn width(&self) -> usize {
        SELECT_COLS
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        vec![
            PairCol::Main(MAIN_BIT),     // [0]
            PairCol::Main(MAIN_OUT1),    // [1]
            PairCol::Main(MAIN_OUT2),    // [2]
            PairCol::Main(MAIN_IN1),     // [3]
            PairCol::Main(MAIN_IN2),     // [4]
            PairCol::Prep(PREP_IS_REAL), // [5]
            PairCol::Prep(PREP_MULT1),   // [6]
            PairCol::Prep(PREP_MULT2),   // [7]
        ]
    }

    fn precompute_lc(&self, builder: &mut AB) {
        let main = builder.main();
        let bit = main[MAIN_BIT].clone();
        let out1 = main[MAIN_OUT1].clone();
        let out2 = main[MAIN_OUT2].clone();
        let in1 = main[MAIN_IN1].clone();
        let in2 = main[MAIN_IN2].clone();

        let prep = builder.preprocessed();
        let addr_bit = prep[PREP_ADDR_BIT].clone();
        let addr_out1 = prep[PREP_ADDR_OUT1].clone();
        let addr_out2 = prep[PREP_ADDR_OUT2].clone();
        let addr_in1 = prep[PREP_ADDR_IN1].clone();
        let addr_in2 = prep[PREP_ADDR_IN2].clone();

        let mem_kind =
            AB::VarMaybeExt::from(AB::F::from_canonical_usize(InteractionKind::Memory as usize));

        // recv bit
        builder
            .retain_precomputed(builder.lookup_denominator(mem_kind.clone(), vec![addr_bit, bit]));

        // recv in1
        builder
            .retain_precomputed(builder.lookup_denominator(mem_kind.clone(), vec![addr_in1, in1]));

        // recv in2
        builder
            .retain_precomputed(builder.lookup_denominator(mem_kind.clone(), vec![addr_in2, in2]));

        // send out1
        builder.retain_precomputed(
            builder.lookup_denominator(mem_kind.clone(), vec![addr_out1, out1]),
        );

        // send out2
        builder.retain_precomputed(builder.lookup_denominator(mem_kind, vec![addr_out2, out2]));
    }

    fn eval(&self, builder: &mut AB) {
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        let local: &[AB::VarMaybeExt] = local_binding.deref();

        let bit = local[RESERVED_BIT].clone();
        let out1 = local[RESERVED_OUT1].clone();
        let out2 = local[RESERVED_OUT2].clone();
        let in1 = local[RESERVED_IN1].clone();
        let in2 = local[RESERVED_IN2].clone();
        let is_real = local[RESERVED_IS_REAL].clone();

        // out1 = in1 + bit * (in2 - in1)
        builder.assert_zero(out1.clone() - in1.clone() - bit * (in2.clone() - in1.clone()));

        // when(is_real): out1 + out2 = in1 + in2
        builder.when(is_real).assert_zero(out1 + out2 - in1 - in2);
    }

    fn lookup(&self, builder: &mut AB) {
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        let local: &[AB::VarMaybeExt] = local_binding.deref();

        let is_real = local[RESERVED_IS_REAL].clone();
        let mult1 = local[RESERVED_MULT1].clone();
        let mult2 = local[RESERVED_MULT2].clone();

        // Order matches precompute_lc: recv(bit), recv(in1), recv(in2), send(out1), send(out2)
        builder.recv(is_real.clone());
        builder.recv(is_real.clone());
        builder.recv(is_real);
        builder.send(mult1);
        builder.send(mult2);
    }
}

impl<F: Field> BaseAir<F> for SelectChipPolyAir {
    fn width(&self) -> usize {
        SELECT_COLS
    }
}

impl<F: Field> MachineAir<F> for SelectChipPolyAir {
    type Record = ExecutionRecord<F>;
    type Program = crate::RecursionProgram<F>;

    fn name(&self) -> String {
        "Select".to_string()
    }

    fn preprocessed_width(&self) -> usize {
        SELECT_PREPROCESSED_COLS
    }

    fn preprocessed_num_rows(&self, program: &Self::Program, instrs_len: usize) -> Option<usize> {
        SelectChip.preprocessed_num_rows(program, instrs_len)
    }

    fn generate_preprocessed_trace(&self, program: &Self::Program) -> Option<CompressedMatrix<F>> {
        SelectChip.generate_preprocessed_trace(program)
    }

    fn generate_dependencies(&self, _: &Self::Record, _: &mut Self::Record) {}

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        SelectChip.num_rows(input)
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        SelectChip.generate_trace(input, output)
    }

    fn included(&self, record: &Self::Record) -> bool {
        SelectChip.included(record)
    }

    fn local_only(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symbolic_analysis() {
        use ::polyair::Chip;

        #[cfg(feature = "koalabear")]
        {
            let chip =
                Chip::<SelectChipPolyAir, p3_koala_bear::KoalaBear, 5>::new(SelectChipPolyAir);
            assert!(chip.degree >= 2);
        }
        #[cfg(feature = "babybear")]
        {
            let chip = Chip::<SelectChipPolyAir, p3_baby_bear::BabyBear, 4>::new(SelectChipPolyAir);
            assert!(chip.degree >= 2);
        }
    }
}
