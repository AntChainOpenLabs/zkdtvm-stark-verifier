use std::ops::Deref;

use crate::{
    chips::poseidon2_wide_kb::{
        columns::preprocessed::Poseidon2PreprocessedColsWideKb, Poseidon2WideKbChip,
    },
    *,
};
use dt_core_machine::operations::poseidon2_kb::{
    air::eval_poseidon2_full,
    permutation::{Poseidon2Cols, NUM_INTERNAL_ROUNDS_M1, NUM_POSEIDON2_DEGREE3_COLS},
    NUM_EXTERNAL_ROUNDS, WIDTH,
};
use dt_stark::{
    air::{FullAir, FullAirBuilder, MachineAir, PairCol},
    sumcheck::trace::CompressedMatrix,
    InteractionKind,
};
use p3_air::BaseAir;
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

const NUM_COLS: usize = NUM_POSEIDON2_DEGREE3_COLS;
const NUM_PREPROCESSED_COLS: usize = core::mem::size_of::<Poseidon2PreprocessedColsWideKb<u8>>();

// Wide chip layout: external_rounds_state[8][16]=128, internal_rounds_state[16]=16,
//                   internal_rounds_s0[19]=19, output_state[16]=16. Total=179.
const EXT_STATE_OFFSET: usize = 0;
const INT_STATE_OFFSET: usize = 128;
const INT_S0_OFFSET: usize = 144;
const OUTPUT_OFFSET: usize = 163;

/// Adapter that implements `Poseidon2Cols` by referencing a slice of `VarMaybeExt`.
struct Poseidon2ColsView<'a, T: Clone> {
    external_rounds_state: Vec<[T; WIDTH]>,
    internal_rounds_state: [T; WIDTH],
    internal_rounds_s0: [T; NUM_INTERNAL_ROUNDS_M1],
    output_state: [T; WIDTH],
    _phantom: std::marker::PhantomData<&'a ()>,
}

impl<'a, T: Clone> Poseidon2ColsView<'a, T> {
    fn from_slice(s: &'a [T]) -> Self {
        let external_rounds_state: Vec<[T; WIDTH]> = (0..NUM_EXTERNAL_ROUNDS)
            .map(|r| core::array::from_fn(|i| s[EXT_STATE_OFFSET + r * WIDTH + i].clone()))
            .collect();
        let internal_rounds_state = core::array::from_fn(|i| s[INT_STATE_OFFSET + i].clone());
        let internal_rounds_s0 = core::array::from_fn(|i| s[INT_S0_OFFSET + i].clone());
        let output_state = core::array::from_fn(|i| s[OUTPUT_OFFSET + i].clone());
        Self {
            external_rounds_state,
            internal_rounds_state,
            internal_rounds_s0,
            output_state,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<'a, T: Clone> Poseidon2Cols<T> for Poseidon2ColsView<'a, T> {
    fn external_rounds_state(&self) -> &[[T; WIDTH]] {
        &self.external_rounds_state
    }
    fn internal_rounds_state(&self) -> &[T; WIDTH] {
        &self.internal_rounds_state
    }
    fn internal_rounds_s0(&self) -> &[T; NUM_INTERNAL_ROUNDS_M1] {
        &self.internal_rounds_s0
    }
    fn perm_output(&self) -> &[T; WIDTH] {
        &self.output_state
    }
    fn get_cols_mut(
        &mut self,
    ) -> (&mut [[T; WIDTH]], &mut [T; WIDTH], &mut [T; NUM_INTERNAL_ROUNDS_M1], &mut [T; WIDTH])
    {
        (
            &mut self.external_rounds_state,
            &mut self.internal_rounds_state,
            &mut self.internal_rounds_s0,
            &mut self.output_state,
        )
    }
}

#[derive(Default, Clone, Copy)]
pub struct Poseidon2WideKbChipPolyAir;

impl<AB: FullAirBuilder> FullAir<AB> for Poseidon2WideKbChipPolyAir {
    fn width(&self) -> usize {
        NUM_COLS
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        let mut cols = Vec::new();
        // All main trace columns (179 cols for the permutation state)
        for i in 0..NUM_COLS {
            cols.push(PairCol::Main(i));
        }
        // Preprocessed: input[0..16], output[0..16] (addr+mult each), is_real_neg
        // Only output mults and is_real_neg are needed in eval()/lookup()
        // output[i].mult is at offset 16 + 2*i + 1 (addr=even, mult=odd within output section)
        for i in 0..WIDTH {
            cols.push(PairCol::Prep(16 + 2 * i + 1)); // output[i].mult
        }
        cols.push(PairCol::Prep(NUM_PREPROCESSED_COLS - 1)); // is_real_neg
        cols
    }

    fn precompute_lc(&self, builder: &mut AB) {
        let mem_kind =
            AB::VarMaybeExt::from(AB::F::from_canonical_usize(InteractionKind::Memory as usize));

        let main = builder.main();
        let prep = builder.preprocessed();

        let mut interactions = Vec::new();

        // 16 input sends: [input_addr[i], state_in[i]]
        for i in 0..WIDTH {
            let addr = prep[i].clone();
            let val = main[i].clone();
            interactions.push(vec![addr, val]);
        }

        // 16 output sends: [output_addr[i], output[i]]
        for i in 0..WIDTH {
            let addr = prep[16 + 2 * i].clone();
            let val = main[OUTPUT_OFFSET + i].clone();
            interactions.push(vec![addr, val]);
        }

        for vals in interactions {
            builder.retain_precomputed(builder.lookup_denominator(mem_kind.clone(), vals));
        }
    }

    fn eval(&self, builder: &mut AB) {
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        let local: &[AB::VarMaybeExt] = local_binding.deref();

        let view = Poseidon2ColsView::from_slice(&local[..NUM_COLS]);
        eval_poseidon2_full(builder, &view);
    }

    fn lookup(&self, builder: &mut AB) {
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        let local: &[AB::VarMaybeExt] = local_binding.deref();

        let is_real_neg = local[NUM_COLS + WIDTH].clone(); // last reserved col

        // Order matches precompute_lc: 16 input sends, 16 output sends
        for _ in 0..WIDTH {
            builder.send(is_real_neg.clone());
        }
        for i in 0..WIDTH {
            let mult = local[NUM_COLS + i].clone(); // output[i].mult
            builder.send(mult);
        }
    }
}

impl<F: Field> BaseAir<F> for Poseidon2WideKbChipPolyAir {
    fn width(&self) -> usize {
        NUM_COLS
    }
}

impl<F: Field> MachineAir<F> for Poseidon2WideKbChipPolyAir {
    type Record = ExecutionRecord<F>;
    type Program = crate::RecursionProgram<F>;

    fn name(&self) -> String {
        "Poseidon2WideKbDeg3".to_string()
    }

    fn preprocessed_width(&self) -> usize {
        NUM_PREPROCESSED_COLS
    }

    fn preprocessed_num_rows(&self, program: &Self::Program, instrs_len: usize) -> Option<usize> {
        Poseidon2WideKbChip::<3>.preprocessed_num_rows(program, instrs_len)
    }

    fn generate_preprocessed_trace(&self, program: &Self::Program) -> Option<CompressedMatrix<F>> {
        Poseidon2WideKbChip::<3>.generate_preprocessed_trace(program)
    }

    fn generate_dependencies(&self, _: &Self::Record, _: &mut Self::Record) {}

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        Poseidon2WideKbChip::<3>.num_rows(input)
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        Poseidon2WideKbChip::<3>.generate_trace(input, output)
    }

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

    #[test]
    fn symbolic_analysis() {
        use ::polyair::Chip;

        let chip = Chip::<Poseidon2WideKbChipPolyAir, p3_koala_bear::KoalaBear, 5>::new(
            Poseidon2WideKbChipPolyAir,
        );
        assert_eq!(chip.num_lookup(), 32);
        println!(
            "Poseidon2WideKbChipPolyAir: degree={}, num_alpha={}, num_lookup={}",
            chip.degree,
            chip.num_alpha,
            chip.num_lookup()
        );
    }
}
