use crate::{
    chips::ext_exp_reverse_bits::{
        ExtExpReverseBitsChip, NUM_EXT_EXP_REVERSE_BITS_COLS,
        NUM_EXT_EXP_REVERSE_BITS_PREPROCESSED_COLS,
    },
    *,
};
use dt_stark::{
    air::{ChallengeExtension, FullAir, FullAirBuilder, MachineAir, PairCol},
    sumcheck::trace::CompressedMatrix,
    InteractionKind,
};
use p3_air::BaseAir;
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;
use std::ops::Deref;

pub struct ExtExpReverseBitsChipPolyAir;

impl Default for ExtExpReverseBitsChipPolyAir {
    fn default() -> Self {
        Self
    }
}

impl Clone for ExtExpReverseBitsChipPolyAir {
    fn clone(&self) -> Self {
        *self
    }
}

impl Copy for ExtExpReverseBitsChipPolyAir {}

impl<AB: FullAirBuilder> FullAir<AB> for ExtExpReverseBitsChipPolyAir {
    fn width(&self) -> usize {
        NUM_EXT_EXP_REVERSE_BITS_COLS
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        let mut cols = Vec::new();

        // Main: ExtExpReverseBitsCols { x[0..D], current_bit, prev_acc[0..D], acc[0..D],
        // multiplier[0..D] } = 4*D + 1 columns
        for i in 0..NUM_EXT_EXP_REVERSE_BITS_COLS {
            cols.push(PairCol::Main(i));
        }

        // Preprocessed: { x_addr[0], exponent_addr[1], prev_acc_addr[2], acc_mem.addr[3],
        // acc_mem.mult[4], is_real[5] } Only acc_mem.mult and is_real are needed in
        // eval()/lookup(); addresses are only in precompute_lc(). But we also need is_real
        // in eval() for the multiplier constraint guard and degree normalization.
        cols.push(PairCol::Prep(4)); // acc_mem.mult
        cols.push(PairCol::Prep(5)); // is_real

        cols
    }

    fn precompute_lc(&self, builder: &mut AB) {
        let d = runtime::D;
        let mem_kind =
            AB::VarMaybeExt::from(AB::F::from_canonical_usize(InteractionKind::Memory as usize));

        let main = builder.main();
        let prep = builder.preprocessed();

        // Preprocessed layout: x_addr[0], exponent_addr[1], prev_acc_addr[2], acc_mem.addr[3],
        // acc_mem.mult[4], is_real[5]
        let x_addr = prep[0].clone();
        let exponent_addr = prep[1].clone();
        let prev_acc_addr = prep[2].clone();
        let acc_addr = prep[3].clone();

        // Main layout: x[0..D], current_bit[D], prev_acc[D+1..2D+1], acc[2D+1..3D+1],
        // multiplier[3D+1..4D+1]
        let x_vals: Vec<_> = (0..d).map(|i| main[i].clone()).collect();
        let current_bit = main[d].clone();
        let prev_acc_vals: Vec<_> = (0..d).map(|i| main[d + 1 + i].clone()).collect();
        let acc_vals: Vec<_> = (0..d).map(|i| main[2 * d + 1 + i].clone()).collect();

        // 1. recv x: [x_addr, x[0], ..., x[D-1]]
        let mut vals_x = vec![x_addr];
        vals_x.extend(x_vals);

        // 2. recv current_bit (single value, no padding needed — zero terms don't affect the
        //    denominator)
        let vals_bit = vec![exponent_addr, current_bit];

        // 3. recv prev_acc: [prev_acc_addr, prev_acc[0], ..., prev_acc[D-1]]
        let mut vals_prev_acc = vec![prev_acc_addr];
        vals_prev_acc.extend(prev_acc_vals);

        // 4. send acc: [acc_addr, acc[0], ..., acc[D-1]]
        let mut vals_acc = vec![acc_addr];
        vals_acc.extend(acc_vals);

        for vals in [vals_x, vals_bit, vals_prev_acc, vals_acc] {
            builder.retain_precomputed(builder.lookup_denominator(mem_kind.clone(), vals));
        }
    }

    fn eval(&self, builder: &mut AB) {
        let d = runtime::D;
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        let local: &[AB::VarMaybeExt] = local_binding.deref();

        let x: Vec<_> = (0..d).map(|i| local[i].clone()).collect();
        let bit = local[d].clone();
        let prev_acc: Vec<_> = (0..d).map(|i| local[d + 1 + i].clone()).collect();
        let acc: Vec<_> = (0..d).map(|i| local[2 * d + 1 + i].clone()).collect();
        let multiplier: Vec<_> = (0..d).map(|i| local[3 * d + 1 + i].clone()).collect();
        let is_real = local[4 * d + 1 + 1].clone();

        // multiplier = 1 + bit*(x - 1): equals x when bit=1, equals 1 when bit=0.
        let x_ext = ChallengeExtension(core::array::from_fn(|i| x[i].clone()));
        let mul_ext = ChallengeExtension(core::array::from_fn(|i| multiplier[i].clone()));
        let one_ext = ChallengeExtension::<AB::VarMaybeExt>::from_base(AB::VarMaybeExt::one());
        let bit_ext = ChallengeExtension::from_base(bit);
        let expected_mul = one_ext.clone() + bit_ext * (x_ext - one_ext);
        for k in 0..d {
            builder
                .when(is_real.clone())
                .assert_zero(mul_ext.0[k].clone() - expected_mul.0[k].clone());
        }

        // acc = prev_acc^2 * multiplier
        let prev_acc_ext = ChallengeExtension(core::array::from_fn(|i| prev_acc[i].clone()));
        let mul_ext = ChallengeExtension(core::array::from_fn(|i| multiplier[i].clone()));
        let expected = prev_acc_ext.clone() * prev_acc_ext * mul_ext;
        for i in 0..d {
            builder.assert_zero(acc[i].clone() - expected.0[i].clone());
        }
    }

    fn lookup(&self, builder: &mut AB) {
        let d = runtime::D;
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        let local: &[AB::VarMaybeExt] = local_binding.deref();

        let acc_mult = local[4 * d + 1].clone();
        let is_real = local[4 * d + 1 + 1].clone();

        // Order matches precompute_lc: recv(x), recv(bit), recv(prev_acc), send(acc)
        builder.recv(is_real.clone());
        builder.recv(is_real.clone());
        builder.recv(is_real);
        builder.send(acc_mult);
    }
}

impl<F: Field> BaseAir<F> for ExtExpReverseBitsChipPolyAir {
    fn width(&self) -> usize {
        NUM_EXT_EXP_REVERSE_BITS_COLS
    }
}

impl<F: Field> MachineAir<F> for ExtExpReverseBitsChipPolyAir {
    type Record = ExecutionRecord<F>;
    type Program = crate::RecursionProgram<F>;

    fn name(&self) -> String {
        #[cfg(feature = "babybear")]
        {
            "ExtExpReverseBitsDeg9".to_string()
        }
        #[cfg(feature = "koalabear")]
        {
            "ExtExpReverseBitsDeg3".to_string()
        }
    }

    fn preprocessed_width(&self) -> usize {
        NUM_EXT_EXP_REVERSE_BITS_PREPROCESSED_COLS
    }

    fn preprocessed_num_rows(&self, program: &Self::Program, instrs_len: usize) -> Option<usize> {
        #[cfg(feature = "babybear")]
        {
            ExtExpReverseBitsChip::<9>.preprocessed_num_rows(program, instrs_len)
        }
        #[cfg(feature = "koalabear")]
        {
            ExtExpReverseBitsChip::<3>.preprocessed_num_rows(program, instrs_len)
        }
    }

    fn generate_preprocessed_trace(&self, program: &Self::Program) -> Option<CompressedMatrix<F>> {
        #[cfg(feature = "babybear")]
        {
            ExtExpReverseBitsChip::<9>.generate_preprocessed_trace(program)
        }
        #[cfg(feature = "koalabear")]
        {
            ExtExpReverseBitsChip::<3>.generate_preprocessed_trace(program)
        }
    }

    fn generate_dependencies(&self, _: &Self::Record, _: &mut Self::Record) {}

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        #[cfg(feature = "babybear")]
        {
            ExtExpReverseBitsChip::<9>.num_rows(input)
        }
        #[cfg(feature = "koalabear")]
        {
            ExtExpReverseBitsChip::<3>.num_rows(input)
        }
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        #[cfg(feature = "babybear")]
        {
            ExtExpReverseBitsChip::<9>.generate_trace(input, output)
        }
        #[cfg(feature = "koalabear")]
        {
            ExtExpReverseBitsChip::<3>.generate_trace(input, output)
        }
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

        #[cfg(feature = "koalabear")]
        {
            let chip = Chip::<ExtExpReverseBitsChipPolyAir, p3_koala_bear::KoalaBear, 5>::new(
                ExtExpReverseBitsChipPolyAir,
            );
            assert_eq!(chip.num_lookup(), 4);
            println!(
                "ExtExpReverseBitsChipPolyAir (eth): degree={}, num_alpha={}, num_lookup={}",
                chip.degree,
                chip.num_alpha,
                chip.num_lookup()
            );
        }
        #[cfg(feature = "babybear")]
        {
            let chip = Chip::<ExtExpReverseBitsChipPolyAir, p3_baby_bear::BabyBear, 4>::new(
                ExtExpReverseBitsChipPolyAir,
            );
            assert_eq!(chip.num_lookup(), 4);
            println!(
                "ExtExpReverseBitsChipPolyAir (legacy): degree={}, num_alpha={}, num_lookup={}",
                chip.degree,
                chip.num_alpha,
                chip.num_lookup()
            );
        }
    }
}
