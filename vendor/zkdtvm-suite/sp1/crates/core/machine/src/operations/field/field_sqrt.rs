use std::fmt::Debug;

use dt_curves::params::{limbs_from_vec, FieldParameters, Limbs};
use dt_derive::AlignedBorrow;
use num::BigUint;
use p3_air::AirBuilder;
use p3_field::Field;

use dt_core_executor::{
    events::{ByteLookupEvent, ByteRecord, FieldOperation},
    ByteOpcode,
};
use dt_stark::air::DTAirBuilder;

use super::{field_op::FieldOpCols, range::FieldLtCols};
use crate::air::WordAirBuilder;
use p3_field::AbstractField;

/// A set of columns to compute the square root in emulated arithmetic.
///
/// *Safety*: The `FieldSqrtCols` asserts that `multiplication.result` is a square root of the given
/// input lying within the range `[0, modulus)` with the least significant bit `lsb`.
#[derive(Debug, Clone, AlignedBorrow)]
#[repr(C)]
pub struct FieldSqrtCols<T, P: FieldParameters> {
    /// The multiplication operation to verify that the sqrt and the input match.
    ///
    /// In order to save space, we actually store the sqrt of the input in `multiplication.result`
    /// since we'll receive the input again in the `eval` function.
    pub multiplication: FieldOpCols<T, P>,

    pub range: FieldLtCols<T, P>,

    // The least significant bit of the square root.
    pub lsb: T,
}

impl<F: Field, P: FieldParameters> FieldSqrtCols<F, P> {
    /// Populates the trace.
    ///
    /// `P` is the parameter of the field that each limb lives in.
    pub fn populate(
        &mut self,
        record: &mut impl ByteRecord,
        a: &BigUint,
        sqrt_fn: impl Fn(&BigUint) -> BigUint,
    ) -> BigUint {
        let modulus = P::modulus();
        assert!(a < &modulus);
        let sqrt = sqrt_fn(a);

        // Use FieldOpCols to compute result * result.
        let sqrt_squared = self.multiplication.populate(record, &sqrt, &sqrt, FieldOperation::Mul);

        // If the result is indeed the square root of a, then result * result = a.
        assert_eq!(sqrt_squared, a.clone());

        // This is a hack to save a column in FieldSqrtCols. We will receive the value a again in
        // the eval function, so we'll overwrite it with the sqrt.
        self.multiplication.result = P::to_limbs_field::<F, _>(&sqrt);

        // Populate the range columns.
        self.range.populate(record, &sqrt, &modulus);

        let sqrt_bytes = P::to_limbs(&sqrt);
        self.lsb = F::from_canonical_u8(sqrt_bytes[0] & 1);

        let and_event = ByteLookupEvent {
            opcode: ByteOpcode::AND,
            a1: self.lsb.as_u32() as u16,
            a2: 0,
            b: sqrt_bytes[0],
            c: 1,
        };
        record.add_byte_lookup_event(and_event);

        // Add the byte range check for `sqrt`.
        record.add_u8_range_checks(
            self.multiplication
                .result
                .0
                .as_slice()
                .iter()
                .map(|x| x.as_u32() as u8)
                .collect::<Vec<_>>()
                .as_slice(),
        );

        sqrt
    }
}

impl<V: Copy, P: FieldParameters> FieldSqrtCols<V, P>
where
    Limbs<V, P::Limbs>: Copy,
{
    /// Calculates the square root of `a`.
    pub fn eval<AB: DTAirBuilder<Var = V>>(
        &self,
        builder: &mut AB,
        a: &Limbs<AB::Var, P::Limbs>,
        is_odd: impl Into<AB::Expr>,
        is_real: impl Into<AB::Expr> + Clone,
    ) where
        V: Into<AB::Expr>,
    {
        // As a space-saving hack, we store the sqrt of the input in `self.multiplication.result`
        // even though it's technically not the result of the multiplication. Now, we should
        // retrieve that value and overwrite that member variable with a.
        let sqrt = self.multiplication.result;
        let mut multiplication = self.multiplication.clone();
        multiplication.result = *a;

        // Compute sqrt * sqrt. We pass in P since we want its BaseField to be the mod.
        multiplication.eval(builder, &sqrt, &sqrt, FieldOperation::Mul, is_real.clone());

        let modulus_limbs = P::to_limbs_field_vec(&P::modulus());
        self.range.eval(
            builder,
            &sqrt,
            &limbs_from_vec::<AB::Expr, P::Limbs, AB::F>(modulus_limbs),
            is_real.clone(),
        );

        // Range check that `sqrt` limbs are bytes.
        builder.slice_range_check_u8(sqrt.0.as_slice(), is_real.clone());

        // Assert that the square root is the positive one, i.e., with least significant bit 0.
        // This is done by computing LSB = least_significant_byte & 1.
        builder.assert_bool(self.lsb);
        builder.when(is_real.clone()).assert_eq(self.lsb, is_odd);
        builder.send_byte(
            ByteOpcode::AND.as_field::<AB::F>(),
            self.lsb,
            sqrt[0],
            AB::F::one(),
            is_real,
        );
    }
}

// ============================================================================
// PolyAir helpers for FieldSqrtCols
// ============================================================================

/// Number of interactions emitted by a `FieldSqrtCols` instance.
///
/// - FieldOpCols (Mul: sqrt * sqrt): `field_op_num_interactions`
/// - FieldLtCols (range check sqrt < modulus): `field_lt_num_interactions`
/// - U8Range for sqrt limbs: `P::NB_LIMBS / 2`
/// - AND byte (lsb = sqrt[0] & 1): `1`
pub const fn field_sqrt_num_interactions<P: FieldParameters>() -> usize {
    super::field_op::field_op_num_interactions::<P>() +
        super::range::field_lt_num_interactions::<P>() +
        P::NB_LIMBS / 2 +
        1
}

/// Precompute lookup denominators for a `FieldSqrtCols` instance.
///
/// `a_limbs` must be the actual multiplication result (= sqrt²) **before** the
/// `FieldSqrtCols` hack that overwrites `multiplication.result` with sqrt.
/// Using `a_limbs` here ensures the field_op denominator matches what the base
/// chip's `generate_dependencies` emits (U8Range for sqrt²), while
/// `slice_u8_range_precompute_lc` below covers sqrt itself.
///
/// Order matches `FieldSqrtCols::eval`:
/// 1. FieldOpCols (Mul: sqrt * sqrt = input) — result denominator uses `a_limbs`
/// 2. FieldLtCols (range: sqrt < modulus)
/// 3. U8Range for sqrt limbs (NB_LIMBS/2 pairs)
/// 4. AND byte (lsb = sqrt[0] & 1)
pub fn field_sqrt_precompute_lc<AB: dt_stark::air::FullAirBuilder, P: FieldParameters>(
    builder: &mut AB,
    cols: &FieldSqrtCols<AB::VarMaybeExt, P>,
    a_limbs: &[AB::VarMaybeExt],
) where
    AB::VarMaybeExt: Clone,
{
    use crate::bytes::polyair::{and_precompute_lc, slice_u8_range_precompute_lc};

    let sqrt_limbs: Vec<AB::VarMaybeExt> = cols.multiplication.result.0.iter().cloned().collect();

    // 1. FieldOpCols for sqrt * sqrt verification — use a_limbs (= sqrt²) as result so the
    //    denominator matches the base chip's BLU emission from multiplication.populate(sqrt, sqrt,
    //    Mul) before the result overwrite.
    super::field_op::field_op_precompute_lc::<AB, P>(
        builder,
        a_limbs,
        &cols.multiplication.carry.0.iter().cloned().collect::<Vec<_>>(),
        &cols.multiplication.witness.0.iter().cloned().collect::<Vec<_>>(),
    );

    // 2. FieldLtCols for range check
    {
        let flags: Vec<AB::VarMaybeExt> = cols.range.byte_flags.0.iter().cloned().collect();
        super::range::field_lt_precompute_lc::<AB, P>(
            builder,
            cols.range.lhs_comparison_byte.clone(),
            cols.range.rhs_comparison_byte.clone(),
            &flags,
        );
    }

    // 3. U8Range for sqrt limbs
    slice_u8_range_precompute_lc(builder, &sqrt_limbs);

    // 4. AND byte: lsb = sqrt[0] & 1
    let one_felt = AB::VarMaybeExt::from(AB::F::one());
    and_precompute_lc(builder, cols.lsb.clone(), sqrt_limbs[0].clone(), one_felt);
}

/// Gate constraints for a `FieldSqrtCols` instance.
///
/// Reproduces `FieldSqrtCols::eval` gate constraints:
/// 1. Mul verification: sqrt * sqrt = input (via field_op_mul_gate_constraints) Note the hack:
///    `multiplication.result` stores sqrt, actual input is `a`.
/// 2. FieldLtCols gate constraints (sqrt < modulus)
/// 3. assert_bool(lsb)
/// 4. when(is_real).assert_eq(lsb, is_odd)
pub fn field_sqrt_gate_constraints<AB: dt_stark::air::FullAirBuilder, P: FieldParameters>(
    builder: &mut AB,
    input_limbs: &[AB::VarMaybeExt],
    cols: &FieldSqrtCols<AB::VarMaybeExt, P>,
    is_odd: AB::VarMaybeExt,
    is_real: AB::VarMaybeExt,
    consts: &super::field_op::FieldOpBetaConsts<AB>,
) where
    AB::VarMaybeExt: Clone,
{
    let sqrt_limbs: Vec<AB::VarMaybeExt> = cols.multiplication.result.0.iter().cloned().collect();

    // 1. Mul verification: sqrt * sqrt = input
    // The hack: multiplication.result stores sqrt, but the Mul constraint checks
    // sqrt * sqrt = a (input). We create a temporary FieldOpCols with result = input.
    {
        use dt_curves::params::Limbs;

        let mut mul_cols = cols.multiplication.clone();
        // Overwrite result with the actual input (a)
        mul_cols.result = Limbs(
            input_limbs
                .iter()
                .cloned()
                .collect::<Vec<_>>()
                .try_into()
                .unwrap_or_else(|_| panic!("input_limbs length mismatch")),
        );
        super::field_op::field_op_mul_gate_constraints::<AB, P>(
            builder,
            &sqrt_limbs,
            &sqrt_limbs,
            &mul_cols,
            super::field_op::field_op_witness_beta_from_coeffs::<AB, P>(
                builder,
                &mul_cols.witness.0.iter().cloned().collect::<Vec<_>>(),
            ),
            consts,
        );
    }

    // 2. FieldLtCols gate constraints (sqrt < modulus)
    {
        let modulus_limbs: Vec<AB::VarMaybeExt> = P::MODULUS
            .iter()
            .map(|&x| AB::VarMaybeExt::from(AB::F::from_canonical_u8(x)))
            .collect();
        super::range::field_lt_gate_constraints::<AB, P>(
            builder,
            &sqrt_limbs,
            &modulus_limbs,
            &cols.range,
            is_real.clone(),
        );
    }

    // 3. assert_bool(lsb)
    let one = AB::one_maybe();
    builder.assert_zero(cols.lsb.clone() * (one - cols.lsb.clone()));

    // 4. when(is_real).assert_eq(lsb, is_odd)
    builder.assert_zero(is_real * (cols.lsb.clone() - is_odd));
}

/// Declare multiplicities for a `FieldSqrtCols` instance's lookups.
///
/// Order matches `field_sqrt_precompute_lc`:
/// 1. FieldOpCols (Mul)
/// 2. FieldLtCols (range)
/// 3. U8Range for sqrt limbs
/// 4. AND byte
pub fn field_sqrt_lookup<AB: dt_stark::air::FullAirBuilder, P: FieldParameters>(
    builder: &mut AB,
    is_real: AB::VarMaybeExt,
) {
    use crate::bytes::polyair::{and_lookup, slice_u8_range_lookup};

    // 1. FieldOpCols
    super::field_op::field_op_lookup::<AB, P>(builder, is_real.clone());
    // 2. FieldLtCols
    super::range::field_lt_lookup::<AB, P>(builder, is_real.clone());
    // 3. U8Range for sqrt limbs
    slice_u8_range_lookup(builder, is_real.clone(), P::NB_LIMBS / 2);
    // 4. AND byte
    and_lookup(builder, is_real);
}

#[cfg(test)]
mod tests {
    use dt_core_executor::{ExecutionRecord, Program};
    use dt_curves::params::{FieldParameters, Limbs};
    use dt_stark::{
        air::{DTAirBuilder, MachineAir},
        sumcheck::trace::CompressedMatrix,
    };
    use num::{BigUint, One, Zero};
    use p3_air::BaseAir;
    use p3_field::Field;

    use crate::utils::{
        pad_to_power_of_two,
        uni_stark::{uni_stark_prove, uni_stark_verify},
    };
    use core::{
        borrow::{Borrow, BorrowMut},
        mem::size_of,
    };
    use dt_core_executor::events::ByteRecord;
    use dt_curves::edwards::ed25519::{ed25519_sqrt, Ed25519BaseField};
    use dt_derive::AlignedBorrow;
    use dt_stark::{baby_bear_poseidon2::BabyBearPoseidon2, StarkGenericConfig};
    use num::bigint::RandBigInt;
    use p3_air::Air;
    use p3_baby_bear::BabyBear;
    use p3_field::AbstractField;
    use p3_matrix::{dense::RowMajorMatrix, Matrix};
    use rand::thread_rng;

    use super::FieldSqrtCols;

    #[derive(AlignedBorrow, Debug)]
    pub struct TestCols<T, P: FieldParameters> {
        pub a: Limbs<T, P::Limbs>,
        pub sqrt: FieldSqrtCols<T, P>,
    }

    pub const NUM_TEST_COLS: usize = size_of::<TestCols<u8, Ed25519BaseField>>();

    struct EdSqrtChip<P: FieldParameters> {
        pub _phantom: std::marker::PhantomData<P>,
    }

    impl<P: FieldParameters> EdSqrtChip<P> {
        pub const fn new() -> Self {
            Self { _phantom: std::marker::PhantomData }
        }
    }

    impl<F: Field, P: FieldParameters> MachineAir<F> for EdSqrtChip<P> {
        type Record = ExecutionRecord;

        type Program = Program;

        fn name(&self) -> String {
            "EdSqrtChip".to_string()
        }

        fn generate_trace(
            &self,
            _: &ExecutionRecord,
            output: &mut ExecutionRecord,
        ) -> CompressedMatrix<F> {
            let mut rng = thread_rng();
            let num_rows = 1 << 8;
            let mut operands: Vec<BigUint> = (0..num_rows - 2)
                .map(|_| {
                    // Take the square of a random number to make sure that the square root exists.
                    let a = rng.gen_biguint(256);
                    let sq = a.clone() * a.clone();
                    // We want to mod by the ed25519 modulus.
                    sq % &Ed25519BaseField::modulus()
                })
                .collect();

            // hardcoded edge cases.
            operands.extend(vec![BigUint::zero(), BigUint::one()]);

            let rows = operands
                .iter()
                .map(|a| {
                    let mut blu_events = Vec::new();
                    let mut row = [F::zero(); NUM_TEST_COLS];
                    let cols: &mut TestCols<F, P> = row.as_mut_slice().borrow_mut();
                    cols.a = P::to_limbs_field::<F, _>(a);
                    cols.sqrt.populate(&mut blu_events, a, |v| ed25519_sqrt(v).unwrap());
                    output.add_byte_lookup_events(blu_events);
                    row
                })
                .collect::<Vec<_>>();
            // Convert the trace to a row major matrix.
            let mut trace =
                RowMajorMatrix::new(rows.into_iter().flatten().collect::<Vec<_>>(), NUM_TEST_COLS);

            // Pad the trace to a power of two.
            pad_to_power_of_two::<NUM_TEST_COLS, F>(&mut trace.values);

            CompressedMatrix::from_full_matrix_no_padding(trace)
        }

        fn included(&self, _: &Self::Record) -> bool {
            true
        }
    }

    impl<F: Field, P: FieldParameters> BaseAir<F> for EdSqrtChip<P> {
        fn width(&self) -> usize {
            NUM_TEST_COLS
        }
    }

    impl<AB, P: FieldParameters> Air<AB> for EdSqrtChip<P>
    where
        AB: DTAirBuilder,
        Limbs<AB::Var, P::Limbs>: Copy,
    {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let local = main.row_slice(0);
            let local: &TestCols<AB::Var, P> = (*local).borrow();

            // eval verifies that local.sqrt.result is indeed the square root of local.a.
            local.sqrt.eval(builder, &local.a, AB::F::zero(), AB::F::one());
        }
    }

    #[test]
    fn generate_trace() {
        let chip: EdSqrtChip<Ed25519BaseField> = EdSqrtChip::new();
        let shard = ExecutionRecord::default();
        let _: CompressedMatrix<BabyBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default());
    }

    #[test]
    fn prove_babybear() {
        let config = BabyBearPoseidon2::new();
        let mut challenger = config.challenger();

        let chip: EdSqrtChip<Ed25519BaseField> = EdSqrtChip::new();
        let shard = ExecutionRecord::default();
        let trace: CompressedMatrix<BabyBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default());
        let proof =
            uni_stark_prove::<BabyBearPoseidon2, _>(&config, &chip, &mut challenger, trace.main);

        let mut challenger = config.challenger();
        uni_stark_verify(&config, &chip, &mut challenger, &proof).unwrap();
    }
}
