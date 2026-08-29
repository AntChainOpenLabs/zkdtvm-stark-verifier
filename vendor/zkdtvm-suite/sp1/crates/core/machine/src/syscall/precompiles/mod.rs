use dt_core_executor::{events::ByteRecord, ExecutionRecord};
use dt_curves::params::FieldParameters;
use p3_field::Field;

use crate::operations::field::range::FieldLtCols;

pub mod edwards;
pub mod fptower;
// pub mod keccak256;
pub mod keccak_dt;
pub mod poseidon_permute;
pub mod sha256;
pub mod u256x2048_mul;
pub mod uint256;
pub mod weierstrass;

pub(crate) fn add_field_lt_bitvec_lookups<F: Field, P: FieldParameters>(
    output: &mut ExecutionRecord,
    cols: &FieldLtCols<F, P>,
) {
    for chunk in cols.byte_flags.0.chunks(16) {
        let mut value = 0u16;
        for (i, flag) in chunk.iter().enumerate() {
            if *flag == F::one() {
                value |= 1u16 << i;
            }
        }
        output.add_bit_vec_lookup(value);
    }
}

pub(crate) const fn required_max_beta_power_for_field<P: FieldParameters>(
    max_lookup_values: usize,
) -> usize {
    let mut max_val = max_lookup_values;
    if P::NB_LIMBS > max_val {
        max_val = P::NB_LIMBS;
    }
    if P::NB_ADD_WITNESS_LIMBS > max_val {
        max_val = P::NB_ADD_WITNESS_LIMBS;
    }
    if P::NB_WITNESS_LIMBS > max_val {
        max_val = P::NB_WITNESS_LIMBS;
    }
    max_val
}
