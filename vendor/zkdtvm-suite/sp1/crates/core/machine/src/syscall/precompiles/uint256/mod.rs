mod air;
mod uint256_polyair;

pub use air::*;
pub use uint256_polyair::*;

#[cfg(test)]
mod tests {

    use dt_core_executor::Program;
    use dt_curves::{params::FieldParameters, uint256::U256Field, utils::biguint_from_limbs};
    use dt_stark::CpuProver;
    use test_artifacts::UINT256_MUL_ELF;

    use crate::{
        io::DTStdin,
        utils::{self, run_test},
    };

    #[test]
    fn test_uint256_mul() {
        utils::setup_logger();
        let program = Program::from(UINT256_MUL_ELF).unwrap();
        run_test::<CpuProver<_, _>>(program, DTStdin::new()).unwrap();
    }

    #[test]
    fn test_uint256_modulus() {
        assert_eq!(biguint_from_limbs(U256Field::MODULUS), U256Field::modulus());
    }
}
