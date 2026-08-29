mod fp;
mod fp2_addsub;
pub mod fp2_addsub_polyair;
mod fp2_mul;
pub mod fp2_mul_polyair;
pub mod fp_polyair;

pub use fp::*;
pub use fp2_addsub::*;
pub use fp2_addsub_polyair::*;
pub use fp2_mul::*;
pub use fp2_mul_polyair::*;
pub use fp_polyair::*;

#[cfg(test)]
mod tests {
    use dt_stark::CpuProver;

    use dt_core_executor::Program;
    use test_artifacts::{
        BLS12381_FP2_ADDSUB_ELF, BLS12381_FP2_MUL_ELF, BLS12381_FP_ELF, BN254_FP2_ADDSUB_ELF,
        BN254_FP2_MUL_ELF, BN254_FP_ELF,
    };

    use crate::{io::DTStdin, utils};

    #[test]
    fn test_bls12381_fp_ops() {
        utils::setup_logger();
        let program = Program::from(BLS12381_FP_ELF).unwrap();
        let stdin = DTStdin::new();
        utils::run_test::<CpuProver<_, _>>(program, stdin).unwrap();
    }

    #[test]
    fn test_bls12381_fp2_addsub() {
        utils::setup_logger();
        let program = Program::from(BLS12381_FP2_ADDSUB_ELF).unwrap();
        let stdin = DTStdin::new();
        utils::run_test::<CpuProver<_, _>>(program, stdin).unwrap();
    }

    #[test]
    fn test_bls12381_fp2_mul() {
        utils::setup_logger();
        let program = Program::from(BLS12381_FP2_MUL_ELF).unwrap();
        let stdin = DTStdin::new();
        utils::run_test::<CpuProver<_, _>>(program, stdin).unwrap();
    }

    #[test]
    fn test_bn254_fp_ops() {
        utils::setup_logger();
        let program = Program::from(BN254_FP_ELF).unwrap();
        let stdin = DTStdin::new();
        utils::run_test::<CpuProver<_, _>>(program, stdin).unwrap();
    }

    #[test]
    fn test_bn254_fp2_addsub() {
        utils::setup_logger();
        let program = Program::from(BN254_FP2_ADDSUB_ELF).unwrap();
        let stdin = DTStdin::new();
        utils::run_test::<CpuProver<_, _>>(program, stdin).unwrap();
    }

    #[test]
    fn test_bn254_fp2_mul() {
        utils::setup_logger();
        let program = Program::from(BN254_FP2_MUL_ELF).unwrap();
        let stdin = DTStdin::new();
        utils::run_test::<CpuProver<_, _>>(program, stdin).unwrap();
    }
}
