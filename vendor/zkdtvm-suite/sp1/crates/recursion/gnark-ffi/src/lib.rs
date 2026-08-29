mod babybear;

pub mod ffi;
pub mod groth16_bn254;
pub mod plonk_bn254;
pub mod proof;
pub mod witness;

pub use groth16_bn254::*;
pub use plonk_bn254::*;
pub use proof::*;
pub use witness::*;

/// The global version for all components of zkDTVM.
///
/// This string should be updated whenever any step in verifying a zkDTVM proof changes, including
/// core, recursion, and plonk-bn254. This string is used to download zkDTVM artifacts and the gnark
/// docker image.
const DT_CIRCUIT_VERSION: &str = include_str!("../assets/DT_VERSION");
