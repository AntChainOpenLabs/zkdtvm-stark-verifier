pub mod config;
pub mod core;
pub mod folder;
pub mod keys;
pub mod proof;
pub mod prover;
pub mod state;
pub mod test;
pub mod trace;
pub mod types;
pub mod utils;
pub mod verifier;

pub fn use_algebraic_decomp() -> bool {
    #[cfg(feature = "koalabear")]
    {
        return crate::koalabear_poseidon2::whir_config().use_algebraic_decomp();
    }
    #[cfg(all(not(feature = "koalabear"), feature = "babybear"))]
    {
        return crate::babybear_config().use_algebraic_decomp();
    }
    #[cfg(all(not(feature = "koalabear"), not(feature = "babybear")))]
    {
        true
    }
}
