//! Types and methods for subproof verification inside the [`crate::Executor`].

use crate::DTReduceProof;
use dt_stark::MachineVerificationError;

use dt_stark::sumcheck::keys::SCStarkVerifyingKey;

#[cfg(feature = "babybear")]
use dt_stark::baby_bear_poseidon2::SCBabyBearPoseidon2 as CoreSC;
#[cfg(feature = "koalabear")]
use dt_stark::koalabear_poseidon2::koala_bear_poseidon2::SCKoalaBearPoseidon2 as CoreSC;

/// Verifier used in runtime when `dt_zkvm::precompiles::verify::verify_dt_proof` is called. This
/// is then used to sanity check that the user passed in the correct proof; the actual constraints
/// happen in the recursion layer.
///
/// This needs to be passed in rather than written directly since the actual implementation relies
/// on crates in recursion that depend on dt-core.
pub trait SubproofVerifier: Sync + Send {
    fn verify_deferred_proof(
        &self,
        proof: &DTReduceProof<CoreSC>,
        vk: &SCStarkVerifyingKey<CoreSC>,
        vk_hash: [u32; 8],
        committed_value_digest: [u32; 8],
    ) -> Result<(), MachineVerificationError<CoreSC>>;
}

/// A dummy verifier which does nothing.
pub struct NoOpSubproofVerifier;
impl SubproofVerifier for NoOpSubproofVerifier {
    fn verify_deferred_proof(
        &self,
        _proof: &DTReduceProof<CoreSC>,
        _vk: &SCStarkVerifyingKey<CoreSC>,
        _vk_hash: [u32; 8],
        _committed_value_digest: [u32; 8],
    ) -> Result<(), MachineVerificationError<CoreSC>> {
        Ok(())
    }
}
