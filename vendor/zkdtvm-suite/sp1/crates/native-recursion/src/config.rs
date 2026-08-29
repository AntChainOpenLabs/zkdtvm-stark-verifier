//! Native recursion configuration boundary.
//!
//! Keep all child-verifier field, hash, PCS, and transcript aliases behind this
//! module so copied recursion code does not keep depending on OpenVM BabyBear
//! configuration names.

use dt_stark::{
    koalabear_poseidon2::{
        self,
        koala_bear_poseidon2::{self, SCKoalaBearPoseidon2, SCKoalaBearSha256Root},
    },
    sumcheck::config::{MlChallenger, MlCom, MlPcsOpeningProof, MlPcsProverData},
};

pub type SC = SCKoalaBearPoseidon2;
/// The SHA256-hashed config for the final root_shrink (L4) proof. The root
/// proof is host-verified only — never fed back into a circuit — so its PCS
/// Merkle commitments and transcript may use SHA256 instead of Poseidon2.
pub type RootSC = SCKoalaBearSha256Root;
pub type F = dt_stark::Val<SC>;
pub type EF = dt_stark::Challenge<SC>;
pub type Val = F;
pub type Challenge = EF;

pub type Digest = koala_bear_poseidon2::DigestHash;
pub type Perm = koala_bear_poseidon2::Perm;
pub type Challenger = koala_bear_poseidon2::Challenger;
pub type Mlpcs = koala_bear_poseidon2::Mlpcs;

pub type ChildMlChallenger = MlChallenger<SC>;
pub type ChildMlCommitment = MlCom<SC>;
pub type ChildMlPcsOpeningProof = MlPcsOpeningProof<SC>;
pub type ChildMlPcsProverData = MlPcsProverData<SC>;
pub type ChildWhirVerificationTrace = dt_stark::sumcheck::config::MlPcsVerificationTrace<SC>;

/// Active KoalaBear challenge extension dimension.
pub const D_EF: usize = 5;
pub const DIGEST_SIZE: usize = koalabear_poseidon2::DIGEST_SIZE;
pub const CHUNK: usize = 8;
pub const POSEIDON2_WIDTH: usize = 16;
