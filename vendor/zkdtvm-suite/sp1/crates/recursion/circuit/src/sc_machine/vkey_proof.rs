use std::marker::PhantomData;

use super::compress::{
    SCDTCompressVerifier, SCDTCompressWitnessValues, SCDTCompressWitnessVariable,
};
use crate::{
    challenger::DuplexChallengerVariable,
    hash::{FieldHasher, FieldHasherVariable},
    merkle_tree::{verify, MerkleProof},
    stark::MerkleProofVariable,
    sumcheck::{
        polyair_folder::RecursivePolyAirConstraintFolder,
        polyair_precompute::RecursivePolyAirPrecomputeRowBuilder, types::BasefoldProofVariable,
        SCBabyBearFriConfig, SCBabyBearFriConfigVariable,
    },
    witness::{WitnessWriter, Witnessable},
    CircuitConfig,
};
use dt_recursion_compiler::{
    circuit::CircuitV2Builder,
    ir::{Builder, Felt},
};
use dt_recursion_core::DIGEST_SIZE;
use dt_stark::{
    air::{FullAir, MachineAir, PolyAirExtendable},
    baby_bear_poseidon2::SCBabyBearPoseidon2,
    sumcheck::config::{MlCom, MlPcsOpeningProof, SCStarkGenericConfig},
    Challenge, SCStarkMachine, Val,
};
use p3_air::Air;
use p3_baby_bear::BabyBear;
use p3_commit::Mmcs;
use p3_field::{extension::BinomialExtensionField, AbstractField};
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::SymbolicAirBuilder;
use polyair::SCStarkMachine as PolyAirStarkMachine;
use serde::{Deserialize, Serialize};

use crate::machine::{DTCompressWithVkeyShape, PublicValuesOutputDigest};

/// A program to verify a batch of recursive proofs and aggregate their public values.
#[derive(Debug, Clone, Copy)]
pub struct SCDTMerkleProofVerifier<C, SC> {
    _phantom: PhantomData<(C, SC)>,
}

/// Witness layout for the compress stage verifier.
pub struct SCDTMerkleProofWitnessVariable<
    C: CircuitConfig<F = SC::Val>,
    SC: FieldHasherVariable<C> + SCBabyBearFriConfigVariable<C>,
> {
    /// The shard proofs to verify.
    pub vk_merkle_proofs: Vec<MerkleProofVariable<C, SC>>,
    /// Hinted values to enable dummy digests.
    pub values: Vec<SC::DigestVariable>,
    /// The root of the merkle tree.
    pub root: SC::DigestVariable,
}

/// An input layout for the reduce verifier.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "SC::Digest: Serialize"))]
#[serde(bound(deserialize = "SC::Digest: Deserialize<'de>"))]
pub struct SCDTMerkleProofWitnessValues<SC: FieldHasher<Val<SC>> + dt_stark::StarkGenericConfig> {
    pub vk_merkle_proofs: Vec<MerkleProof<Val<SC>, SC>>,
    pub values: Vec<SC::Digest>,
    pub root: SC::Digest,
}

impl<C, SC> SCDTMerkleProofVerifier<C, SC>
where
    SC: SCBabyBearFriConfigVariable<C>,
    C: CircuitConfig<F = SC::Val, EF = Challenge<SC>>,
{
    /// Verify (via Merkle tree) that the vkey digests of a proof belong to a specified set (encoded
    /// the Merkle tree proofs in input).
    pub fn verify(
        builder: &mut Builder<C>,
        digests: Vec<SC::DigestVariable>,
        input: SCDTMerkleProofWitnessVariable<C, SC>,
        value_assertions: bool,
    ) {
        let SCDTMerkleProofWitnessVariable { vk_merkle_proofs, values, root } = input;
        for ((proof, value), expected_value) in
            vk_merkle_proofs.into_iter().zip(values).zip(digests)
        {
            verify(builder, proof, value, root);
            if value_assertions {
                SC::assert_digest_eq(builder, expected_value, value);
            } else {
                SC::assert_digest_eq(builder, value, value);
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SCDTCompressWithVKeyVerifier<C, SC, A, const D: usize> {
    _phantom: PhantomData<(C, SC, A)>,
}

/// Witness layout for the verifier of the proof shape phase of the compress stage.
pub struct SCDTCompressWithVKeyWitnessVariable<
    C: CircuitConfig<F = SC::Val>,
    SC: SCBabyBearFriConfigVariable<C>,
> {
    pub compress_var: SCDTCompressWitnessVariable<C, SC>,
    pub merkle_var: SCDTMerkleProofWitnessVariable<C, SC>,
}

/// An input layout for the verifier of the proof shape phase of the compress stage.
#[derive(Clone)]
pub struct SCDTCompressWithVKeyWitnessValues<SC: SCStarkGenericConfig + FieldHasher<Val<SC>>> {
    pub compress_val: SCDTCompressWitnessValues<SC>,
    pub merkle_val: SCDTMerkleProofWitnessValues<SC>,
}

impl<C, SC, A, const D: usize> SCDTCompressWithVKeyVerifier<C, SC, A, D>
where
    SC: SCBabyBearFriConfigVariable<C, FriChallengerVariable = DuplexChallengerVariable<C>>
        + FieldHasherVariable<C, DigestVariable = [Felt<C::F>; DIGEST_SIZE]>,
    C: CircuitConfig<F = SC::Val, EF = Challenge<SC>>,
    SC::ValMmcs: Mmcs<SC::Val, ProverData<RowMajorMatrix<SC::Val>>: Clone>,
    A: MachineAir<SC::Val>,
    Val<SC>: PolyAirExtendable<D>,
    Builder<C>: CircuitV2Builder<C>,
{
    /// Verify the proof shape phase of the compress stage.
    pub fn verify(
        builder: &mut Builder<C>,
        machine: &PolyAirStarkMachine<SC, A, D>,
        input: SCDTCompressWithVKeyWitnessVariable<C, SC>,
        value_assertions: bool,
        kind: PublicValuesOutputDigest,
    ) where
        A: for<'a> FullAir<RecursivePolyAirConstraintFolder<'a, C>>,
        A: for<'a> FullAir<RecursivePolyAirPrecomputeRowBuilder<'a, C>>,
    {
        let values = input
            .compress_var
            .vks_and_proofs
            .iter()
            .map(|(vk, _)| vk.hash(builder))
            .collect::<Vec<_>>();
        let vk_root = input.merkle_var.root.map(|x| builder.eval(x));
        SCDTMerkleProofVerifier::verify(builder, values, input.merkle_var, value_assertions);
        SCDTCompressVerifier::verify(builder, machine, input.compress_var, vk_root, kind);
    }
}

impl<SC: SCBabyBearFriConfig + FieldHasher<Val<SC>>> SCDTCompressWithVKeyWitnessValues<SC> {
    pub fn shape(&self) -> DTCompressWithVkeyShape {
        let merkle_tree_height = self.merkle_val.vk_merkle_proofs.first().unwrap().path.len();
        DTCompressWithVkeyShape { compress_shape: self.compress_val.shape(), merkle_tree_height }
    }
}

impl SCDTMerkleProofWitnessValues<SCBabyBearPoseidon2> {
    pub fn dummy(num_proofs: usize, height: usize) -> Self {
        let dummy_digest = [BabyBear::zero(); DIGEST_SIZE];
        let vk_merkle_proofs =
            vec![MerkleProof { index: 0, path: vec![dummy_digest; height] }; num_proofs];
        let values = vec![dummy_digest; num_proofs];

        Self { vk_merkle_proofs, values, root: dummy_digest }
    }
}

impl SCDTCompressWithVKeyWitnessValues<SCBabyBearPoseidon2> {
    pub fn dummy<
        A: MachineAir<BabyBear> + for<'a> Air<SymbolicAirBuilder<BabyBear>>,
        AE: MachineAir<BinomialExtensionField<BabyBear, 4>>,
    >(
        machine: &SCStarkMachine<SCBabyBearPoseidon2, A, AE>,
        shape: &DTCompressWithVkeyShape,
    ) -> Self {
        let compress_val =
            SCDTCompressWitnessValues::<SCBabyBearPoseidon2>::dummy(machine, &shape.compress_shape);
        let num_proofs = compress_val.vks_and_proofs.len();
        let merkle_val = SCDTMerkleProofWitnessValues::<SCBabyBearPoseidon2>::dummy(
            num_proofs,
            shape.merkle_tree_height,
        );
        Self { compress_val, merkle_val }
    }

    pub fn dummy_polyair<A: MachineAir<BabyBear>, const D: usize>(
        machine: &PolyAirStarkMachine<SCBabyBearPoseidon2, A, D>,
        shape: &DTCompressWithVkeyShape,
    ) -> Self
    where
        BabyBear: PolyAirExtendable<D>,
    {
        let compress_val = SCDTCompressWitnessValues::<SCBabyBearPoseidon2>::dummy_polyair(
            machine,
            &shape.compress_shape,
        );
        let num_proofs = compress_val.vks_and_proofs.len();
        let merkle_val = SCDTMerkleProofWitnessValues::<SCBabyBearPoseidon2>::dummy(
            num_proofs,
            shape.merkle_tree_height,
        );
        Self { compress_val, merkle_val }
    }
}

#[cfg(feature = "koalabear")]
impl
    SCDTMerkleProofWitnessValues<
        dt_stark::koalabear_poseidon2::koala_bear_poseidon2::SCKoalaBearPoseidon2,
    >
{
    pub fn dummy(num_proofs: usize, height: usize) -> Self {
        use p3_koala_bear::KoalaBear;
        let dummy_digest = [KoalaBear::zero(); DIGEST_SIZE];
        let vk_merkle_proofs =
            vec![MerkleProof { index: 0, path: vec![dummy_digest; height] }; num_proofs];
        let values = vec![dummy_digest; num_proofs];

        Self { vk_merkle_proofs, values, root: dummy_digest }
    }
}

#[cfg(feature = "koalabear")]
impl
    SCDTCompressWithVKeyWitnessValues<
        dt_stark::koalabear_poseidon2::koala_bear_poseidon2::SCKoalaBearPoseidon2,
    >
{
    pub fn dummy<
        A: MachineAir<p3_koala_bear::KoalaBear>,
        AE: MachineAir<
            Challenge<dt_stark::koalabear_poseidon2::koala_bear_poseidon2::SCKoalaBearPoseidon2>,
        >,
    >(
        machine: &SCStarkMachine<
            dt_stark::koalabear_poseidon2::koala_bear_poseidon2::SCKoalaBearPoseidon2,
            A,
            AE,
        >,
        shape: &DTCompressWithVkeyShape,
    ) -> Self
    where
        A: for<'a> Air<SymbolicAirBuilder<p3_koala_bear::KoalaBear>>,
    {
        type KBConfig = dt_stark::koalabear_poseidon2::koala_bear_poseidon2::SCKoalaBearPoseidon2;
        let compress_val =
            SCDTCompressWitnessValues::<KBConfig>::dummy(machine, &shape.compress_shape);
        let num_proofs = compress_val.vks_and_proofs.len();
        let merkle_val =
            SCDTMerkleProofWitnessValues::<KBConfig>::dummy(num_proofs, shape.merkle_tree_height);
        Self { compress_val, merkle_val }
    }

    pub fn dummy_polyair<A: MachineAir<p3_koala_bear::KoalaBear>, const D: usize>(
        machine: &PolyAirStarkMachine<
            dt_stark::koalabear_poseidon2::koala_bear_poseidon2::SCKoalaBearPoseidon2,
            A,
            D,
        >,
        shape: &DTCompressWithVkeyShape,
    ) -> Self
    where
        p3_koala_bear::KoalaBear: PolyAirExtendable<D>,
    {
        type KBConfig = dt_stark::koalabear_poseidon2::koala_bear_poseidon2::SCKoalaBearPoseidon2;
        let compress_val =
            SCDTCompressWitnessValues::<KBConfig>::dummy_polyair(machine, &shape.compress_shape);
        let num_proofs = compress_val.vks_and_proofs.len();
        let merkle_val =
            SCDTMerkleProofWitnessValues::<KBConfig>::dummy(num_proofs, shape.merkle_tree_height);
        Self { compress_val, merkle_val }
    }
}

impl<
        C: CircuitConfig<F = SC::Val, EF = Challenge<SC>>,
        SC: SCBabyBearFriConfigVariable<C> + dt_stark::StarkGenericConfig,
    > Witnessable<C> for SCDTCompressWithVKeyWitnessValues<SC>
where
    SC: FieldHasher<Val<SC>>,
    <SC as FieldHasher<Val<SC>>>::Digest: Witnessable<C, WitnessVariable = SC::DigestVariable>,
    MlCom<SC>: Witnessable<C, WitnessVariable = <SC as FieldHasherVariable<C>>::DigestVariable>,
    MlPcsOpeningProof<SC>: Witnessable<C, WitnessVariable = BasefoldProofVariable<C, SC>>,
    SCDTCompressWitnessValues<SC>:
        Witnessable<C, WitnessVariable = SCDTCompressWitnessVariable<C, SC>>,
    SCDTMerkleProofWitnessValues<SC>:
        Witnessable<C, WitnessVariable = SCDTMerkleProofWitnessVariable<C, SC>>,
{
    type WitnessVariable = SCDTCompressWithVKeyWitnessVariable<C, SC>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        SCDTCompressWithVKeyWitnessVariable {
            compress_var: self.compress_val.read(builder),
            merkle_var: self.merkle_val.read(builder),
        }
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        self.compress_val.write(witness);
        self.merkle_val.write(witness);
    }
}
