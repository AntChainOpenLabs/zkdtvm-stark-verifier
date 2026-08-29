use crate::{
    hash::FieldHasherVariable,
    sumcheck::{
        types::{
            BasefoldProofVariable, InputPrunedVariable, PrunedBatchProofVariable,
            PrunedFriQueryProofVariable, SCShardProofVariable, StackingReductionProofVariable,
            SumcheckInstanceProofVariable, SumcheckProofVariable, UniPolyVariable,
            WhirIoppRoundQueryVariable, WhirIoppRoundVariable, WhirPrunedIoppRoundVariable,
            WhirRoundPrunedQueryProofVariable, WhirRoundQueryProofVariable,
        },
        SCBabyBearFriConfigVariable,
    },
    witness::{WitnessWriter, Witnessable},
    BatchOpeningVariable, CircuitConfig, FriCommitPhaseProofStepVariable, FriQueryProofVariable,
};
#[cfg(feature = "babybear")]
use dt_recursion_compiler::config::OuterConfig;
use dt_recursion_compiler::ir::{Builder, Ext, Felt};
#[cfg(feature = "babybear")]
use dt_recursion_core::stark::{
    OuterBasefoldProof, OuterBatchOpening, OuterChallenge, OuterChallengeMmcs, OuterVal,
    SCBabyBearPoseidon2Outer,
};
#[cfg(feature = "koalabear")]
use dt_stark::koalabear_poseidon2::koala_bear_poseidon2::{
    Challenge as KBInnerChallenge, ChallengeMmcs as KBInnerChallengeMmcs, SCKoalaBearPoseidon2,
    Val as KBInnerVal, ValMmcs as KBInnerValMmcs,
};
use dt_stark::{
    baby_bear_poseidon2::SCBabyBearPoseidon2,
    sumcheck::{
        config::{MlCom, MlPcsOpeningProof},
        proof::{
            SCChipOpenedValues, SCShardCommitment, SCShardOpenedValues, SCShardProof, SumcheckProof,
        },
        types::UniPolyEvals,
    },
    Challenge, InnerBasefoldProof, InnerBatchOpening, InnerChallenge, InnerChallengeMmcs, InnerVal,
    Val,
};
use p3_commit::Mmcs;
use p3_field::Field;
use p3_fri::{BatchOpening, CommitPhaseProofStep, QueryProof};
use p3_merkle_tree::{compute_bfs_schedule, BfsStep, PrunedBatchProof};
#[cfg(feature = "koalabear")]
use pcs::basefold::basefold_pcs::{BasefoldInputProof, BasefoldProof};
use pcs::{
    basefold::sumcheck::SumcheckInstanceProof, utils::unipoly::UniPoly, whir::WhirRoundQueryProof,
};
use std::collections::BTreeMap;

#[cfg(feature = "koalabear")]
type KBInnerBasefoldProof = BasefoldProof<
    KBInnerChallenge,
    KBInnerChallengeMmcs,
    KBInnerVal,
    BasefoldInputProof<KBInnerVal, KBInnerValMmcs>,
>;
#[cfg(feature = "koalabear")]
type KBInnerBatchOpening = BatchOpening<KBInnerVal, KBInnerValMmcs>;

// ============================================================
// Path-pruning witness bridge (C-B1-step3).
//
// Generic over W (hash output element) and D (digest size); 4 callers
// monomorphise it (BB inner / BB outer / KB inner / KB outer). The trait
// bound `[W; D]: Witnessable<C, WitnessVariable = H::DigestVariable>`
// is satisfied at each caller because:
//   - sp1 already has `impl<C, F, W, D> Witnessable<C> for Hash<F, W, D>` producing
//     `[W::WitnessVariable; D]`,
//   - and sp1 already has `FieldHasherVariable` impls fixing `H::DigestVariable` to `[Felt<C::F>;
//     D]` (BB/KB inner) or `[Var<Bn254Fr>; D]` (BB/KB outer), matching what `[W; D]` reads to.
//
// Schedule is computed in native land via `compute_bfs_schedule` and
// shipped as a hint (Felt 0/1 mask + sibling cursor + per-layer counts).
// Stage 2 will add circuit constraints recomputing the mask from
// `sorted_indices` LSBs to neutralise risk R1 (mal-formed hint).
// ============================================================

#[allow(dead_code)]
fn read_pruned_batch_proof_var<C, H, W, const D: usize>(
    p: &PrunedBatchProof<W, D>,
    log_max_height: usize,
    builder: &mut Builder<C>,
) -> crate::sumcheck::types::PrunedBatchProofVariable<C, H>
where
    C: CircuitConfig,
    H: FieldHasherVariable<C>,
    [W; D]: Witnessable<C, WitnessVariable = H::DigestVariable>,
{
    use p3_field::AbstractField;

    let sorted_indices: Vec<Felt<C::F>> =
        p.sorted_indices.iter().map(|i| builder.eval(C::F::from_canonical_u32(*i))).collect();
    let siblings: Vec<H::DigestVariable> = p.siblings.iter().map(|d| d.read(builder)).collect();
    let layer_widths: Vec<Felt<C::F>> =
        p.layer_widths.iter().map(|w| builder.eval(C::F::from_canonical_u32(*w))).collect();

    let schedule = compute_bfs_schedule(&p.sorted_indices, log_max_height);

    let per_layer_step_count: Vec<Felt<C::F>> = schedule
        .layer_active_size
        .iter()
        .map(|s| builder.eval(C::F::from_canonical_u32(*s)))
        .collect();

    // Re-run the native BFS active-walk in lock-step with `schedule.per_layer_steps`
    // to compute, for each ConsumeSibling step, the LSB (0/1) of the active
    // node's index at that layer. This LSB tells the circuit whether the
    // active digest goes on the left (0) or right (1) of the pair-merge
    // with the consumed sibling.
    //
    // Bug-fix: previously this loop pushed a running cursor (0/1/2/3/...)
    // into per_layer_sibling_pos, which `pcs.rs::verify_batch_pruned` then
    // fed into `num2bits(_, 1)` expecting a 0/1 swap-bit. When cursor >= 2
    // the high bits are non-zero, num2bits_f_circuit's high-bit-zero
    // assertion triggered an inverse(0) -> RuntimeError "DivF X/0".
    let mut active_walk: Vec<usize> = p.sorted_indices.iter().map(|&i| i as usize).collect();
    let mut per_layer_pair_merge_mask: Vec<Vec<Felt<C::F>>> =
        Vec::with_capacity(schedule.layer_count);
    let mut per_layer_sibling_pos: Vec<Vec<bool>> = Vec::with_capacity(schedule.layer_count);
    for layer_steps in schedule.per_layer_steps.iter() {
        let mut mask_row: Vec<Felt<C::F>> = Vec::with_capacity(layer_steps.len());
        let mut pos_row: Vec<bool> = Vec::with_capacity(layer_steps.len());
        let mut active_idx: usize = 0;
        for step in layer_steps.iter() {
            // active node consumed at this step (always the leftmost
            // unconsumed entry in `active_walk`).
            let active_node = active_walk[active_idx];
            // pos = active_node's position in its parent pair (0=left, 1=right).
            // For PairMerge this is unused (mask=1 path), but we still emit
            // a valid bool to keep indexing aligned with the native schedule.
            let pos_bit = (active_node & 1) != 0;
            pos_row.push(pos_bit);
            let bit_val = match step {
                BfsStep::PairMerge => {
                    // Two adjacent sibling pair: skip both.
                    active_idx += 2;
                    C::F::one()
                }
                BfsStep::ConsumeSibling => {
                    active_idx += 1;
                    C::F::zero()
                }
            };
            mask_row.push(builder.eval(bit_val));
        }
        per_layer_pair_merge_mask.push(mask_row);
        per_layer_sibling_pos.push(pos_row);

        // Lift to parents (mirror native compute_bfs_schedule).
        let mut parents: Vec<usize> = active_walk.iter().map(|&idx| idx >> 1).collect();
        parents.dedup();
        active_walk = parents;
    }

    crate::sumcheck::types::PrunedBatchProofVariable {
        sorted_indices,
        siblings,
        layer_widths,
        per_layer_step_count,
        per_layer_pair_merge_mask,
        per_layer_sibling_pos,
        native_schedule: Some(schedule),
    }
}

fn read_pruned_fri_opened_values<C, E>(
    values: &[Vec<Vec<Vec<E>>>],
    builder: &mut Builder<C>,
) -> Vec<Vec<Vec<Vec<Ext<C::F, C::EF>>>>>
where
    C: CircuitConfig,
    E: Witnessable<C, WitnessVariable = Ext<C::F, C::EF>>,
{
    values
        .iter()
        .map(|round| {
            round
                .iter()
                .map(|slot| {
                    slot.iter()
                        .map(|matrix| matrix.iter().map(|value| value.read(builder)).collect())
                        .collect()
                })
                .collect()
        })
        .collect()
}

fn write_pruned_fri_opened_values<C, E>(
    values: &[Vec<Vec<Vec<E>>>],
    witness: &mut impl WitnessWriter<C>,
) where
    C: CircuitConfig,
    E: Witnessable<C, WitnessVariable = Ext<C::F, C::EF>>,
{
    for round in values {
        for slot in round {
            for matrix in slot {
                matrix.write(witness);
            }
        }
    }
}

pub trait SCWitnessable<C: CircuitConfig> {
    type SCWitnessVariable;

    fn sc_read(&self, builder: &mut Builder<C>) -> Self::SCWitnessVariable;

    fn sc_write(&self, witness: &mut impl WitnessWriter<C>);
}

fn read_whir_round_iopp<C, H, EF, M, W, ProofElem, const DIGEST_ELEMS: usize>(
    proof: &WhirRoundQueryProof<EF, M, W>,
    builder: &mut Builder<C>,
) -> WhirRoundQueryProofVariable<C, H>
where
    C: CircuitConfig,
    H: FieldHasherVariable<C>,
    EF: Field + Witnessable<C, WitnessVariable = Ext<C::F, C::EF>>,
    M: Mmcs<EF, PrunedProof = PrunedBatchProof<ProofElem, DIGEST_ELEMS>>,
    W: Witnessable<C, WitnessVariable = Felt<C::F>>,
    [ProofElem; DIGEST_ELEMS]: Witnessable<C, WitnessVariable = H::DigestVariable>,
    CommitPhaseProofStep<EF, M>:
        SCWitnessable<C, SCWitnessVariable = FriCommitPhaseProofStepVariable<C, H>>,
{
    let rounds = proof
        .rounds
        .iter()
        .map(|round| WhirIoppRoundVariable {
            query_proofs: round
                .query_proofs
                .iter()
                .map(|query| {
                    assert!(
                        query.next_opening.is_none(),
                        "WHIR recursion expects accumulator-bound folded values",
                    );
                    WhirIoppRoundQueryVariable {
                        current_opening: query.current_opening.sc_read(builder),
                    }
                })
                .collect(),
        })
        .collect();
    let pruned = proof.pruned.as_ref().map(|pruned| WhirRoundPrunedQueryProofVariable {
        rounds: pruned
            .rounds
            .iter()
            .map(|round| WhirPrunedIoppRoundVariable {
                pruned_proof: read_pruned_batch_proof_var(
                    &round.pruned_proof,
                    round.pruned_proof.layer_widths.len(),
                    builder,
                ),
                opened_rows: round
                    .opened_rows
                    .iter()
                    .map(|slot| {
                        slot.iter()
                            .map(|matrix| matrix.iter().map(|value| value.read(builder)).collect())
                            .collect()
                    })
                    .collect(),
                query_to_unique_slot: round
                    .query_to_unique_slot
                    .iter()
                    .map(|&slot| slot as usize)
                    .collect(),
            })
            .collect(),
    });
    let query_witnesses = proof.query_witnesses.iter().map(|w| w.read(builder)).collect();
    let folding_witnesses = proof.folding_witnesses.iter().map(|w| w.read(builder)).collect();
    WhirRoundQueryProofVariable { rounds, pruned, query_witnesses, folding_witnesses }
}

fn write_whir_round_iopp<C, EF, M, W, ProofElem, const DIGEST_ELEMS: usize>(
    proof: &WhirRoundQueryProof<EF, M, W>,
    witness: &mut impl WitnessWriter<C>,
) where
    C: CircuitConfig,
    EF: Field + Witnessable<C>,
    M: Mmcs<EF, PrunedProof = PrunedBatchProof<ProofElem, DIGEST_ELEMS>>,
    W: Witnessable<C, WitnessVariable = Felt<C::F>>,
    [ProofElem; DIGEST_ELEMS]: Witnessable<C>,
    CommitPhaseProofStep<EF, M>: SCWitnessable<C>,
{
    for round in proof.rounds.iter() {
        for query in round.query_proofs.iter() {
            assert!(
                query.next_opening.is_none(),
                "WHIR recursion expects accumulator-bound folded values",
            );
            query.current_opening.sc_write(witness);
        }
    }
    if let Some(pruned) = proof.pruned.as_ref() {
        for round in pruned.rounds.iter() {
            round.pruned_proof.siblings.write(witness);
            for slot in round.opened_rows.iter() {
                for matrix in slot.iter() {
                    matrix.write(witness);
                }
            }
        }
    }
    for round_witness in proof.query_witnesses.iter() {
        round_witness.write(witness);
    }
    for folding_witness in proof.folding_witnesses.iter() {
        folding_witness.write(witness);
    }
}

impl<
        C: CircuitConfig<F = Val<SC>, EF = Challenge<SC>>,
        SC: SCBabyBearFriConfigVariable<C> + dt_stark::StarkGenericConfig,
    > Witnessable<C> for SCShardProof<SC>
where
    MlCom<SC>: Witnessable<C, WitnessVariable = <SC as FieldHasherVariable<C>>::DigestVariable>,
    MlPcsOpeningProof<SC>: Witnessable<C, WitnessVariable = BasefoldProofVariable<C, SC>>,
    Val<SC>: Witnessable<C, WitnessVariable = Felt<C::F>>,
    SumcheckProof<SC>: Witnessable<C, WitnessVariable = SumcheckProofVariable<C>>,
    SCShardOpenedValues<Val<SC>, Challenge<SC>>:
        Witnessable<C, WitnessVariable = SCShardOpenedValues<Felt<C::F>, Ext<C::F, C::EF>>>,
{
    type WitnessVariable = SCShardProofVariable<C, SC>;
    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let commitment = self.commitment.read(builder);
        let opened_values = self.opened_values.read(builder);
        let opening_proof = self.opening_proof.read(builder);
        let sumcheck_proof = self.sumcheck_proof.read(builder);
        let dimensions = self.dimensions.clone();
        let public_values = self.public_values.read(builder);
        let chip_ordering = self.chip_ordering.clone();

        SCShardProofVariable {
            commitment,
            opened_values,
            opening_proof,
            sumcheck_proof,
            dimensions,
            public_values,
            chip_ordering,
        }
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        self.commitment.write(witness);
        self.opened_values.write(witness);
        self.opening_proof.write(witness);
        self.sumcheck_proof.write(witness);
        self.public_values.write(witness);
    }
}

impl<C: CircuitConfig, T: Witnessable<C>> Witnessable<C> for SCShardCommitment<T> {
    type WitnessVariable = SCShardCommitment<T::WitnessVariable>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let main_commit = self.main_commit.read(builder);
        let permutation_commit = match self.permutation_commit.as_ref() {
            Some(commit) => Some(commit.read(builder)),
            None => None,
        };
        Self::WitnessVariable { main_commit, permutation_commit }
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        self.main_commit.write(witness);
        if let Some(commit) = self.permutation_commit.as_ref() {
            commit.write(witness)
        };
    }
}

impl<C: CircuitConfig<F = InnerVal, EF = InnerChallenge>> Witnessable<C>
    for SCShardOpenedValues<InnerVal, InnerChallenge>
{
    type WitnessVariable = SCShardOpenedValues<Felt<C::F>, Ext<C::F, C::EF>>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let chips = self.chips.read(builder);
        Self::WitnessVariable { chips, _field: core::marker::PhantomData }
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        self.chips.write(witness);
    }
}

impl<C: CircuitConfig<F = InnerVal, EF = InnerChallenge>> Witnessable<C>
    for SCChipOpenedValues<InnerVal, InnerChallenge>
{
    type WitnessVariable = SCChipOpenedValues<Felt<C::F>, Ext<C::F, C::EF>>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let preprocessed =
            dt_stark::SCAirOpenedValues { local: self.preprocessed.local.read(builder) };
        let main = dt_stark::SCAirOpenedValues { local: self.main.local.read(builder) };
        let permutation =
            dt_stark::SCAirOpenedValues { local: self.permutation.local.read(builder) };
        let local_cumulative_sum = self.local_cumulative_sum.read(builder);
        let log_height = self.log_height;
        Self::WitnessVariable {
            preprocessed,
            main,
            permutation,
            local_cumulative_sum,
            log_height,
            _field: core::marker::PhantomData,
        }
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        self.preprocessed.local.write(witness);
        self.main.local.write(witness);
        self.permutation.local.write(witness);
        self.local_cumulative_sum.write(witness);
    }
}

impl<
        C: CircuitConfig<F = Val<SC>, EF = Challenge<SC>>,
        SC: SCBabyBearFriConfigVariable<C> + dt_stark::StarkGenericConfig,
    > Witnessable<C> for SumcheckProof<SC>
where
    UniPolyEvals<Challenge<SC>>: Witnessable<C, WitnessVariable = UniPolyEvals<Ext<C::F, C::EF>>>,
{
    type WitnessVariable = SumcheckProofVariable<C>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let unipolys = self
            .unipolys
            .iter()
            .map(|p| UniPolyVariable::new_from_unipoly_evals(p.read(builder)))
            .collect();
        Self::WitnessVariable { unipolys }
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        self.unipolys.iter().for_each(|p| p.write(witness));
    }
}

impl<C: CircuitConfig<F = InnerVal, EF = InnerChallenge>> Witnessable<C>
    for UniPolyEvals<InnerChallenge>
{
    type WitnessVariable = UniPolyEvals<Ext<C::F, C::EF>>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let evals = self.evals.iter().map(|&p| p.read(builder)).collect();
        Self::WitnessVariable { evals }
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        self.evals.iter().for_each(|&p| witness.write_ext(p));
    }
}
impl<C: CircuitConfig<F = InnerVal, EF = InnerChallenge, Bit = Felt<InnerVal>>> Witnessable<C>
    for InnerBasefoldProof
{
    type WitnessVariable = BasefoldProofVariable<C, SCBabyBearPoseidon2>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let sumcheck_transcript = self.sumcheck_transcript.read(builder);
        let iopp_oracles = self.iopp_oracles.read(builder);
        let ood_values = self.ood_values.read(builder);
        let iopp_queries = self.iopp_queries.sc_read(builder);
        let round_iopp =
            self.round_iopp.as_ref().map(|round_iopp| read_whir_round_iopp(round_iopp, builder));
        // [SS] env=0: per_query carries full openings; env=1: per_query is
        // empty (size saving), circuit uses input_pruned instead.
        let query_openings = self.query_openings.per_query.sc_read(builder);
        let grinding_batching_witness = self.grinding_batching_witness.read(builder);
        let grinding_query_witness = self.grinding_query_witness.read(builder);
        let final_poly = self.final_poly.read(builder);

        // [SS] Read PCS input pruned data (round_opened_values + round_pruned + q2u).
        let input_pruned = self.query_openings.pruned.as_ref().map(|pruned| {
            let round_opened_values: Vec<Vec<Vec<Vec<Felt<C::F>>>>> = pruned
                .round_opened_values
                .iter()
                .map(|per_unique| {
                    per_unique
                        .iter()
                        .map(|per_mat| per_mat.iter().map(|row| row.read(builder)).collect())
                        .collect()
                })
                .collect();
            let round_pruned: Vec<PrunedBatchProofVariable<C, SCBabyBearPoseidon2>> = pruned
                .round_pruned
                .iter()
                .map(|rp| read_pruned_batch_proof_var(rp, rp.layer_widths.len(), builder))
                .collect();
            let query_to_unique_slot: Vec<Vec<usize>> = pruned
                .query_to_unique_slot
                .iter()
                .map(|row| row.iter().map(|&k| k as usize).collect())
                .collect();
            InputPrunedVariable { round_pruned, round_opened_values, query_to_unique_slot }
        });

        Self::WitnessVariable {
            stack_log_height: self.stack_log_height,
            sumcheck_transcript,
            iopp_oracles,
            ood_values,
            iopp_queries,
            round_iopp,
            query_openings,
            input_pruned,
            grinding_batching_witness,
            grinding_query_witness,
            final_poly,
            iopp_pruned: self.iopp_pruned.as_ref().map(|pf| PrunedFriQueryProofVariable {
                round_pruned_proofs: pf
                    .round_pruned_proofs
                    .iter()
                    .enumerate()
                    .map(|(_round_idx, rp)| {
                        read_pruned_batch_proof_var(rp, rp.layer_widths.len(), builder)
                    })
                    .collect(),
                round_opened_values: read_pruned_fri_opened_values(
                    &pf.round_opened_values,
                    builder,
                ),
                sibling_values: pf
                    .sibling_values
                    .iter()
                    .map(|row| row.iter().map(|v| v.read(builder)).collect())
                    .collect(),
                query_to_unique_slot: pf
                    .query_to_unique_slot
                    .iter()
                    .map(|row| row.iter().map(|&k| k as usize).collect())
                    .collect(),
            }),
            stacking_reduction: self.stacking_reduction.as_ref().map(|reduction| {
                StackingReductionProofVariable { sumcheck: reduction.sumcheck.read(builder) }
            }),
        }
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        self.sumcheck_transcript.write(witness);
        self.iopp_oracles.write(witness);
        self.ood_values.write(witness);
        self.iopp_queries.sc_write(witness);
        if let Some(round_iopp) = self.round_iopp.as_ref() {
            write_whir_round_iopp(round_iopp, witness);
        }
        // [SS] env=0: writes full per_query; env=1: writes empty vec (balanced).
        self.query_openings.per_query.sc_write(witness);
        self.grinding_batching_witness.write(witness);
        self.grinding_query_witness.write(witness);
        self.final_poly.write(witness);
        // [SS] Write PCS input pruned data.
        if let Some(pruned) = self.query_openings.pruned.as_ref() {
            for per_unique in pruned.round_opened_values.iter() {
                for per_mat in per_unique.iter() {
                    for row in per_mat.iter() {
                        row.write(witness);
                    }
                }
            }
            for rp in pruned.round_pruned.iter() {
                rp.siblings.write(witness);
            }
        }
        // IOPP pruned data (unchanged).
        if let Some(pf) = self.iopp_pruned.as_ref() {
            for rp in pf.round_pruned_proofs.iter() {
                rp.siblings.write(witness);
            }
            write_pruned_fri_opened_values(&pf.round_opened_values, witness);
            for row in pf.sibling_values.iter() {
                row.write(witness);
            }
        }
    }
}
#[cfg(feature = "babybear")]
impl Witnessable<OuterConfig> for OuterBasefoldProof {
    type WitnessVariable = BasefoldProofVariable<OuterConfig, SCBabyBearPoseidon2Outer>;

    fn read(&self, builder: &mut Builder<OuterConfig>) -> Self::WitnessVariable {
        let sumcheck_transcript = self.sumcheck_transcript.read(builder);
        let iopp_oracles = self.iopp_oracles.read(builder);
        let ood_values = self.ood_values.read(builder);
        let iopp_queries = self.iopp_queries.sc_read(builder);
        let round_iopp =
            self.round_iopp.as_ref().map(|round_iopp| read_whir_round_iopp(round_iopp, builder));
        let query_openings = self.query_openings.per_query.sc_read(builder);
        let grinding_batching_witness = self.grinding_batching_witness.read(builder);
        let grinding_query_witness = self.grinding_query_witness.read(builder);
        let final_poly = self.final_poly.read(builder);

        // [SS] Read PCS input pruned data (Outer/Bn254 path).
        let input_pruned = self.query_openings.pruned.as_ref().map(|pruned| {
            use p3_field::AbstractField;
            let round_opened_values: Vec<
                Vec<Vec<Vec<Felt<<OuterConfig as dt_recursion_compiler::ir::Config>::F>>>>,
            > = pruned
                .round_opened_values
                .iter()
                .map(|per_unique| {
                    per_unique
                        .iter()
                        .map(|per_mat| per_mat.iter().map(|row| row.read(builder)).collect())
                        .collect()
                })
                .collect();
            let round_pruned = pruned
                .round_pruned
                .iter()
                .map(|rp| read_pruned_batch_proof_var(rp, rp.layer_widths.len(), builder))
                .collect();
            let query_to_unique_slot: Vec<Vec<usize>> = pruned
                .query_to_unique_slot
                .iter()
                .map(|row| row.iter().map(|&k| k as usize).collect())
                .collect();
            InputPrunedVariable { round_pruned, round_opened_values, query_to_unique_slot }
        });

        Self::WitnessVariable {
            stack_log_height: self.stack_log_height,
            sumcheck_transcript,
            iopp_oracles,
            ood_values,
            iopp_queries,
            round_iopp,
            query_openings,
            input_pruned,
            grinding_batching_witness,
            grinding_query_witness,
            final_poly,
            iopp_pruned: self.iopp_pruned.as_ref().map(|pf| PrunedFriQueryProofVariable {
                round_pruned_proofs: pf
                    .round_pruned_proofs
                    .iter()
                    .enumerate()
                    .map(|(_round_idx, rp)| {
                        read_pruned_batch_proof_var(rp, rp.layer_widths.len(), builder)
                    })
                    .collect(),
                round_opened_values: read_pruned_fri_opened_values(
                    &pf.round_opened_values,
                    builder,
                ),
                sibling_values: pf
                    .sibling_values
                    .iter()
                    .map(|row| row.iter().map(|v| v.read(builder)).collect())
                    .collect(),
                query_to_unique_slot: pf
                    .query_to_unique_slot
                    .iter()
                    .map(|row| row.iter().map(|&k| k as usize).collect())
                    .collect(),
            }),
            stacking_reduction: self.stacking_reduction.as_ref().map(|reduction| {
                StackingReductionProofVariable { sumcheck: reduction.sumcheck.read(builder) }
            }),
        }
    }

    fn write(&self, witness: &mut impl WitnessWriter<OuterConfig>) {
        self.sumcheck_transcript.write(witness);
        self.iopp_oracles.write(witness);
        self.ood_values.write(witness);
        self.iopp_queries.sc_write(witness);
        if let Some(round_iopp) = self.round_iopp.as_ref() {
            write_whir_round_iopp(round_iopp, witness);
        }
        self.query_openings.per_query.sc_write(witness);
        self.grinding_batching_witness.write(witness);
        self.grinding_query_witness.write(witness);
        self.final_poly.write(witness);
        if let Some(pruned) = self.query_openings.pruned.as_ref() {
            for per_unique in pruned.round_opened_values.iter() {
                for per_mat in per_unique.iter() {
                    for row in per_mat.iter() {
                        row.write(witness);
                    }
                }
            }
            for rp in pruned.round_pruned.iter() {
                rp.siblings.write(witness);
            }
        }
        if let Some(pf) = self.iopp_pruned.as_ref() {
            for rp in pf.round_pruned_proofs.iter() {
                rp.siblings.write(witness);
            }
            write_pruned_fri_opened_values(&pf.round_opened_values, witness);
            for row in pf.sibling_values.iter() {
                row.write(witness);
            }
        }
        if let Some(reduction) = self.stacking_reduction.as_ref() {
            reduction.sumcheck.write(witness);
        }
    }
}

impl<C: CircuitConfig<F = InnerVal, EF = InnerChallenge>> Witnessable<C>
    for SumcheckInstanceProof<InnerChallenge>
{
    type WitnessVariable = SumcheckInstanceProofVariable<C>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let uni_polys = self.uni_polys.read(builder);
        Self::WitnessVariable { uni_polys }
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        self.uni_polys.write(witness);
    }
}

impl<C: CircuitConfig<F = InnerVal, EF = InnerChallenge>> Witnessable<C>
    for UniPoly<InnerChallenge>
{
    type WitnessVariable = UniPolyVariable<C>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let evals = self.coeffs.read(builder);
        Self::WitnessVariable { evals }
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        self.coeffs.write(witness);
    }
}
//TODO
impl<C> SCWitnessable<C> for InnerBatchOpening
where
    C: CircuitConfig<F = InnerVal, EF = InnerChallenge, Bit = Felt<InnerVal>>,
{
    type SCWitnessVariable = BatchOpeningVariable<C, SCBabyBearPoseidon2>;

    fn sc_read(&self, builder: &mut Builder<C>) -> Self::SCWitnessVariable {
        let opened_values = self
            .opened_values
            .read(builder)
            .into_iter()
            .map(|a| a.into_iter().map(|b| vec![b]).collect())
            .collect();
        let opening_proof = self.opening_proof.read(builder);
        Self::SCWitnessVariable { opened_values, opening_proof }
    }

    fn sc_write(&self, witness: &mut impl WitnessWriter<C>) {
        self.opened_values.write(witness);
        self.opening_proof.write(witness);
    }
}
#[cfg(feature = "babybear")]
impl SCWitnessable<OuterConfig> for OuterBatchOpening {
    type SCWitnessVariable = BatchOpeningVariable<OuterConfig, SCBabyBearPoseidon2Outer>;

    fn sc_read(&self, builder: &mut Builder<OuterConfig>) -> Self::SCWitnessVariable {
        let opened_values = self
            .opened_values
            .read(builder)
            .into_iter()
            .map(|a| a.into_iter().map(|b| vec![b]).collect())
            .collect();
        let opening_proof = self.opening_proof.read(builder);
        Self::SCWitnessVariable { opened_values, opening_proof }
    }

    fn sc_write(&self, witness: &mut impl WitnessWriter<OuterConfig>) {
        self.opened_values.write(witness);
        self.opening_proof.write(witness);
    }
}

impl<C: CircuitConfig, T: SCWitnessable<C>> SCWitnessable<C> for Vec<T> {
    type SCWitnessVariable = Vec<T::SCWitnessVariable>;

    fn sc_read(&self, builder: &mut Builder<C>) -> Self::SCWitnessVariable {
        self.iter().map(|x| x.sc_read(builder)).collect()
    }

    fn sc_write(&self, witness: &mut impl WitnessWriter<C>) {
        for x in self.iter() {
            x.sc_write(witness);
        }
    }
}
//TODO
impl<C: CircuitConfig<F = InnerVal, EF = InnerChallenge, Bit = Felt<InnerVal>>> SCWitnessable<C>
    for QueryProof<InnerChallenge, InnerChallengeMmcs>
{
    type SCWitnessVariable = FriQueryProofVariable<C, SCBabyBearPoseidon2>;

    fn sc_read(&self, builder: &mut Builder<C>) -> Self::SCWitnessVariable {
        let commit_phase_openings = self.commit_phase_openings.sc_read(builder);
        Self::SCWitnessVariable { commit_phase_openings }
    }

    fn sc_write(&self, witness: &mut impl WitnessWriter<C>) {
        self.commit_phase_openings.sc_write(witness);
    }
}
#[cfg(feature = "babybear")]
impl SCWitnessable<OuterConfig> for QueryProof<OuterChallenge, OuterChallengeMmcs> {
    type SCWitnessVariable = FriQueryProofVariable<OuterConfig, SCBabyBearPoseidon2Outer>;

    fn sc_read(&self, builder: &mut Builder<OuterConfig>) -> Self::SCWitnessVariable {
        let commit_phase_openings = self.commit_phase_openings.sc_read(builder);
        Self::SCWitnessVariable { commit_phase_openings }
    }

    fn sc_write(&self, witness: &mut impl WitnessWriter<OuterConfig>) {
        self.commit_phase_openings.sc_write(witness);
    }
}
//TODO
impl<C: CircuitConfig<F = InnerVal, EF = InnerChallenge, Bit = Felt<InnerVal>>> SCWitnessable<C>
    for CommitPhaseProofStep<InnerChallenge, InnerChallengeMmcs>
{
    type SCWitnessVariable = FriCommitPhaseProofStepVariable<C, SCBabyBearPoseidon2>;

    fn sc_read(&self, builder: &mut Builder<C>) -> Self::SCWitnessVariable {
        let sibling_value = self.sibling_value.read(builder);
        let leaf_values = self.opened_values.read(builder);
        let opening_proof = self.opening_proof.read(builder);
        Self::SCWitnessVariable { sibling_value, leaf_values, opening_proof }
    }

    fn sc_write(&self, witness: &mut impl WitnessWriter<C>) {
        self.sibling_value.write(witness);
        self.opened_values.write(witness);
        self.opening_proof.write(witness);
    }
}
#[cfg(feature = "babybear")]
impl SCWitnessable<OuterConfig> for CommitPhaseProofStep<OuterChallenge, OuterChallengeMmcs> {
    type SCWitnessVariable = FriCommitPhaseProofStepVariable<OuterConfig, SCBabyBearPoseidon2Outer>;

    fn sc_read(&self, builder: &mut Builder<OuterConfig>) -> Self::SCWitnessVariable {
        let sibling_value = self.sibling_value.read(builder);
        let leaf_values = self.opened_values.read(builder);
        let opening_proof = self.opening_proof.read(builder);
        Self::SCWitnessVariable { sibling_value, leaf_values, opening_proof }
    }

    fn sc_write(&self, witness: &mut impl WitnessWriter<OuterConfig>) {
        self.sibling_value.write(witness);
        self.opened_values.write(witness);
        self.opening_proof.write(witness);
    }
}

// ======================== KoalaBear Witnessable implementations ========================

#[cfg(feature = "koalabear")]
impl<C: CircuitConfig<F = KBInnerVal, EF = KBInnerChallenge>> Witnessable<C>
    for SCShardOpenedValues<KBInnerVal, KBInnerChallenge>
{
    type WitnessVariable = SCShardOpenedValues<Felt<C::F>, Ext<C::F, C::EF>>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let chips = self.chips.read(builder);
        Self::WitnessVariable { chips, _field: core::marker::PhantomData }
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        self.chips.write(witness);
    }
}

#[cfg(feature = "koalabear")]
impl<C: CircuitConfig<F = KBInnerVal, EF = KBInnerChallenge>> Witnessable<C>
    for SCChipOpenedValues<KBInnerVal, KBInnerChallenge>
{
    type WitnessVariable = SCChipOpenedValues<Felt<C::F>, Ext<C::F, C::EF>>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let preprocessed =
            dt_stark::SCAirOpenedValues { local: self.preprocessed.local.read(builder) };
        let main = dt_stark::SCAirOpenedValues { local: self.main.local.read(builder) };
        let permutation =
            dt_stark::SCAirOpenedValues { local: self.permutation.local.read(builder) };
        let local_cumulative_sum = self.local_cumulative_sum.read(builder);
        let log_height = self.log_height;
        Self::WitnessVariable {
            preprocessed,
            main,
            permutation,
            local_cumulative_sum,
            log_height,
            _field: core::marker::PhantomData,
        }
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        self.preprocessed.local.write(witness);
        self.main.local.write(witness);
        self.permutation.local.write(witness);
        self.local_cumulative_sum.write(witness);
    }
}

#[cfg(feature = "koalabear")]
impl<C: CircuitConfig<F = KBInnerVal, EF = KBInnerChallenge>> Witnessable<C>
    for UniPolyEvals<KBInnerChallenge>
{
    type WitnessVariable = UniPolyEvals<Ext<C::F, C::EF>>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let evals = self.evals.iter().map(|&p| p.read(builder)).collect();
        Self::WitnessVariable { evals }
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        self.evals.iter().for_each(|&p| witness.write_ext(p));
    }
}

#[cfg(feature = "koalabear")]
impl<C: CircuitConfig<F = KBInnerVal, EF = KBInnerChallenge, Bit = Felt<KBInnerVal>>> Witnessable<C>
    for KBInnerBasefoldProof
{
    type WitnessVariable = BasefoldProofVariable<C, SCKoalaBearPoseidon2>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let sumcheck_transcript = self.sumcheck_transcript.read(builder);
        let iopp_oracles = self.iopp_oracles.read(builder);
        let ood_values = self.ood_values.read(builder);
        let iopp_queries = self.iopp_queries.sc_read(builder);
        let round_iopp =
            self.round_iopp.as_ref().map(|round_iopp| read_whir_round_iopp(round_iopp, builder));
        let query_openings = self.query_openings.per_query.sc_read(builder);
        let grinding_batching_witness = self.grinding_batching_witness.read(builder);
        let grinding_query_witness = self.grinding_query_witness.read(builder);
        let final_poly = self.final_poly.read(builder);

        // [SS] Read PCS input pruned data (KoalaBear inner).
        let input_pruned = self.query_openings.pruned.as_ref().map(|pruned| {
            let round_opened_values = pruned
                .round_opened_values
                .iter()
                .map(|per_unique| {
                    per_unique
                        .iter()
                        .map(|per_mat| per_mat.iter().map(|row| row.read(builder)).collect())
                        .collect()
                })
                .collect();
            let round_pruned = pruned
                .round_pruned
                .iter()
                .map(|rp| read_pruned_batch_proof_var(rp, rp.layer_widths.len(), builder))
                .collect();
            let query_to_unique_slot = pruned
                .query_to_unique_slot
                .iter()
                .map(|row| row.iter().map(|&k| k as usize).collect())
                .collect();
            InputPrunedVariable { round_pruned, round_opened_values, query_to_unique_slot }
        });

        Self::WitnessVariable {
            stack_log_height: self.stack_log_height,
            sumcheck_transcript,
            iopp_oracles,
            ood_values,
            iopp_queries,
            round_iopp,
            query_openings,
            input_pruned,
            grinding_batching_witness,
            grinding_query_witness,
            final_poly,
            iopp_pruned: self.iopp_pruned.as_ref().map(|pf| PrunedFriQueryProofVariable {
                round_pruned_proofs: pf
                    .round_pruned_proofs
                    .iter()
                    .enumerate()
                    .map(|(_round_idx, rp)| {
                        read_pruned_batch_proof_var(rp, rp.layer_widths.len(), builder)
                    })
                    .collect(),
                round_opened_values: read_pruned_fri_opened_values(
                    &pf.round_opened_values,
                    builder,
                ),
                sibling_values: pf
                    .sibling_values
                    .iter()
                    .map(|row| row.iter().map(|v| v.read(builder)).collect())
                    .collect(),
                query_to_unique_slot: pf
                    .query_to_unique_slot
                    .iter()
                    .map(|row| row.iter().map(|&k| k as usize).collect())
                    .collect(),
            }),
            stacking_reduction: self.stacking_reduction.as_ref().map(|reduction| {
                StackingReductionProofVariable { sumcheck: reduction.sumcheck.read(builder) }
            }),
        }
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        self.sumcheck_transcript.write(witness);
        self.iopp_oracles.write(witness);
        self.ood_values.write(witness);
        self.iopp_queries.sc_write(witness);
        if let Some(round_iopp) = self.round_iopp.as_ref() {
            write_whir_round_iopp(round_iopp, witness);
        }
        self.query_openings.per_query.sc_write(witness);
        self.grinding_batching_witness.write(witness);
        self.grinding_query_witness.write(witness);
        self.final_poly.write(witness);
        if let Some(pruned) = self.query_openings.pruned.as_ref() {
            for per_unique in pruned.round_opened_values.iter() {
                for per_mat in per_unique.iter() {
                    for row in per_mat.iter() {
                        row.write(witness);
                    }
                }
            }
            for rp in pruned.round_pruned.iter() {
                rp.siblings.write(witness);
            }
        }
        if let Some(pf) = self.iopp_pruned.as_ref() {
            for rp in pf.round_pruned_proofs.iter() {
                rp.siblings.write(witness);
            }
            write_pruned_fri_opened_values(&pf.round_opened_values, witness);
            for row in pf.sibling_values.iter() {
                row.write(witness);
            }
        }
        if let Some(reduction) = self.stacking_reduction.as_ref() {
            reduction.sumcheck.write(witness);
        }
    }
}

#[cfg(feature = "koalabear")]
impl<C: CircuitConfig<F = KBInnerVal, EF = KBInnerChallenge>> Witnessable<C>
    for SumcheckInstanceProof<KBInnerChallenge>
{
    type WitnessVariable = SumcheckInstanceProofVariable<C>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let uni_polys = self.uni_polys.read(builder);
        Self::WitnessVariable { uni_polys }
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        self.uni_polys.write(witness);
    }
}

#[cfg(feature = "koalabear")]
impl<C: CircuitConfig<F = KBInnerVal, EF = KBInnerChallenge>> Witnessable<C>
    for UniPoly<KBInnerChallenge>
{
    type WitnessVariable = UniPolyVariable<C>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let evals = self.coeffs.read(builder);
        Self::WitnessVariable { evals }
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        self.coeffs.write(witness);
    }
}

#[cfg(feature = "koalabear")]
impl<C: CircuitConfig<F = KBInnerVal, EF = KBInnerChallenge, Bit = Felt<KBInnerVal>>>
    SCWitnessable<C> for KBInnerBatchOpening
{
    type SCWitnessVariable = BatchOpeningVariable<C, SCKoalaBearPoseidon2>;

    fn sc_read(&self, builder: &mut Builder<C>) -> Self::SCWitnessVariable {
        let opened_values = self
            .opened_values
            .read(builder)
            .into_iter()
            .map(|a| a.into_iter().map(|b| vec![b]).collect())
            .collect();
        let opening_proof = self.opening_proof.read(builder);
        Self::SCWitnessVariable { opened_values, opening_proof }
    }

    fn sc_write(&self, witness: &mut impl WitnessWriter<C>) {
        self.opened_values.write(witness);
        self.opening_proof.write(witness);
    }
}

#[cfg(feature = "koalabear")]
impl<C: CircuitConfig<F = KBInnerVal, EF = KBInnerChallenge, Bit = Felt<KBInnerVal>>>
    SCWitnessable<C> for QueryProof<KBInnerChallenge, KBInnerChallengeMmcs>
{
    type SCWitnessVariable = FriQueryProofVariable<C, SCKoalaBearPoseidon2>;

    fn sc_read(&self, builder: &mut Builder<C>) -> Self::SCWitnessVariable {
        let commit_phase_openings = self.commit_phase_openings.sc_read(builder);
        Self::SCWitnessVariable { commit_phase_openings }
    }

    fn sc_write(&self, witness: &mut impl WitnessWriter<C>) {
        self.commit_phase_openings.sc_write(witness);
    }
}

#[cfg(feature = "koalabear")]
impl<C: CircuitConfig<F = KBInnerVal, EF = KBInnerChallenge, Bit = Felt<KBInnerVal>>>
    SCWitnessable<C> for CommitPhaseProofStep<KBInnerChallenge, KBInnerChallengeMmcs>
{
    type SCWitnessVariable = FriCommitPhaseProofStepVariable<C, SCKoalaBearPoseidon2>;

    fn sc_read(&self, builder: &mut Builder<C>) -> Self::SCWitnessVariable {
        let sibling_value = self.sibling_value.read(builder);
        let leaf_values = self.opened_values.read(builder);
        let opening_proof = self.opening_proof.read(builder);
        Self::SCWitnessVariable { sibling_value, leaf_values, opening_proof }
    }

    fn sc_write(&self, witness: &mut impl WitnessWriter<C>) {
        self.sibling_value.write(witness);
        self.opened_values.write(witness);
        self.opening_proof.write(witness);
    }
}

// ======================== KoalaBear Outer Witnessable implementations ========================

#[cfg(feature = "koalabear")]
use dt_recursion_compiler::config::SCOuterConfig;
#[cfg(feature = "koalabear")]
use dt_recursion_core::stark::{
    SCKoalaBearPoseidon2Outer, SCOuterBasefoldProof, SCOuterBatchOpening, SCOuterChallenge,
    SCOuterChallengeMmcs,
};

// Under `ext5` the SC inner challenge (`KBInnerChallenge`) is the quintic
// extension, so the generic SC sumcheck Witnessable impls above no longer cover
// the binomial-quartic `SCOuterChallenge` used by the KoalaBear wrap layer.
// Provide quartic-specific impls here, gated to `ext5` to avoid duplicating the
// generic impls under the default ext4 build (where the two types coincide).
#[cfg(all(feature = "koalabear", feature = "ext5"))]
impl Witnessable<SCOuterConfig> for SumcheckInstanceProof<SCOuterChallenge> {
    type WitnessVariable = SumcheckInstanceProofVariable<SCOuterConfig>;

    fn read(&self, builder: &mut Builder<SCOuterConfig>) -> Self::WitnessVariable {
        let uni_polys = self.uni_polys.read(builder);
        Self::WitnessVariable { uni_polys }
    }

    fn write(&self, witness: &mut impl WitnessWriter<SCOuterConfig>) {
        self.uni_polys.write(witness);
    }
}

#[cfg(all(feature = "koalabear", feature = "ext5"))]
impl Witnessable<SCOuterConfig> for UniPoly<SCOuterChallenge> {
    type WitnessVariable = UniPolyVariable<SCOuterConfig>;

    fn read(&self, builder: &mut Builder<SCOuterConfig>) -> Self::WitnessVariable {
        let evals = self.coeffs.read(builder);
        Self::WitnessVariable { evals }
    }

    fn write(&self, witness: &mut impl WitnessWriter<SCOuterConfig>) {
        self.coeffs.write(witness);
    }
}

#[cfg(feature = "koalabear")]
impl Witnessable<SCOuterConfig> for SCOuterBasefoldProof {
    type WitnessVariable = BasefoldProofVariable<SCOuterConfig, SCKoalaBearPoseidon2Outer>;

    fn read(&self, builder: &mut Builder<SCOuterConfig>) -> Self::WitnessVariable {
        let sumcheck_transcript = self.sumcheck_transcript.read(builder);
        let iopp_oracles = self.iopp_oracles.read(builder);
        let ood_values = self.ood_values.read(builder);
        let iopp_queries = self.iopp_queries.sc_read(builder);
        let round_iopp =
            self.round_iopp.as_ref().map(|round_iopp| read_whir_round_iopp(round_iopp, builder));
        let query_openings = self.query_openings.per_query.sc_read(builder);
        let grinding_batching_witness = self.grinding_batching_witness.read(builder);
        let grinding_query_witness = self.grinding_query_witness.read(builder);
        let final_poly = self.final_poly.read(builder);

        // [SS] Read PCS input pruned data (KoalaBear outer).
        let input_pruned = self.query_openings.pruned.as_ref().map(|pruned| {
            let round_opened_values = pruned
                .round_opened_values
                .iter()
                .map(|per_unique| {
                    per_unique
                        .iter()
                        .map(|per_mat| per_mat.iter().map(|row| row.read(builder)).collect())
                        .collect()
                })
                .collect();
            let round_pruned = pruned
                .round_pruned
                .iter()
                .map(|rp| read_pruned_batch_proof_var(rp, rp.layer_widths.len(), builder))
                .collect();
            let query_to_unique_slot = pruned
                .query_to_unique_slot
                .iter()
                .map(|row| row.iter().map(|&k| k as usize).collect())
                .collect();
            InputPrunedVariable { round_pruned, round_opened_values, query_to_unique_slot }
        });

        Self::WitnessVariable {
            stack_log_height: self.stack_log_height,
            sumcheck_transcript,
            iopp_oracles,
            ood_values,
            iopp_queries,
            round_iopp,
            query_openings,
            input_pruned,
            grinding_batching_witness,
            grinding_query_witness,
            final_poly,
            iopp_pruned: self.iopp_pruned.as_ref().map(|pf| PrunedFriQueryProofVariable {
                round_pruned_proofs: pf
                    .round_pruned_proofs
                    .iter()
                    .enumerate()
                    .map(|(_round_idx, rp)| {
                        read_pruned_batch_proof_var(rp, rp.layer_widths.len(), builder)
                    })
                    .collect(),
                round_opened_values: read_pruned_fri_opened_values(
                    &pf.round_opened_values,
                    builder,
                ),
                sibling_values: pf
                    .sibling_values
                    .iter()
                    .map(|row| row.iter().map(|v| v.read(builder)).collect())
                    .collect(),
                query_to_unique_slot: pf
                    .query_to_unique_slot
                    .iter()
                    .map(|row| row.iter().map(|&k| k as usize).collect())
                    .collect(),
            }),
            stacking_reduction: self.stacking_reduction.as_ref().map(|reduction| {
                StackingReductionProofVariable { sumcheck: reduction.sumcheck.read(builder) }
            }),
        }
    }

    fn write(&self, witness: &mut impl WitnessWriter<SCOuterConfig>) {
        self.sumcheck_transcript.write(witness);
        self.iopp_oracles.write(witness);
        self.ood_values.write(witness);
        self.iopp_queries.sc_write(witness);
        if let Some(round_iopp) = self.round_iopp.as_ref() {
            write_whir_round_iopp(round_iopp, witness);
        }
        self.query_openings.per_query.sc_write(witness);
        self.grinding_batching_witness.write(witness);
        self.grinding_query_witness.write(witness);
        self.final_poly.write(witness);
        if let Some(pruned) = self.query_openings.pruned.as_ref() {
            for per_unique in pruned.round_opened_values.iter() {
                for per_mat in per_unique.iter() {
                    for row in per_mat.iter() {
                        row.write(witness);
                    }
                }
            }
            for rp in pruned.round_pruned.iter() {
                rp.siblings.write(witness);
            }
        }
        if let Some(pf) = self.iopp_pruned.as_ref() {
            for rp in pf.round_pruned_proofs.iter() {
                rp.siblings.write(witness);
            }
            write_pruned_fri_opened_values(&pf.round_opened_values, witness);
            for row in pf.sibling_values.iter() {
                row.write(witness);
            }
        }
        if let Some(reduction) = self.stacking_reduction.as_ref() {
            reduction.sumcheck.write(witness);
        }
    }
}

#[cfg(feature = "koalabear")]
impl SCWitnessable<SCOuterConfig> for SCOuterBatchOpening {
    type SCWitnessVariable = BatchOpeningVariable<SCOuterConfig, SCKoalaBearPoseidon2Outer>;

    fn sc_read(&self, builder: &mut Builder<SCOuterConfig>) -> Self::SCWitnessVariable {
        let opened_values = self
            .opened_values
            .read(builder)
            .into_iter()
            .map(|a| a.into_iter().map(|b| vec![b]).collect())
            .collect();
        let opening_proof = self.opening_proof.read(builder);
        Self::SCWitnessVariable { opened_values, opening_proof }
    }

    fn sc_write(&self, witness: &mut impl WitnessWriter<SCOuterConfig>) {
        self.opened_values.write(witness);
        self.opening_proof.write(witness);
    }
}

#[cfg(feature = "koalabear")]
impl SCWitnessable<SCOuterConfig> for QueryProof<SCOuterChallenge, SCOuterChallengeMmcs> {
    type SCWitnessVariable = FriQueryProofVariable<SCOuterConfig, SCKoalaBearPoseidon2Outer>;

    fn sc_read(&self, builder: &mut Builder<SCOuterConfig>) -> Self::SCWitnessVariable {
        let commit_phase_openings = self.commit_phase_openings.sc_read(builder);
        Self::SCWitnessVariable { commit_phase_openings }
    }

    fn sc_write(&self, witness: &mut impl WitnessWriter<SCOuterConfig>) {
        self.commit_phase_openings.sc_write(witness);
    }
}

#[cfg(feature = "koalabear")]
impl SCWitnessable<SCOuterConfig> for CommitPhaseProofStep<SCOuterChallenge, SCOuterChallengeMmcs> {
    type SCWitnessVariable =
        FriCommitPhaseProofStepVariable<SCOuterConfig, SCKoalaBearPoseidon2Outer>;

    fn sc_read(&self, builder: &mut Builder<SCOuterConfig>) -> Self::SCWitnessVariable {
        let sibling_value = self.sibling_value.read(builder);
        let leaf_values = self.opened_values.read(builder);
        let opening_proof = self.opening_proof.read(builder);
        Self::SCWitnessVariable { sibling_value, leaf_values, opening_proof }
    }

    fn sc_write(&self, witness: &mut impl WitnessWriter<SCOuterConfig>) {
        self.sibling_value.write(witness);
        self.opened_values.write(witness);
        self.opening_proof.write(witness);
    }
}

impl<C: CircuitConfig, T: Witnessable<C>> Witnessable<C> for BTreeMap<i32, T> {
    type WitnessVariable = BTreeMap<i32, T::WitnessVariable>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let mut new_map = BTreeMap::new();
        for (k, v) in self.iter() {
            new_map.insert(*k, v.read(builder));
        }
        new_map
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        self.iter().for_each(|(_k, v)| {
            v.write(witness);
        });
    }
}
