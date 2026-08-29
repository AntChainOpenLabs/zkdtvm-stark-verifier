use crate::{
    air::PublicValues,
    config::Challenge,
    global_d11::{
        observe_owner_registry_v2, observe_program_boundary_v1, validate_global146_identity,
        verify_global_interval_root_v4, BoundaryOwnerRegistryV2, BoundaryOwnerV2,
        GlobalBoundaryKindV2, ProgramImageBoundaryV1, GLOBAL146_COMPOSITE_IDENTITY,
    },
    sumcheck::{
        config::SCStarkGenericConfig,
        keys::{SCStarkProvingKey, SCStarkVerifyingKey},
        trace::CompressedMatrix,
    },
    word::Word,
    PROOF_MAX_NUM_PVS,
};
use hashbrown::HashMap;
use itertools::Itertools;
use p3_air::Air;
use p3_baby_bear::BabyBear;
use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::Pcs;
use p3_field::{AbstractExtensionField, AbstractField, PrimeField32, TwoAdicField};
use p3_matrix::{dense::RowMajorMatrix, Dimensions, Matrix};
use p3_maybe_rayon::prelude::*;
use p3_symmetric::CryptographicHasher;
use p3_uni_stark::{get_symbolic_constraints, SymbolicAirBuilder};
use p3_util::log2_strict_usize;
use pcs::basefold::mlpcs::MlPCS;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{borrow::Borrow, cmp::Reverse, env, fmt::Debug};

#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

#[cfg(target_arch = "wasm32")]
#[derive(Clone, Copy)]
struct Instant;

#[cfg(target_arch = "wasm32")]
impl Instant {
    fn now() -> Self {
        Self
    }

    fn elapsed(self) -> std::time::Duration {
        std::time::Duration::ZERO
    }
}
use tracing::instrument;

use super::debug_constraints;
use crate::{
    air::{InteractionScope, MachineAir, MachineProgram},
    count_permutation_constraints,
    lookup::{
        debug_interactions_with_all_chips, debug_interactions_with_all_chips_sumcheck,
        InteractionKind,
    },
    record::MachineRecord,
    DebugConstraintBuilder, ShardProof, VerifierConstraintFolder,
};

use super::{Chip, MachineProof, VerificationError, Verifier};
use crate::config::{Com, Dom, PcsProverData, StarkGenericConfig, Val};

use crate::sumcheck::{
    proof::SCMachineProof,
    verifier::{SumcheckVerificationError, Verifier as SumcheckVerifier},
};

/// A chip in a machine.
pub type MachineChip<SC, A> = Chip<Val<SC>, A>;

/// A chip for sumcheck rounds
pub type SumcheckChip<SC, A> = Chip<Challenge<SC>, A>;

/// Compute the expected local cumulative sum contribution from dangling interactions.
///
/// This accounts for two sources of imbalance:
///
/// 1. **CPU State**: each instruction chip chains `receive_state → send_state`. Only the first
///    receive and last send remain unbalanced.
///
/// 2. **`MemoryGlobalAddr`**: each `MemoryGlobal` row sends its `addr` and receives `prev_addr`.
///    The first row's receive (`prev_addr` from PV) and last row's send (`last_addr` from PV) are
///    unmatched, creating a net imbalance that the verifier must account for.
///
/// The projective Global-chain contribution is authenticated separately by the public claim.
pub fn compute_expected_state_imbalance<SC: StarkGenericConfig>(
    public_values: &[Val<SC>],
    permutation_challenges: &[Challenge<SC>],
) -> Challenge<SC> {
    let pv: &PublicValues<Word<Val<SC>>, Val<SC>> = public_values.borrow();
    let alpha = permutation_challenges[0];
    let beta = permutation_challenges[1];
    let one = Challenge::<SC>::one();
    let mut result = Challenge::<SC>::zero();

    // --- CPU State imbalance ---
    let start_clk = pv.start_clk;
    let exit_clk = pv.exit_clk;
    if start_clk != exit_clk {
        let beta2 = beta * beta;
        let beta3 = beta2 * beta;

        let state_kind = Challenge::<SC>::from_canonical_usize(InteractionKind::State as usize);
        let shard_term = beta * Challenge::<SC>::from_base(pv.execution_shard);

        let recv_fp = alpha +
            state_kind +
            shard_term +
            beta2 * Challenge::<SC>::from_base(start_clk) +
            beta3 * Challenge::<SC>::from_base(pv.start_pc);

        let send_fp = alpha +
            state_kind +
            shard_term +
            beta2 * Challenge::<SC>::from_base(exit_clk) +
            beta3 * Challenge::<SC>::from_base(pv.next_pc);

        result += one / send_fp - one / recv_fp;
    }

    // --- MemoryGlobalAddr imbalance ---
    let beta2 = beta * beta;
    let addr_kind =
        Challenge::<SC>::from_canonical_usize(InteractionKind::MemoryGlobalAddr as usize);

    let prev_init = pv.previous_init_addr;
    let last_init = pv.last_init_addr;
    if prev_init != last_init {
        let base = alpha + addr_kind;
        let recv_fp = base + beta2 * Challenge::<SC>::from_base(prev_init);
        let send_fp = base + beta2 * Challenge::<SC>::from_base(last_init);
        result += one / send_fp - one / recv_fp;
    }

    let prev_fin = pv.previous_finalize_addr;
    let last_fin = pv.last_finalize_addr;
    if prev_fin != last_fin {
        let base = alpha + addr_kind + beta;
        let recv_fp = base + beta2 * Challenge::<SC>::from_base(prev_fin);
        let send_fp = base + beta2 * Challenge::<SC>::from_base(last_fin);
        result += one / send_fp - one / recv_fp;
    }

    result
}

/// A STARK for proving RISC-V execution.
pub struct StarkMachine<SC: StarkGenericConfig, A> {
    /// The STARK settings for the RISC-V STARK.
    config: SC,
    /// The chips that make up the RISC-V STARK machine, in order of their execution.
    chips: Vec<Chip<Val<SC>, A>>,

    /// The number of public values elements that the machine uses
    num_pv_elts: usize,

    /// Contains a global bus.  This should be true for the core machine and false otherwise.
    contains_global_bus: bool,

    /// VK-bound Global owner inventory.
    global_boundary_registry: BoundaryOwnerRegistryV2,
}

impl<SC: StarkGenericConfig, A> StarkMachine<SC, A> {
    /// Creates a new [`StarkMachine`].
    pub fn new(
        config: SC,
        chips: Vec<Chip<Val<SC>, A>>,
        num_pv_elts: usize,
        contains_global_bus: bool,
    ) -> Self
    where
        A: MachineAir<Val<SC>>,
    {
        let owners = chips
            .iter()
            .filter_map(MachineAir::global_boundary_owner)
            .map(|owner| BoundaryOwnerV2 { owner, kind: GlobalBoundaryKindV2::Projective })
            .collect();
        let global_boundary_registry = BoundaryOwnerRegistryV2::new(owners)
            .expect("machine Global boundary owner registry must be canonical");
        Self { config, chips, num_pv_elts, contains_global_bus, global_boundary_registry }
    }

    /// Return the immutable owner registry bound into the key authority.
    pub const fn global_boundary_registry(&self) -> &BoundaryOwnerRegistryV2 {
        &self.global_boundary_registry
    }
}

/// A STARK for proving RISC-V execution.
pub struct SCStarkMachine<SC: SCStarkGenericConfig, A, AE> {
    /// The STARK settings for the RISC-V STARK.
    config: SC,
    /// The chips that make up the RISC-V STARK machine, in order of their execution.
    chips: Vec<Chip<Val<SC>, A>>,
    /// The chips for extension fields
    chips_ext: Vec<Chip<Challenge<SC>, AE>>,
    /// The number of public values elements that the machine uses
    num_pv_elts: usize,
    /// Contains a global bus.  This should be true for the core machine and false otherwise.
    contains_global_bus: bool,
    /// VK-bound Global owner inventory.
    global_boundary_registry: BoundaryOwnerRegistryV2,
}

impl<SC: SCStarkGenericConfig, A, AE> SCStarkMachine<SC, A, AE> {
    /// Creates a new [`SCStarkMachine`].
    pub fn new(
        config: SC,
        chips: Vec<Chip<Val<SC>, A>>,
        chips_ext: Vec<Chip<Challenge<SC>, AE>>,
        num_pv_elts: usize,
        contains_global_bus: bool,
    ) -> Self
    where
        A: MachineAir<Val<SC>>,
    {
        let owners = chips
            .iter()
            .filter_map(MachineAir::global_boundary_owner)
            .map(|owner| BoundaryOwnerV2 { owner, kind: GlobalBoundaryKindV2::Projective })
            .collect();
        let global_boundary_registry = BoundaryOwnerRegistryV2::new(owners)
            .expect("machine Global boundary owner registry must be canonical");
        Self {
            config,
            chips,
            chips_ext,
            num_pv_elts,
            contains_global_bus,
            global_boundary_registry,
        }
    }

    // TODO: remove this
    /// Returns whether the machine has a global bus (true for core, false for recursion).
    pub const fn has_global_bus(&self) -> bool {
        self.contains_global_bus
    }

    /// Return the immutable owner registry bound into the key authority.
    pub const fn global_boundary_registry(&self) -> &BoundaryOwnerRegistryV2 {
        &self.global_boundary_registry
    }
}

/// A proving key for a STARK.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "PcsProverData<SC>: Serialize"))]
#[serde(bound(deserialize = "PcsProverData<SC>: DeserializeOwned"))]
pub struct StarkProvingKey<SC: StarkGenericConfig> {
    /// The commitment to the preprocessed traces.
    pub commit: Com<SC>,
    /// The start pc of the program.
    pub pc_start: Val<SC>,
    /// Program-image boundary authenticated by the projective Global relation.
    pub program_boundary: ProgramImageBoundaryV1<u32>,
    /// Ordered Global owner inventory.
    pub owner_registry: BoundaryOwnerRegistryV2,
    /// Product-local Global146 protocol identity.
    pub global146_identity: [u8; 32],
    /// The preprocessed traces.
    pub traces: Vec<RowMajorMatrix<Val<SC>>>,
    /// The pcs data for the preprocessed traces.
    pub data: PcsProverData<SC>,
    /// The preprocessed chip ordering.
    pub chip_ordering: HashMap<String, usize>,
    /// The preprocessed chip local only information.
    pub local_only: Vec<bool>,
    /// The number of total constraints for each chip.
    pub constraints_map: HashMap<String, usize>,
}

impl<SC: StarkGenericConfig> StarkProvingKey<SC> {
    /// Observes the values of the proving key into the challenger.
    pub fn observe_into(&self, challenger: &mut SC::Challenger)
    where
        Val<SC>: PrimeField32,
    {
        validate_global146_identity(&self.global146_identity)
            .expect("admitted proving key has the current Global146 identity");
        challenger.observe(self.commit.clone());
        challenger.observe(self.pc_start);
        observe_program_boundary_v1::<Val<SC>, _>(challenger, &self.program_boundary)
            .expect("admitted proving key has a valid program boundary");
        observe_owner_registry_v2::<Val<SC>, _>(challenger, &self.owner_registry)
            .expect("admitted proving key has a valid owner registry");
    }
}

/// A verifying key for a STARK.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "Dom<SC>: Serialize"))]
#[serde(bound(deserialize = "Dom<SC>: DeserializeOwned"))]
pub struct StarkVerifyingKey<SC: StarkGenericConfig> {
    /// The commitment to the preprocessed traces.
    pub commit: Com<SC>,
    /// The start pc of the program.
    pub pc_start: Val<SC>,
    /// Program-image boundary authenticated by the projective Global relation.
    pub program_boundary: ProgramImageBoundaryV1<u32>,
    /// Ordered Global owner inventory.
    pub owner_registry: BoundaryOwnerRegistryV2,
    /// Product-local Global146 protocol identity.
    pub global146_identity: [u8; 32],
    /// The chip information.
    pub chip_information: Vec<(String, Dom<SC>, Dimensions)>,
    /// The chip ordering.
    pub chip_ordering: HashMap<String, usize>,
}

impl<SC: StarkGenericConfig> StarkVerifyingKey<SC> {
    /// Observes the values of the verifying key into the challenger.
    pub fn observe_into(&self, challenger: &mut SC::Challenger)
    where
        Val<SC>: PrimeField32,
    {
        validate_global146_identity(&self.global146_identity)
            .expect("admitted verifying key has the current Global146 identity");
        challenger.observe(self.commit.clone());
        challenger.observe(self.pc_start);
        observe_program_boundary_v1::<Val<SC>, _>(challenger, &self.program_boundary)
            .expect("admitted verifying key has a valid program boundary");
        observe_owner_registry_v2::<Val<SC>, _>(challenger, &self.owner_registry)
            .expect("admitted verifying key has a valid owner registry");
    }
}

impl<SC: StarkGenericConfig> Debug for StarkVerifyingKey<SC> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VerifyingKey").finish()
    }
}

impl StarkVerifyingKey<crate::baby_bear_poseidon2::BabyBearPoseidon2> {
    /// Canonical host hash matching the direct recursion-circuit VK hash.
    #[must_use]
    pub fn hash_babybear(&self) -> [BabyBear; crate::DIGEST_SIZE] {
        validate_global146_identity(&self.global146_identity)
            .expect("hashed direct VK has the current Global146 identity");
        self.owner_registry.validate().expect("hashed direct VK has a canonical owner registry");

        let mut inputs = Vec::new();
        let commitment: &[BabyBear; crate::DIGEST_SIZE] = self.commit.as_ref();
        inputs.extend_from_slice(commitment);
        inputs.push(self.pc_start);
        inputs.extend(
            crate::global_d11::canonical_program_boundary_transcript_fields_v1::<BabyBear>(
                &self.program_boundary,
            )
            .expect("hashed direct VK has a canonical program boundary"),
        );
        inputs.push(BabyBear::from_canonical_usize(self.owner_registry.owners.len()));
        for entry in &self.owner_registry.owners {
            inputs.push(BabyBear::from_canonical_u32(entry.owner.0));
            inputs.push(BabyBear::from_canonical_u8(entry.kind as u8));
        }
        inputs.extend(self.owner_registry.digest.map(BabyBear::from_canonical_u8));
        inputs.extend(self.global146_identity.map(BabyBear::from_canonical_u8));
        let seed = crate::global_d11::program_global_seed::<BabyBear>(&self.program_boundary)
            .expect("hashed direct VK has a canonical ProgramGlobalSeed");
        inputs.extend_from_slice(seed.x.coefficients());
        inputs.extend_from_slice(seed.y.coefficients());
        inputs.extend_from_slice(seed.z.coefficients());
        for (name, domain, dimensions) in &self.chip_information {
            inputs.push(BabyBear::from_canonical_usize(domain.log_n));
            inputs.push(BabyBear::from_canonical_usize(1 << domain.log_n));
            inputs.push(domain.shift);
            inputs.push(BabyBear::two_adic_generator(domain.log_n));
            inputs.push(BabyBear::from_canonical_usize(dimensions.width));
            inputs.push(BabyBear::from_canonical_usize(dimensions.height));
            inputs.push(BabyBear::from_canonical_usize(name.len()));
            inputs.extend(name.bytes().map(BabyBear::from_canonical_u8));
        }

        crate::InnerHash::new(crate::inner_perm()).hash_iter(inputs)
    }
}

impl<SC: StarkGenericConfig, A: MachineAir<Val<SC>>> StarkMachine<SC, A> {
    /// Returns an iterator over the chips in the machine that are included in the given shard.
    pub fn shard_chips_ordered<'a, 'b>(
        &'a self,
        chip_ordering: &'b HashMap<String, usize>,
    ) -> impl Iterator<Item = &'b MachineChip<SC, A>>
    where
        'a: 'b,
    {
        self.chips
            .iter()
            .filter(|chip| chip_ordering.contains_key(&chip.name()))
            .sorted_by_key(|chip| chip_ordering.get(&chip.name()))
    }

    /// Returns the config of the machine.
    pub const fn config(&self) -> &SC {
        &self.config
    }

    /// Get an array containing a `ChipRef` for all the chips of this RISC-V STARK machine.
    pub fn chips(&self) -> &[MachineChip<SC, A>] {
        &self.chips
    }

    /// Returns the number of public values elements.
    pub const fn num_pv_elts(&self) -> usize {
        self.num_pv_elts
    }

    /// Returns an iterator over the chips in the machine that are included in the given shard.
    pub fn shard_chips<'a, 'b>(
        &'a self,
        shard: &'b A::Record,
    ) -> impl Iterator<Item = &'b MachineChip<SC, A>>
    where
        'a: 'b,
    {
        self.chips.iter().filter(|chip| chip.included(shard))
    }

    /// Debugs the constraints of the given records.
    #[allow(clippy::too_many_lines, clippy::needless_pass_by_value)]
    #[instrument("debug constraints", level = "debug", skip_all)]
    pub fn debug_constraints(
        &self,
        pk: &StarkProvingKey<SC>,
        records: Vec<A::Record>,
        challenger: &mut SC::Challenger,
    ) where
        SC::Val: PrimeField32,
        A: for<'a> Air<DebugConstraintBuilder<'a, Val<SC>, Challenge<SC>>>,
    {
        tracing::debug!("checking constraints for each shard");

        // Obtain the challenges used for the global permutation argument.
        let mut permutation_challenges: Vec<Challenge<SC>> = Vec::new();
        for _ in 0..2 {
            permutation_challenges.push(challenger.sample_ext_element());
        }

        for shard in records.iter() {
            // Filter the chips based on what is used.
            let chips = self.shard_chips(shard).collect::<Vec<_>>();

            // Generate the main trace for each chip.
            let pre_traces = chips
                .iter()
                .map(|chip| pk.chip_ordering.get(&chip.name()).map(|index| &pk.traces[*index]))
                .collect::<Vec<_>>();
            let mut traces = chips
                .par_iter()
                .map(|chip| {
                    let compressed = chip.generate_trace(shard, &mut A::Record::default());
                    compressed.decompress()
                })
                .zip(pre_traces)
                .collect::<Vec<_>>();

            // Generate the permutation traces.
            let mut permutation_traces = Vec::with_capacity(chips.len());
            let mut chip_cumulative_sums: Vec<Challenge<SC>> = Vec::with_capacity(chips.len());
            tracing::debug_span!("generate permutation traces").in_scope(|| {
                chips
                    .par_iter()
                    .zip(traces.par_iter_mut())
                    .map(|(chip, (main_trace, pre_trace))| {
                        let (trace, local_sum) = chip.generate_permutation_trace(
                            *pre_trace,
                            main_trace,
                            &permutation_challenges,
                        );
                        (trace, local_sum)
                    })
                    .unzip_into_vecs(&mut permutation_traces, &mut chip_cumulative_sums);
            });

            let local_cumulative_sum = chip_cumulative_sums.iter().copied().sum::<Challenge<SC>>();

            // Compute expected imbalance from public values.
            let pv_vec = shard.public_values::<SC::Val>();
            let expected_local_sum =
                compute_expected_state_imbalance::<SC>(&pv_vec, &permutation_challenges);

            if local_cumulative_sum != expected_local_sum {
                tracing::warn!(
                    "Local cumulative sum mismatch: actual = {:?}, expected = {:?}",
                    local_cumulative_sum,
                    expected_local_sum,
                );
                tracing::debug_span!("debug local interactions").in_scope(|| {
                    debug_interactions_with_all_chips::<SC, A>(
                        self,
                        pk,
                        std::slice::from_ref(shard),
                        InteractionKind::all_kinds(),
                        InteractionScope::Local,
                    )
                });
                panic!("Local cumulative sum does not match expected state imbalance");
            }

            // Compute some statistics.
            for i in 0..chips.len() {
                let trace_width = traces[i].0.width();
                let pre_width = traces[i].1.map_or(0, p3_matrix::Matrix::width);
                let permutation_width = permutation_traces[i].width() *
                    <Challenge<SC> as AbstractExtensionField<SC::Val>>::D;
                let total_width = trace_width + pre_width + permutation_width;
                tracing::debug!(
                    "{:<11} | Main Cols = {:<5} | Pre Cols = {:<5} | Perm Cols = {:<5} | Rows = {:<10} | Cells = {:<10}",
                    chips[i].name(),
                    trace_width,
                    pre_width,
                    permutation_width,
                    traces[i].0.height(),
                    total_width * traces[i].0.height(),
                );
            }

            if env::var("SKIP_CONSTRAINTS").is_err() {
                tracing::info_span!("debug constraints").in_scope(|| {
                    for i in 0..chips.len() {
                        let preprocessed_trace =
                            pk.chip_ordering.get(&chips[i].name()).map(|index| &pk.traces[*index]);
                        debug_constraints::<SC, A>(
                            chips[i],
                            preprocessed_trace,
                            &traces[i].0,
                            &permutation_traces[i],
                            &permutation_challenges,
                            &shard.public_values(),
                            &chip_cumulative_sums[i],
                        );
                    }
                });
            }
        }

        tracing::info!("Constraints verified successfully");
    }
}

impl<SC: StarkGenericConfig, A: MachineAir<Val<SC>> + Air<SymbolicAirBuilder<Val<SC>>>>
    StarkMachine<SC, A>
{
    /// Returns whether the machine contains a global bus.
    pub const fn contains_global_bus(&self) -> bool {
        self.contains_global_bus
    }

    /// Returns the id of all chips in the machine that have preprocessed columns.
    pub fn preprocessed_chip_ids(&self) -> Vec<usize> {
        self.chips
            .iter()
            .enumerate()
            .filter(|(_, chip)| chip.preprocessed_width() > 0)
            .map(|(i, _)| i)
            .collect()
    }

    /// Returns the indices of the chips in the machine that are included in the given shard.
    pub fn chips_sorted_indices(&self, proof: &ShardProof<SC>) -> Vec<Option<usize>> {
        self.chips().iter().map(|chip| proof.chip_ordering.get(&chip.name()).copied()).collect()
    }

    /// The setup preprocessing phase used by the canonical direct surface.
    pub fn setup_core(&self, program: &A::Program) -> (StarkProvingKey<SC>, StarkVerifyingKey<SC>) {
        let parent_span = tracing::debug_span!("generate preprocessed traces");
        let (named_preprocessed_traces, num_constraints): (Vec<_>, Vec<_>) =
            parent_span.in_scope(|| {
                self.chips()
                    .par_iter()
                    .map(|chip| {
                        let chip_name = chip.name();
                        let begin = Instant::now();
                        let prep_trace = chip.generate_preprocessed_trace(program);
                        tracing::debug!(
                            parent: &parent_span,
                            "generated preprocessed trace for chip {} in {:?}",
                            chip_name,
                            begin.elapsed()
                        );
                        // Assert that the chip width data is correct.
                        let expected_width = prep_trace.as_ref().map_or(0, CompressedMatrix::width);
                        assert_eq!(
                            expected_width,
                            chip.preprocessed_width(),
                            "Incorrect number of preprocessed columns for chip {chip_name}"
                        );

                        // Count the number of constraints.
                        let num_main_constraints = get_symbolic_constraints(
                            &chip.air,
                            chip.preprocessed_width(),
                            PROOF_MAX_NUM_PVS,
                        )
                        .len();

                        let num_permutation_constraints = count_permutation_constraints(
                            &chip.sends,
                            &chip.receives,
                            chip.logup_batch_size(),
                            chip.air.commit_scope(),
                        );

                        (
                            prep_trace.map(move |t| (chip.name(), chip.local_only(), t)),
                            (chip_name, num_main_constraints + num_permutation_constraints),
                        )
                    })
                    .unzip()
            });

        let mut named_preprocessed_traces =
            named_preprocessed_traces.into_iter().flatten().collect::<Vec<_>>();

        // Order the chips and traces by trace size (biggest first), and get the ordering map.
        named_preprocessed_traces
            .sort_by_key(|(name, _, trace)| (Reverse(trace.dimensions().height), name.clone()));

        let pcs = self.config.pcs();
        let (chip_information, domains_and_traces): (Vec<_>, Vec<_>) = named_preprocessed_traces
            .iter()
            .map(|(name, _, trace)| {
                let decompressed = trace.decompress();
                let domain = pcs.natural_domain_for_degree(decompressed.height());
                ((name.to_owned(), domain, trace.dimensions()), (domain, decompressed))
            })
            .unzip();

        // Commit to the batch of traces (decompressed for PCS).
        let (commit, data) = tracing::debug_span!("commit to preprocessed traces")
            .in_scope(|| pcs.commit(domains_and_traces));

        // Get the chip ordering.
        let chip_ordering = named_preprocessed_traces
            .iter()
            .enumerate()
            .map(|(i, (name, _, _))| (name.to_owned(), i))
            .collect::<HashMap<_, _>>();

        let local_only = named_preprocessed_traces
            .iter()
            .map(|(_, local_only, _)| local_only.to_owned())
            .collect::<Vec<_>>();

        let constraints_map: HashMap<_, _> = num_constraints.into_iter().collect();

        // Get the preprocessed traces (decompressed for StarkProvingKey storage).
        let traces = named_preprocessed_traces
            .into_iter()
            .map(|(_, _, trace)| trace.decompress())
            .collect::<Vec<_>>();

        let pc_start = program.pc_start();
        let owner_registry = self.global_boundary_registry.clone();
        owner_registry.validate().expect("machine owner registry must remain canonical");
        let program_boundary = if owner_registry.owners.is_empty() {
            ProgramImageBoundaryV1::Infinity
        } else {
            program
                .initial_global_boundary()
                .expect("Global machine program must provide a canonical program-image boundary")
        };

        (
            StarkProvingKey {
                commit: commit.clone(),
                pc_start,
                program_boundary: program_boundary.clone(),
                owner_registry: owner_registry.clone(),
                global146_identity: GLOBAL146_COMPOSITE_IDENTITY,
                traces,
                data,
                chip_ordering: chip_ordering.clone(),
                local_only,
                constraints_map,
            },
            StarkVerifyingKey {
                commit,
                pc_start,
                program_boundary,
                owner_registry,
                global146_identity: GLOBAL146_COMPOSITE_IDENTITY,
                chip_information,
                chip_ordering,
            },
        )
    }

    /// The setup preprocessing phase.
    ///
    /// Given a program, this function generates the proving and verifying keys. The keys correspond
    /// to the program code and other preprocessed colunms such as lookup tables.
    #[instrument("setup machine", level = "debug", skip_all)]
    #[allow(clippy::map_unwrap_or)]
    #[allow(clippy::redundant_closure_for_method_calls)]
    pub fn setup(&self, program: &A::Program) -> (StarkProvingKey<SC>, StarkVerifyingKey<SC>) {
        self.setup_core(program)
    }

    /// Generates the dependencies of the given records.
    #[allow(clippy::needless_for_each)]
    pub fn generate_dependencies(
        &self,
        records: &mut [A::Record],
        opts: &<A::Record as MachineRecord>::Config,
        chips_filter: Option<&[String]>,
    ) {
        let chips = self
            .chips
            .iter()
            .filter(|chip| {
                if let Some(chips_filter) = chips_filter {
                    chips_filter.contains(&chip.name())
                } else {
                    true
                }
            })
            .collect::<Vec<_>>();

        records.iter_mut().for_each(|record| {
            chips.iter().for_each(|chip| {
                let mut output = A::Record::default();
                chip.generate_dependencies(record, &mut output);
                record.append(&mut output);
            });
            tracing::debug_span!("register nonces").in_scope(|| record.register_nonces(opts));
        });
    }

    /// Verify that a proof is complete and valid given a verifying key and a claimed digest.
    #[instrument("verify", level = "info", skip_all)]
    #[allow(clippy::match_bool)]
    pub fn verify(
        &self,
        vk: &StarkVerifyingKey<SC>,
        proof: &MachineProof<SC>,
        challenger: &mut SC::Challenger,
    ) -> Result<(), MachineVerificationError<SC>>
    where
        Val<SC>: PrimeField32,
        SC::Challenger: Clone,
        A: for<'a> Air<VerifierConstraintFolder<'a, SC>>,
    {
        let interaction_kinds = InteractionKind::all_kinds();
        self.verify_with_interaction_kinds(vk, proof, challenger, &interaction_kinds)
    }

    /// Verify a machine proof with an explicit active interaction kind set.
    #[instrument("verify", level = "info", skip_all)]
    #[allow(clippy::match_bool)]
    pub fn verify_with_interaction_kinds(
        &self,
        vk: &StarkVerifyingKey<SC>,
        proof: &MachineProof<SC>,
        challenger: &mut SC::Challenger,
        interaction_kinds: &[InteractionKind],
    ) -> Result<(), MachineVerificationError<SC>>
    where
        Val<SC>: PrimeField32,
        SC::Challenger: Clone,
        A: for<'a> Air<VerifierConstraintFolder<'a, SC>>,
    {
        if self.global_boundary_registry != vk.owner_registry ||
            vk.owner_registry.validate().is_err()
        {
            return Err(MachineVerificationError::InvalidVerificationKey);
        }

        // Observe the preprocessed commitment and direct-role Global metadata.
        vk.observe_into(challenger);

        // Verify the shard proofs.
        if proof.shard_proofs.is_empty() {
            return Err(MachineVerificationError::EmptyProof);
        }

        tracing::debug_span!("verify shard proofs").in_scope(|| {
            for (i, shard_proof) in proof.shard_proofs.iter().enumerate() {
                tracing::debug_span!("verifying shard", shard = i).in_scope(|| {
                    let chips =
                        self.shard_chips_ordered(&shard_proof.chip_ordering).collect::<Vec<_>>();
                    let mut shard_challenger = challenger.clone();
                    shard_challenger
                        .observe_slice(&shard_proof.public_values[0..self.num_pv_elts()]);
                    Verifier::verify_shard_with_interaction_kinds(
                        &self.config,
                        vk,
                        &chips,
                        &mut shard_challenger,
                        shard_proof,
                        interaction_kinds,
                    )
                    .map_err(MachineVerificationError::InvalidShardProof)
                })?;
            }

            Ok(())
        })?;

        if !vk.owner_registry.owners.is_empty() {
            let claims = proof
                .shard_proofs
                .iter()
                .map(|shard| {
                    let pv: &PublicValues<Word<Val<SC>>, Val<SC>> =
                        shard.public_values.as_slice().borrow();
                    pv.global
                })
                .collect::<Vec<_>>();
            verify_global_interval_root_v4(&vk.program_boundary, &claims).map_err(|_| {
                MachineVerificationError::InvalidPublicValues("invalid Global interval chain")
            })?;
        }
        Ok(())
    }

    /// Verify a native-recursion machine proof with recursion-only interaction kinds.
    #[instrument("verify", level = "info", skip_all)]
    #[allow(clippy::match_bool)]
    pub fn verify_with_recursion_interactions(
        &self,
        vk: &StarkVerifyingKey<SC>,
        proof: &MachineProof<SC>,
        challenger: &mut SC::Challenger,
    ) -> Result<(), MachineVerificationError<SC>>
    where
        SC::Challenger: Clone,
        A: for<'a> Air<VerifierConstraintFolder<'a, SC>>,
    {
        self.verify_with_interaction_kinds(
            vk,
            proof,
            challenger,
            InteractionKind::recursion_kinds(),
        )
    }
}

impl<SC: SCStarkGenericConfig, A: MachineAir<Val<SC>>, AE: MachineAir<Challenge<SC>>>
    SCStarkMachine<SC, A, AE>
{
    /// Returns an iterator over the chips in the machine that are included in the given shard.
    pub fn shard_chips_ordered<'a, 'b>(
        &'a self,
        chip_ordering: &'b HashMap<String, usize>,
    ) -> impl Iterator<Item = &'b MachineChip<SC, A>>
    where
        'a: 'b,
    {
        self.chips
            .iter()
            .filter(|chip| chip_ordering.contains_key(&chip.name()))
            .sorted_by_key(|chip| chip_ordering.get(&chip.name()))
    }
    /// Returns an iterator over the chips in the machine that are included in the given shard.
    pub fn shard_chips_ext_ordered<'a, 'b>(
        &'a self,
        chip_ordering: &'b HashMap<String, usize>,
    ) -> impl Iterator<Item = &'b SumcheckChip<SC, AE>>
    where
        'a: 'b,
    {
        self.chips_ext
            .iter()
            .filter(|chip| chip_ordering.contains_key(&chip.name()))
            .sorted_by_key(|chip| chip_ordering.get(&chip.name()))
    }
    /// Returns the config of the machine.
    pub const fn config(&self) -> &SC {
        &self.config
    }

    /// Get an array containing a `ChipRef` for all the chips of this RISC-V STARK machine.
    pub fn chips(&self) -> &[MachineChip<SC, A>] {
        &self.chips
    }

    /// Get an array containing a `ChipRef` for all the chips ext of this RISC-V STARK machine.
    pub fn chips_ext(&self) -> &[SumcheckChip<SC, AE>] {
        &self.chips_ext
    }

    /// Returns the number of public values elements.
    pub const fn num_pv_elts(&self) -> usize {
        self.num_pv_elts
    }

    /// Returns an iterator over the chips in the machine that are included in the given shard.
    pub fn shard_chips<'a, 'b>(
        &'a self,
        shard: &'b A::Record,
    ) -> impl Iterator<Item = &'b MachineChip<SC, A>>
    where
        'a: 'b,
    {
        self.chips.iter().filter(|chip| chip.included(shard))
    }
    /// Returns an iterator over the `chips_ext` in the machine that are included in the given
    /// shard.
    pub fn shard_chips_ext<'a, 'b>(
        &'a self,
        shard: &'b AE::Record,
    ) -> impl Iterator<Item = &'b SumcheckChip<SC, AE>>
    where
        'a: 'b,
    {
        self.chips_ext.iter().filter(|chip| chip.included(shard))
    }

    /// Debugs the constraints of the given records.
    #[allow(clippy::too_many_lines, clippy::needless_pass_by_value)]
    #[instrument("debug constraints", level = "debug", skip_all)]
    pub fn debug_constraints(
        &self,
        pk: &SCStarkProvingKey<SC>,
        records: Vec<A::Record>,
        challenger: &mut SC::MlChallenger,
    ) where
        SC::Val: PrimeField32,
        A: for<'a> Air<DebugConstraintBuilder<'a, Val<SC>, Challenge<SC>>>,
    {
        tracing::debug!("checking constraints for each shard");

        // Obtain the challenges used for the global permutation argument.
        let mut permutation_challenges: Vec<Challenge<SC>> = Vec::new();
        for _ in 0..2 {
            permutation_challenges.push(challenger.sample_ext_element());
        }

        for shard in records.iter() {
            // Filter the chips based on what is used.
            let chips = self.shard_chips(shard).collect::<Vec<_>>();

            // Generate the main trace for each chip.
            let pre_traces_decompressed: Vec<Option<RowMajorMatrix<Val<SC>>>> = chips
                .iter()
                .map(|chip| {
                    pk.chip_ordering.get(&chip.name()).map(|index| pk.traces[*index].decompress())
                })
                .collect();
            let pre_traces: Vec<Option<&RowMajorMatrix<Val<SC>>>> =
                pre_traces_decompressed.iter().map(|t| t.as_ref()).collect();
            let mut traces = chips
                .par_iter()
                .map(|chip| {
                    let compressed = chip.generate_trace(shard, &mut A::Record::default());
                    compressed.decompress()
                })
                .zip(pre_traces)
                .collect::<Vec<_>>();

            // Generate the permutation traces.
            let mut permutation_traces = Vec::with_capacity(chips.len());
            let mut chip_cumulative_sums: Vec<Challenge<SC>> = Vec::with_capacity(chips.len());
            tracing::debug_span!("generate permutation traces").in_scope(|| {
                chips
                    .par_iter()
                    .zip(traces.par_iter_mut())
                    .map(|(chip, (main_trace, pre_trace))| {
                        let (trace, local_sum) = chip.generate_permutation_trace(
                            *pre_trace,
                            main_trace,
                            &permutation_challenges,
                        );
                        (trace, local_sum)
                    })
                    .unzip_into_vecs(&mut permutation_traces, &mut chip_cumulative_sums);
            });

            let local_cumulative_sum = chip_cumulative_sums.iter().copied().sum::<Challenge<SC>>();

            // Compute expected imbalance from public values (only for core proofs).
            let expected_local_sum = if self.contains_global_bus {
                let pv_vec = shard.public_values::<Val<SC>>();
                compute_expected_state_imbalance::<SC>(&pv_vec, &permutation_challenges)
            } else {
                Challenge::<SC>::zero()
            };

            if local_cumulative_sum != expected_local_sum {
                tracing::error!("Per-chip local cumulative sums:");
                for i in 0..chips.len() {
                    let local_sum = chip_cumulative_sums[i];
                    if local_sum != Challenge::<SC>::zero() {
                        tracing::error!(
                            "  chip[{}] {} local_sum={:?}",
                            i,
                            chips[i].name(),
                            local_sum
                        );
                    }
                }
                tracing::error!(
                    "total_local_sum={:?}, expected={:?}",
                    local_cumulative_sum,
                    expected_local_sum
                );

                if std::env::var("DEBUG_INTERACTIONS_FULL").is_ok() {
                    tracing::debug_span!("debug local interactions").in_scope(|| {
                        debug_interactions_with_all_chips_sumcheck::<SC, A, AE>(
                            self,
                            pk,
                            std::slice::from_ref(shard),
                            InteractionKind::all_kinds(),
                            InteractionScope::Local,
                        )
                    });
                }
                panic!("Local cumulative sum does not match expected state imbalance");
            }

            // Compute some statistics.
            for i in 0..chips.len() {
                let trace_width = traces[i].0.width();
                let pre_width = traces[i].1.map_or(0, p3_matrix::Matrix::width);
                let permutation_width: usize = permutation_traces[i].width() *
                    <Challenge<SC> as AbstractExtensionField<Val<SC>>>::D;
                let total_width = trace_width + pre_width + permutation_width;
                tracing::debug!(
                    "{:<11} | Main Cols = {:<5} | Pre Cols = {:<5} | Perm Cols = {:<5} | Rows = {:<10} | Cells = {:<10}",
                    chips[i].name(),
                    trace_width,
                    pre_width,
                    permutation_width,
                    traces[i].0.height(),
                    total_width * traces[i].0.height(),
                );
            }

            if env::var("SKIP_CONSTRAINTS").is_err() {
                tracing::info_span!("debug constraints").in_scope(|| {
                    for i in 0..chips.len() {
                        let decompressed = pk
                            .chip_ordering
                            .get(&chips[i].name())
                            .map(|index| pk.traces[*index].decompress());
                        let preprocessed_trace = decompressed.as_ref();
                        debug_constraints::<SC, A>(
                            chips[i],
                            preprocessed_trace,
                            &traces[i].0,
                            &permutation_traces[i],
                            &permutation_challenges,
                            &shard.public_values(),
                            &chip_cumulative_sums[i],
                        );
                    }
                });
            }
        }

        tracing::info!("Constraints verified successfully");
    }
}

impl<
        SC: SCStarkGenericConfig,
        A: MachineAir<Val<SC>> + Air<SymbolicAirBuilder<Val<SC>>>,
        AE: MachineAir<Challenge<SC>>,
    > SCStarkMachine<SC, A, AE>
{
    /// Returns whether the machine contains a global bus.
    pub const fn contains_global_bus(&self) -> bool {
        self.contains_global_bus
    }

    /// Returns the id of all chips in the machine that have preprocessed columns.
    pub fn preprocessed_chip_ids(&self) -> Vec<usize> {
        self.chips
            .iter()
            .enumerate()
            .filter(|(_, chip)| chip.preprocessed_width() > 0)
            .map(|(i, _)| i)
            .collect()
    }

    /// Returns the indices of the chips in the machine that are included in the given shard.
    pub fn chips_sorted_indices(&self, proof: &ShardProof<SC>) -> Vec<Option<usize>> {
        self.chips().iter().map(|chip| proof.chip_ordering.get(&chip.name()).copied()).collect()
    }

    /// The setup preprocessing phase using compressed preprocessed traces when available.
    pub fn setup_core(
        &self,
        program: &A::Program,
    ) -> (SCStarkProvingKey<SC>, SCStarkVerifyingKey<SC>) {
        use crate::sumcheck::trace::CompressedMatrix;
        let parent_span = tracing::debug_span!("generate preprocessed traces (v2)");
        let (named_preprocessed_traces, num_constraints): (Vec<_>, Vec<_>) =
            parent_span.in_scope(|| {
                self.chips()
                    .par_iter()
                    .map(|chip| {
                        let chip_name = chip.name();
                        let begin = Instant::now();
                        let prep_trace = chip.generate_preprocessed_trace(program);
                        tracing::debug!(
                            parent: &parent_span,
                            "generated preprocessed trace for chip {} in {:?}",
                            chip_name,
                            begin.elapsed()
                        );
                        // Assert that the chip width data is correct.
                        let expected_width = prep_trace.as_ref().map_or(0, CompressedMatrix::width);
                        assert_eq!(
                            expected_width,
                            chip.preprocessed_width(),
                            "Incorrect number of preprocessed columns for chip {chip_name}"
                        );

                        // Count the number of constraints.
                        let num_main_constraints = get_symbolic_constraints(
                            &chip.air,
                            chip.preprocessed_width(),
                            PROOF_MAX_NUM_PVS,
                        )
                        .len();

                        let num_permutation_constraints = count_permutation_constraints(
                            &chip.sends,
                            &chip.receives,
                            chip.logup_batch_size(),
                            chip.air.commit_scope(),
                        );

                        (
                            prep_trace.map(move |t| (chip.name(), chip.local_only(), t)),
                            (chip_name, num_main_constraints + num_permutation_constraints),
                        )
                    })
                    .unzip()
            });

        let mut named_preprocessed_traces =
            named_preprocessed_traces.into_iter().flatten().collect::<Vec<_>>();

        // Order the chips and traces by trace size (biggest first), and get the ordering map.
        named_preprocessed_traces
            .sort_by_key(|(name, _, trace)| (Reverse(trace.height()), name.clone()));

        let pcs = self.config.mlpcs();
        let (chip_information, prep_traces): (Vec<_>, Vec<_>) = named_preprocessed_traces
            .iter()
            .map(|(name, _, c)| ((name.to_owned(), c.dimensions()), c))
            .unzip();
        let preprocessed_max_log_height = prep_traces
            .iter()
            .filter(|trace| trace.width() > 0)
            .map(|trace| log2_strict_usize(trace.height()))
            .max();
        let preprocessed_stack_log_height =
            self.config.mlpcs_target_stack_log_height(preprocessed_max_log_height);
        let commit_options =
            self.config.mlpcs_commit_options_for_stack_log_height(preprocessed_stack_log_height);

        let (commit, data) = tracing::debug_span!("commit to preprocessed traces")
            .in_scope(|| pcs.commit_with_options(prep_traces, commit_options));

        // Get the chip ordering.
        let chip_ordering = named_preprocessed_traces
            .iter()
            .enumerate()
            .map(|(i, (name, _, _))| (name.to_owned(), i))
            .collect::<HashMap<_, _>>();

        let local_only = named_preprocessed_traces
            .iter()
            .map(|(_, local_only, _)| local_only.to_owned())
            .collect::<Vec<_>>();

        let constraints_map: HashMap<_, _> = num_constraints.into_iter().collect();

        // Get the preprocessed traces
        let traces =
            named_preprocessed_traces.into_iter().map(|(_, _, trace)| trace).collect::<Vec<_>>();

        let pc_start = program.pc_start();
        let owner_registry = self.global_boundary_registry.clone();
        owner_registry.validate().expect("machine owner registry must remain canonical");
        let program_boundary = if owner_registry.owners.is_empty() {
            ProgramImageBoundaryV1::Infinity
        } else {
            program
                .initial_global_boundary()
                .expect("Global machine program must provide a canonical program-image boundary")
        };

        (
            SCStarkProvingKey {
                commit: commit.clone(),
                pc_start,
                program_boundary: program_boundary.clone(),
                owner_registry: owner_registry.clone(),
                global146_identity: GLOBAL146_COMPOSITE_IDENTITY,
                traces,
                data,
                preprocessed_pcs_stack_log_height: preprocessed_stack_log_height,
                chip_ordering: chip_ordering.clone(),
                local_only,
                constraints_map: constraints_map.clone(),
            },
            SCStarkVerifyingKey {
                commit,
                pc_start,
                program_boundary,
                owner_registry,
                global146_identity: GLOBAL146_COMPOSITE_IDENTITY,
                chip_information,
                chip_ordering,
                constraints_map,
            },
        )
    }

    /// The setup preprocessing phase.
    ///
    /// Given a program, this function generates the proving and verifying keys. The keys correspond
    /// to the program code and other preprocessed columns such as lookup tables.
    #[instrument("setup machine", level = "debug", skip_all)]
    #[allow(clippy::map_unwrap_or)]
    #[allow(clippy::redundant_closure_for_method_calls)]
    pub fn setup(&self, program: &A::Program) -> (SCStarkProvingKey<SC>, SCStarkVerifyingKey<SC>) {
        self.setup_core(program)
    }

    /// Generates the dependencies of the given records.
    #[allow(clippy::needless_for_each)]
    pub fn generate_dependencies(
        &self,
        records: &mut [A::Record],
        opts: &<A::Record as MachineRecord>::Config,
        chips_filter: Option<&[String]>,
    ) {
        let chips = self
            .chips
            .iter()
            .filter(|chip| {
                if let Some(chips_filter) = chips_filter {
                    chips_filter.contains(&chip.name())
                } else {
                    true
                }
            })
            .collect::<Vec<_>>();

        records.iter_mut().for_each(|record| {
            chips.iter().for_each(|chip| {
                let mut output = A::Record::default();
                chip.generate_dependencies(record, &mut output);
                record.append(&mut output);
            });
            tracing::debug_span!("register nonces").in_scope(|| record.register_nonces(opts));
        });
    }

    /// Verify a machine proof.
    #[instrument("verify", level = "info", skip_all)]
    pub fn verify(
        &self,
        vk: &SCStarkVerifyingKey<SC>,
        proof: &SCMachineProof<SC>,
        challenger: &mut SC::MlChallenger,
        num_skip_rounds: usize,
        chip_log_height_threshold: usize,
    ) -> Result<(), MachineVerificationError<SC>>
    where
        Val<SC>: PrimeField32,
        SC::MlChallenger: Clone,
        A: for<'a> Air<crate::sumcheck::folder::SumcheckVerifierConstraintFolder<'a, SC>>,
    {
        let interaction_kinds = InteractionKind::all_kinds();
        self.verify_with_interaction_kinds(
            vk,
            proof,
            challenger,
            num_skip_rounds,
            chip_log_height_threshold,
            &interaction_kinds,
        )
    }

    /// Verify a machine proof with an explicit active interaction kind set.
    #[instrument("verify", level = "info", skip_all)]
    pub fn verify_with_interaction_kinds(
        &self,
        vk: &SCStarkVerifyingKey<SC>,
        proof: &SCMachineProof<SC>,
        challenger: &mut SC::MlChallenger,
        num_skip_rounds: usize,
        chip_log_height_threshold: usize,
        interaction_kinds: &[InteractionKind],
    ) -> Result<(), MachineVerificationError<SC>>
    where
        Val<SC>: PrimeField32,
        SC::MlChallenger: Clone,
        A: for<'a> Air<crate::sumcheck::folder::SumcheckVerifierConstraintFolder<'a, SC>>,
    {
        tracing::info!("verify with univariate skip parameter k={}", num_skip_rounds);

        if self.global_boundary_registry != vk.owner_registry ||
            vk.owner_registry.validate().is_err()
        {
            return Err(MachineVerificationError::InvalidVerificationKey);
        }

        vk.observe_into(challenger);

        if proof.shard_proofs.is_empty() {
            return Err(MachineVerificationError::EmptyProof);
        }

        tracing::debug_span!("verify shard proofs v2").in_scope(|| {
            for (i, shard_proof) in proof.shard_proofs.iter().enumerate() {
                tracing::debug_span!("verifying shard v2", shard = i).in_scope(|| {
                    let chips =
                        self.shard_chips_ordered(&shard_proof.chip_ordering).collect::<Vec<_>>();
                    let mut shard_challenger = challenger.clone();
                    shard_challenger
                        .observe_slice(&shard_proof.public_values[0..self.num_pv_elts()]);
                    SumcheckVerifier::verify_shard_with_interaction_kinds(
                        &self.config,
                        vk,
                        &chips,
                        &mut shard_challenger,
                        shard_proof,
                        num_skip_rounds,
                        chip_log_height_threshold,
                        self.contains_global_bus,
                        interaction_kinds,
                    )
                    .map_err(MachineVerificationError::InvalidShardProofSumcheck)
                })?;
            }

            Ok(())
        })?;

        if !vk.owner_registry.owners.is_empty() {
            let claims = proof
                .shard_proofs
                .iter()
                .map(|shard| {
                    let pv: &PublicValues<Word<Val<SC>>, Val<SC>> =
                        shard.public_values.as_slice().borrow();
                    pv.global
                })
                .collect::<Vec<_>>();
            verify_global_interval_root_v4(&vk.program_boundary, &claims).map_err(|_| {
                MachineVerificationError::InvalidPublicValues("invalid Global interval chain")
            })?;
        }
        Ok(())
    }

    /// Verify a native-recursion machine proof with recursion-only interaction kinds.
    #[instrument("verify", level = "info", skip_all)]
    pub fn verify_with_recursion_interactions(
        &self,
        vk: &SCStarkVerifyingKey<SC>,
        proof: &SCMachineProof<SC>,
        challenger: &mut SC::MlChallenger,
        num_skip_rounds: usize,
        chip_log_height_threshold: usize,
    ) -> Result<(), MachineVerificationError<SC>>
    where
        SC::MlChallenger: Clone,
        A: for<'a> Air<crate::sumcheck::folder::SumcheckVerifierConstraintFolder<'a, SC>>,
    {
        self.verify_with_interaction_kinds(
            vk,
            proof,
            challenger,
            num_skip_rounds,
            chip_log_height_threshold,
            InteractionKind::recursion_kinds(),
        )
    }
}

/// Errors that can occur during machine verification.
pub enum MachineVerificationError<SC: StarkGenericConfig> {
    /// An error occurred during the verification of a shard proof.
    InvalidShardProof(VerificationError<SC>),
    /// An error occurred during the verification of a shard proof with sumcheck.
    InvalidShardProofSumcheck(SumcheckVerificationError<SC>),
    /// An error occurred during the verification of a global proof.
    InvalidGlobalProof(VerificationError<SC>),
    /// The cumulative sum is non-zero.
    NonZeroCumulativeSum(InteractionScope, usize),
    /// The public values digest is invalid.
    InvalidPublicValuesDigest,
    /// The debug interactions failed.
    DebugInteractionsFailed,
    /// The proof is empty.
    EmptyProof,
    /// The public values are invalid.
    InvalidPublicValues(&'static str),
    /// The number of shards is too large.
    TooManyShards,
    /// The chip occurrence is invalid.
    InvalidChipOccurrence(String),
    /// The CPU is missing in the first shard.
    MissingCpuInFirstShard,
    /// The CPU log degree is too large.
    CpuLogDegreeTooLarge(usize),
    /// The verification key is not allowed.
    InvalidVerificationKey,
    /// The native recursion arm rejected the proof (authoritative: the presented vk
    /// matched the frozen native root vk, so no DSL fallthrough may mask this).
    NativeRecursion(String),
}

impl<SC: StarkGenericConfig> Debug for MachineVerificationError<SC> {
    #[allow(clippy::uninlined_format_args)]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MachineVerificationError::InvalidShardProof(e) => {
                write!(f, "Invalid shard proof: {:?}", e)
            }
            MachineVerificationError::InvalidShardProofSumcheck(e) => {
                write!(f, "Invalid shard proof: {:?}", e)
            }
            MachineVerificationError::InvalidGlobalProof(e) => {
                write!(f, "Invalid global proof: {:?}", e)
            }
            MachineVerificationError::NonZeroCumulativeSum(scope, shard) => {
                write!(f, "Non-zero cumulative sum.  Scope: {}, Shard: {}", scope, shard)
            }
            MachineVerificationError::InvalidPublicValuesDigest => {
                write!(f, "Invalid public values digest")
            }
            MachineVerificationError::EmptyProof => {
                write!(f, "Empty proof")
            }
            MachineVerificationError::DebugInteractionsFailed => {
                write!(f, "Debug interactions failed")
            }
            MachineVerificationError::InvalidPublicValues(s) => {
                write!(f, "Invalid public values: {}", s)
            }
            MachineVerificationError::TooManyShards => {
                write!(f, "Too many shards")
            }
            MachineVerificationError::InvalidChipOccurrence(s) => {
                write!(f, "Invalid chip occurrence: {}", s)
            }
            MachineVerificationError::MissingCpuInFirstShard => {
                write!(f, "Missing CPU in first shard")
            }
            MachineVerificationError::CpuLogDegreeTooLarge(log_degree) => {
                write!(f, "CPU log degree too large: {}", log_degree)
            }
            MachineVerificationError::InvalidVerificationKey => {
                write!(f, "Invalid verification key")
            }
            MachineVerificationError::NativeRecursion(s) => {
                write!(f, "Native recursion verification failed: {}", s)
            }
        }
    }
}

impl<SC: StarkGenericConfig> std::fmt::Display for MachineVerificationError<SC> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self, f)
    }
}

impl<SC: StarkGenericConfig> std::error::Error for MachineVerificationError<SC> {}

impl<SC: StarkGenericConfig> MachineVerificationError<SC> {
    /// This function will check if the verification error is from constraints failing.
    pub fn is_constraints_failing(&self, expected_chip_name: &str) -> bool {
        if let MachineVerificationError::InvalidShardProof(
            VerificationError::OodEvaluationMismatch(chip_name),
        ) = self
        {
            return chip_name == expected_chip_name;
        }

        false
    }

    /// This function will check if the verification error is from local cumulative sum failing.
    pub fn is_local_cumulative_sum_failing(&self) -> bool {
        matches!(
            self,
            MachineVerificationError::InvalidShardProof(VerificationError::CumulativeSumsError(
                "local cumulative sum is not zero"
            ))
        )
    }
}
