pub mod core;
pub mod evaluator;
pub mod permutation;
pub mod precompute;
pub mod prover;
pub mod state;
pub mod symbolic;
pub mod verifier;

use std::{borrow::Borrow, cmp::Reverse};

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

use crate::symbolic::SymbolicAirBuilder;
use dt_stark::{
    air::{FullAir, MachineAir, MachineProgram, PairCol, PolyAirExtendable},
    global_d11::{
        verify_global_interval_root_v4, BoundaryOwnerRegistryV2, BoundaryOwnerV2,
        GlobalBoundaryKindV2, ProgramImageBoundaryV1,
    },
    sumcheck::{
        config::SCStarkGenericConfig,
        keys::{SCStarkProvingKey, SCStarkVerifyingKey},
        proof::SCMachineProof,
    },
    Challenge, MachineRecord, MachineVerificationError, ShardProof, Val,
};
use hashbrown::HashMap;
use itertools::Itertools;
use p3_air::BaseAir;
use p3_challenger::CanObserve;
use p3_field::{Field, PrimeField32};
use p3_matrix::{compressed::CompressedMatrix, dense::RowMajorMatrix};
use p3_maybe_rayon::prelude::*;
use pcs::basefold::mlpcs::MlPCS;

#[derive(Clone)]
pub struct Chip<A, F: Field + PolyAirExtendable<D>, const D: usize> {
    pub air: A,
    pub symbolic_builder: SymbolicAirBuilder<F, D>,
    pub degree: usize,
    pub num_alpha: usize,
}

impl<A, F: Field + PolyAirExtendable<D>, const D: usize> Chip<A, F, D> {
    pub fn new(air: A) -> Self
    where
        A: FullAir<SymbolicAirBuilder<F, D>> + MachineAir<F>,
    {
        Self::new_with_degree_floor(air, 0)
    }

    pub fn new_with_degree_floor(air: A, min_degree: usize) -> Self
    where
        A: FullAir<SymbolicAirBuilder<F, D>> + MachineAir<F>,
    {
        let symbolic_builder = SymbolicAirBuilder::from_air(&air);
        let mut degree = symbolic_builder.get_max_degree();
        if symbolic_builder.lookup_infos.len() != 0 {
            degree = degree.max(3)
        }
        degree = degree.max(min_degree);
        let logup_batch_size = degree - 1;
        let num_alpha = symbolic_builder.get_num_constraint(logup_batch_size);
        Self { air, symbolic_builder, degree, num_alpha }
    }
    #[inline]
    pub fn logup_batch_size(&self) -> usize {
        self.degree - 1
    }

    #[inline]
    pub fn required_max_beta_power(&self) -> usize {
        self.symbolic_builder.beta_powers.len()
    }

    #[inline]
    pub fn num_precompute(&self) -> usize {
        self.symbolic_builder.precomputed_lc_output.len()
    }

    pub fn reserved_poly(&self) -> &[PairCol] {
        &self.symbolic_builder.reserved_poly_output
    }

    pub fn num_lookup(&self) -> usize {
        self.symbolic_builder.lookup_infos.len()
    }

    pub fn perm_width(&self) -> usize {
        self.num_lookup().div_ceil(self.logup_batch_size())
    }
}

impl<A, F: Field + PolyAirExtendable<D>, const D: usize> BaseAir<F> for Chip<A, F, D>
where
    A: BaseAir<F>,
{
    fn width(&self) -> usize {
        self.air.width()
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        panic!("Chip should not use the `BaseAir` method, but the `MachineAir` method.")
    }
}

impl<A: MachineAir<F>, F: Field + PolyAirExtendable<D>, const D: usize> MachineAir<F>
    for Chip<A, F, D>
{
    type Record = A::Record;

    type Program = A::Program;

    fn name(&self) -> String {
        self.air.name()
    }

    fn preprocessed_width(&self) -> usize {
        <A as MachineAir<F>>::preprocessed_width(&self.air)
    }

    fn preprocessed_num_rows(&self, program: &Self::Program, instrs_len: usize) -> Option<usize> {
        <A as MachineAir<F>>::preprocessed_num_rows(&self.air, program, instrs_len)
    }

    fn generate_preprocessed_trace(&self, program: &A::Program) -> Option<CompressedMatrix<F>> {
        <A as MachineAir<F>>::generate_preprocessed_trace(&self.air, program)
    }

    fn num_rows(&self, input: &A::Record) -> Option<usize> {
        <A as MachineAir<F>>::num_rows(&self.air, input)
    }

    fn generate_trace(&self, input: &A::Record, output: &mut A::Record) -> CompressedMatrix<F> {
        <A as MachineAir<F>>::generate_trace(&self.air, input, output)
    }

    fn generate_dependencies(&self, input: &A::Record, output: &mut A::Record) {
        self.air.generate_dependencies(input, output);
    }

    fn included(&self, shard: &Self::Record) -> bool {
        self.air.included(shard)
    }

    fn commit_scope(&self) -> dt_stark::air::InteractionScope {
        self.air.commit_scope()
    }

    fn local_only(&self) -> bool {
        self.air.local_only()
    }

    fn global_boundary_owner(&self) -> Option<dt_stark::global_d11::StableChipId> {
        self.air.global_boundary_owner()
    }

    fn extract_global_claim(
        &self,
        trace: &CompressedMatrix<F>,
    ) -> Result<Option<dt_stark::air::GlobalClaim<F>>, String> {
        self.air.extract_global_claim(trace)
    }
}

pub struct SCStarkMachine<SC: SCStarkGenericConfig, A: MachineAir<Val<SC>>, const D: usize>
where
    Val<SC>: PolyAirExtendable<D>,
{
    pub config: SC,
    pub chips: Vec<Chip<A, Val<SC>, D>>,
    pub num_pv_elts: usize,
    pub contains_global_bus: bool,
    pub global_boundary_registry: BoundaryOwnerRegistryV2,
}

impl<SC: SCStarkGenericConfig, A: MachineAir<Val<SC>>, const D: usize> SCStarkMachine<SC, A, D>
where
    Val<SC>: PolyAirExtendable<D>,
{
    pub fn new(
        config: SC,
        chips: Vec<Chip<A, Val<SC>, D>>,
        num_pv_elts: usize,
        contains_global_bus: bool,
    ) -> Self {
        let owners = chips
            .iter()
            .filter_map(MachineAir::global_boundary_owner)
            .map(|owner| BoundaryOwnerV2 { owner, kind: GlobalBoundaryKindV2::Projective })
            .collect();
        let global_boundary_registry = BoundaryOwnerRegistryV2::new(owners)
            .expect("machine Global boundary owner registry must be canonical");
        Self { config, chips, num_pv_elts, contains_global_bus, global_boundary_registry }
    }
}

impl<SC: SCStarkGenericConfig, A: MachineAir<Val<SC>>, const D: usize> SCStarkMachine<SC, A, D>
where
    Val<SC>: PolyAirExtendable<D>,
{
    /// Returns an iterator over the chips in the machine that are included in the given shard.
    pub fn shard_chips_ordered<'a, 'b>(
        &'a self,
        chip_ordering: &'b HashMap<String, usize>,
    ) -> impl Iterator<Item = &'b Chip<A, Val<SC>, D>>
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
    pub fn chips(&self) -> &[Chip<A, Val<SC>, D>] {
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
    ) -> impl Iterator<Item = &'b Chip<A, Val<SC>, D>>
    where
        'a: 'b,
    {
        self.chips.iter().filter(|chip| chip.included(shard))
    }
}

impl<SC: SCStarkGenericConfig, A: MachineAir<Val<SC>>, const D: usize> SCStarkMachine<SC, A, D>
where
    Val<SC>: PolyAirExtendable<D>,
{
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
                        (
                            prep_trace.map(move |t| (chip.name(), chip.local_only(), t)),
                            (chip_name, chip.num_alpha),
                        )
                    })
                    .unzip()
            });

        let mut named_preprocessed_traces =
            named_preprocessed_traces.into_iter().flatten().collect::<Vec<_>>();

        // Order the chips and traces by trace size (biggest first), and get the ordering map.
        named_preprocessed_traces
            .sort_by_key(|(name, _, trace)| (Reverse(trace.height()), name.clone()));
        for (name, _, trace) in named_preprocessed_traces.iter() {
            tracing::info!(
                "{:<40}: width = {:?}, total_height = {}, stored_height = {}",
                name,
                trace.width(),
                trace.total_height,
                trace.stored_height()
            );
        }

        let pcs = self.config.mlpcs();
        let (chip_information, prep_traces): (Vec<_>, Vec<_>) = named_preprocessed_traces
            .iter()
            .map(|(name, _, c)| ((name.to_owned(), c.dimensions()), c))
            .unzip();

        let preprocessed_max_log_height = prep_traces
            .iter()
            .filter(|trace| trace.width() > 0)
            .map(|trace| p3_util::log2_strict_usize(trace.height()))
            .max();
        let preprocessed_stack_log_height =
            self.config.mlpcs_target_stack_log_height(preprocessed_max_log_height);
        let commit_options =
            self.config.mlpcs_commit_options_for_stack_log_height(preprocessed_stack_log_height);
        let (commit, data) = tracing::info_span!("commit to preprocessed traces")
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
                global146_identity: dt_stark::global_d11::GLOBAL146_COMPOSITE_IDENTITY,
                traces,
                data,
                chip_ordering: chip_ordering.clone(),
                local_only,
                constraints_map: constraints_map.clone(),
                preprocessed_pcs_stack_log_height: preprocessed_stack_log_height,
            },
            SCStarkVerifyingKey {
                commit,
                pc_start,
                program_boundary,
                owner_registry,
                global146_identity: dt_stark::global_d11::GLOBAL146_COMPOSITE_IDENTITY,
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
    pub fn verify(
        &self,
        vk: &SCStarkVerifyingKey<SC>,
        proof: &SCMachineProof<SC>,
        challenger: &mut SC::MlChallenger,
        num_skip_rounds: usize,
        chip_log_height_threshold: usize,
    ) -> Result<(), MachineVerificationError<SC>>
    where
        SC::MlChallenger: Clone,
        Val<SC>: PrimeField32,
        A: for<'a> FullAir<
            crate::precompute::PrecomputeRowBuilder<'a, Val<SC>, Challenge<SC>, Challenge<SC>>,
        >,
        A: for<'a> FullAir<
            crate::verifier::SumcheckVerifierConstraintFolder<'a, Val<SC>, Challenge<SC>>,
        >,
    {
        tracing::info!("verify with univariate skip parameter k={}", num_skip_rounds);

        if self.global_boundary_registry != vk.owner_registry ||
            vk.owner_registry.validate().is_err()
        {
            return Err(MachineVerificationError::InvalidVerificationKey);
        }

        let phase_observe = Instant::now();
        vk.observe_into(challenger);
        pcs::whir::profile::add_ms("verify.vk_observe_us", phase_observe.elapsed().as_micros());

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
                    crate::verifier::Verifier::<SC, A, D>::verify_shard(
                        &self.config,
                        vk,
                        &chips,
                        &mut shard_challenger,
                        shard_proof,
                        num_skip_rounds,
                        chip_log_height_threshold,
                        self.contains_global_bus,
                    )
                    .map_err(MachineVerificationError::InvalidShardProofSumcheck)
                })?;
            }

            Ok(())
        })?;

        if !self.contains_global_bus {
            return Ok(());
        }
        let claims = proof
            .shard_proofs
            .iter()
            .map(|shard| {
                let pv: &dt_stark::air::PublicValues<dt_stark::Word<Val<SC>>, Val<SC>> =
                    shard.public_values.as_slice().borrow();
                pv.global
            })
            .collect::<Vec<_>>();
        verify_global_interval_root_v4(&vk.program_boundary, &claims).map_err(|_| {
            MachineVerificationError::InvalidPublicValues("invalid Global interval chain")
        })
    }
}
