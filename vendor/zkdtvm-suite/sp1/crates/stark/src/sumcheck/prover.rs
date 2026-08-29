#![allow(missing_docs)]
#![allow(clippy::type_complexity)]

use core::fmt::Display;
use std::{borrow::Borrow, cmp::Reverse, error::Error, time::Instant};

use hashbrown::HashMap;
use itertools::any;
use p3_air::{Air, BaseAir};
use p3_challenger::{CanObserve, FieldChallenger};
use p3_field::{AbstractExtensionField, AbstractField, ExtensionField, Field, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Dimensions, Matrix};
use p3_maybe_rayon::prelude::*;
use p3_uni_stark::SymbolicAirBuilder;
use p3_util::log2_strict_usize;
use pcs::basefold::mlpcs::MlPCS;
use serde::{de::DeserializeOwned, Serialize};

use crate::{
    air::{derive_active_shape_v1, observe_active_shape_v1, MachineAir, PublicValues},
    chip::Chip,
    global_d11::validate_global_claim,
    lookup::InteractionBuilder,
    opts::DTCoreOpts,
    record::MachineRecord,
    sumcheck::{
        config::{MlChallenger, MlCom, MlPcsOpeningProof, MlPcsProverData, SCStarkGenericConfig},
        core::SumcheckProtocol,
        folder::{
            PaddingRowConstraintFolder, SumcheckConstraintFolder, SumcheckConstraintFolderExt,
        },
        keys::{SCMachineProvingKey, SCStarkProvingKey, SCStarkVerifyingKey},
        proof::{
            SCChipOpenedValues, SCMachineProof, SCShardCommitment, SCShardMainData,
            SCShardOpenedValues, SCShardProof, SumcheckProof,
        },
        state::ChipState,
        trace::CompressedMatrix,
        use_algebraic_decomp as configured_use_algebraic_decomp,
        utils::{compute_num_chips_each_round, compute_powers_of_alpha, extend_pv},
    },
    Challenge, DebugConstraintBuilder, MachineChip, SCStarkMachine, SumcheckChip, Val,
};

type SumcheckRunOutput<SC> =
    (Vec<Challenge<SC>>, Vec<SCChipOpenedValues<Val<SC>, Challenge<SC>>>, SumcheckProof<SC>);

/// A sumcheck prover implementation based on x86 and ARM CPUs.
pub struct SumcheckProver<SC: SCStarkGenericConfig, A, AE> {
    pub machine: SCStarkMachine<SC, A, AE>,
}

/// An error that occurs during the execution of the [`SumcheckProver`].
#[derive(Debug, Clone, Copy)]
pub struct SumcheckProverError;

impl Display for SumcheckProverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DefaultProverError")
    }
}

impl Error for SumcheckProverError {}

#[allow(clippy::too_many_arguments)]
fn prove_sumcheck_with_decomp<'a, SC, A, AE, const USE_ALGEBRAIC_DECOMP: bool>(
    eq_challenges: Vec<Challenge<SC>>,
    bit_expand_poly_points: Vec<Val<SC>>,
    chip_states: Vec<ChipState<'a, SC, A, AE>>,
    num_rounds: usize,
    num_rounds_linear: usize,
    num_skip_rounds: usize,
    chip_log_height_threshold: usize,
    permutation_challenges: [Challenge<SC>; 2],
    public_values: &'a [Val<SC>],
    public_values_ext: Vec<Challenge<SC>>,
    num_chips_each_round: Vec<usize>,
    max_height: usize,
    challenger: &mut MlChallenger<SC>,
) -> SumcheckRunOutput<SC>
where
    SC: SCStarkGenericConfig,
    Val<SC>: Field,
    Challenge<SC>: ExtensionField<Val<SC>>,
    A: MachineAir<Val<SC>> + for<'b> Air<SumcheckConstraintFolder<'b, SC>>,
    AE: MachineAir<Challenge<SC>> + for<'b> Air<SumcheckConstraintFolderExt<'b, SC>>,
{
    let mut sumcheck_protocol = SumcheckProtocol::<'_, SC, A, AE, USE_ALGEBRAIC_DECOMP>::new(
        eq_challenges,
        bit_expand_poly_points,
        chip_states,
        num_rounds,
        num_rounds_linear,
        num_skip_rounds,
        chip_log_height_threshold,
        permutation_challenges,
        public_values,
        public_values_ext,
        num_chips_each_round,
    );
    sumcheck_protocol.prove(challenger);

    tracing::trace!("PROVER: final claim = {:?}", sumcheck_protocol.state.claim);
    for (i, cs) in sumcheck_protocol.state.chip_states.iter().enumerate() {
        tracing::trace!("  PROVER chip[{}]: claim={:?}, perm_claim={:?}, num_constraints={}, perm_last_alpha={:?}",
            i, cs.claim, cs.perm_claim, cs.num_constraints,
            cs.powers_of_alpha.last());
    }

    let sumcheck_challenges = sumcheck_protocol.state.sumcheck_challenges.clone();
    let bit_expand_poly = &sumcheck_protocol.state.bit_expand_poly;
    let mut extended_sumcheck_challenges = Vec::with_capacity(max_height);
    extended_sumcheck_challenges
        .extend(sumcheck_challenges.iter().take(num_rounds_linear).copied());
    for challenge in &sumcheck_challenges[num_rounds_linear..] {
        extended_sumcheck_challenges.extend(bit_expand_poly.evals_all(*challenge));
    }
    let opened_values = sumcheck_protocol.state.finalize();
    let sumcheck_proof = SumcheckProof { unipolys: sumcheck_protocol.unipolys };

    (extended_sumcheck_challenges, opened_values, sumcheck_proof)
}

/// Trait for sumcheck-based machine provers.
///
/// Defines the interface for proving STARK constraints via the sumcheck protocol,
/// including trace generation, commitment, opening, and full proof generation.
pub trait SCMachineProver<
    SC: SCStarkGenericConfig,
    A: MachineAir<SC::Val>,
    AE: MachineAir<Challenge<SC>>,
>: 'static + Send + Sync
{
    /// The matrix type used to store traces on the proving device.
    type DeviceMatrix: Matrix<SC::Val>;

    /// The compressed matrix type used to store traces(commit 出,open
    /// 用)。GPU=DeviceCompressedMatrixConcrete,host=CompressedMatrix。
    type DeviceCompressedMatrix;

    /// The PCS prover data type produced during commitment.
    type DeviceProverData;

    /// The proving key type for this prover.
    type DeviceProvingKey: SCMachineProvingKey<SC>;

    /// The error type returned by fallible operations.
    type Error: Error + Send + Sync;

    /// Creates a new prover from the given machine.
    fn new(machine: SCStarkMachine<SC, A, AE>) -> Self;

    /// Returns a reference to the underlying machine.
    fn machine(&self) -> &SCStarkMachine<SC, A, AE>;

    /// Runs the setup phase: generates proving and verifying keys from the program.
    fn setup(&self, program: &A::Program) -> (Self::DeviceProvingKey, SCStarkVerifyingKey<SC>);

    /// Copy the proving key from the host to the device.
    fn pk_to_device(&self, pk: &SCStarkProvingKey<SC>) -> Self::DeviceProvingKey;

    /// Copy the proving key from the device to the host.
    fn pk_to_host(&self, pk: &Self::DeviceProvingKey) -> SCStarkProvingKey<SC>;

    /// Generates main traces for all chips included in the given record (as compressed matrices).
    ///
    /// Returns a list of `(chip_name, compressed_trace)` pairs, produced in parallel.
    fn generate_traces(&self, record: &A::Record) -> Vec<(String, CompressedMatrix<Val<SC>>)> {
        self.generate_traces_filtered(record, None)
    }

    /// Generates main traces for the included chips selected by `chips_filter`.
    fn generate_traces_filtered(
        &self,
        record: &A::Record,
        chips_filter: Option<&[String]>,
    ) -> Vec<(String, CompressedMatrix<Val<SC>>)> {
        let shard_chips = self
            .shard_chips(record)
            .filter(|chip| {
                chips_filter.is_none_or(|filter| filter.contains(&chip.name()))
            })
            .collect::<Vec<_>>();

        let parent_span = tracing::debug_span!("generate traces for shard");
        parent_span.in_scope(|| {
            shard_chips
                .par_iter()
                .map(|chip| {
                    let chip_name = chip.name();
                    let begin = Instant::now();
                    let trace = chip.generate_trace(record, &mut A::Record::default());
                    tracing::debug!(
                        parent: &parent_span,
                        "generated trace for chip {} in {:?}",
                        chip_name,
                        begin.elapsed()
                    );
                    (chip_name, trace)
                })
                .collect::<Vec<_>>()
        })
    }

    /// Commits to the compressed main traces.
    ///
    /// Decompresses traces for PCS commitment while keeping compressed forms for sumcheck.
    fn commit(
        &self,
        record: &A::Record,
        compressed_traces: Vec<(String, CompressedMatrix<Val<SC>>)>,
    ) -> SCShardMainData<SC, Self::DeviceCompressedMatrix, Self::DeviceProverData>;

    /// Commits to the compressed main traces while matching a setup-time preprocessed PCS stack
    /// height when one is available.
    fn commit_with_pcs_stack_log_height(
        &self,
        record: &A::Record,
        compressed_traces: Vec<(String, CompressedMatrix<Val<SC>>)>,
        preprocessed_pcs_stack_log_height: Option<usize>,
    ) -> SCShardMainData<SC, Self::DeviceCompressedMatrix, Self::DeviceProverData> {
        let _ = preprocessed_pcs_stack_log_height;
        self.commit(record, compressed_traces)
    }

    /// Computes the sumcheck opening proof for a single shard.
    ///
    /// Runs the sumcheck protocol over compressed matrices, then produces a PCS opening proof.
    fn open(
        &self,
        pk: &Self::DeviceProvingKey,
        data: SCShardMainData<SC, Self::DeviceCompressedMatrix, Self::DeviceProverData>,
        challenger: &mut MlChallenger<SC>,
        num_skip_rounds: usize,
        chip_log_height_threshold: usize,
    ) -> Result<SCShardProof<SC>, Self::Error>;

    /// Proves all shards end-to-end: generates traces, commits, and opens each shard.
    fn prove(
        &self,
        pk: &Self::DeviceProvingKey,
        records: Vec<A::Record>,
        challenger: &mut SC::MlChallenger,
        opts: <A::Record as MachineRecord>::Config,
        num_skip_rounds: usize,
        chip_log_height_threshold: usize,
    ) -> Result<SCMachineProof<SC>, Self::Error>
    where
        A: for<'a> Air<DebugConstraintBuilder<'a, Val<SC>, Challenge<SC>>>;

    /// Returns the STARK configuration for the machine.
    fn config(&self) -> &SC {
        self.machine().config()
    }

    /// Returns the number of public values elements.
    fn num_pv_elts(&self) -> usize {
        self.machine().num_pv_elts()
    }

    /// Returns an iterator over the base-field chips included in the given record.
    fn shard_chips<'a, 'b>(
        &'a self,
        record: &'b A::Record,
    ) -> impl Iterator<Item = &'b MachineChip<SC, A>>
    where
        'a: 'b,
        SC: 'b,
    {
        self.machine().shard_chips(record)
    }

    /// Returns an iterator over the extension-field chips included in the given record.
    fn shard_chips_ext<'a, 'b>(
        &'a self,
        record: &'b AE::Record,
    ) -> impl Iterator<Item = &'b SumcheckChip<SC, AE>>
    where
        'a: 'b,
        SC: 'b,
    {
        self.machine().shard_chips_ext(record)
    }
}

impl<SC, A, AE> SCMachineProver<SC, A, AE> for SumcheckProver<SC, A, AE>
where
    SC: 'static + SCStarkGenericConfig + Send + Sync,
    A: MachineAir<SC::Val>
        + Air<InteractionBuilder<Val<SC>>>
        + Air<SymbolicAirBuilder<Val<SC>>>
        + for<'a> Air<SumcheckConstraintFolder<'a, SC>>
        + for<'a> Air<PaddingRowConstraintFolder<'a, SC>>,
    AE: MachineAir<Challenge<SC>>
        + Air<InteractionBuilder<Challenge<SC>>>
        + Air<SymbolicAirBuilder<Challenge<SC>>>
        + for<'a> Air<SumcheckConstraintFolderExt<'a, SC>>,
    A::Record: MachineRecord<Config = DTCoreOpts>,
    SC::Val: PrimeField32,
    MlCom<SC>: Send + Sync,
    MlPcsProverData<SC>: Send + Sync + Serialize + DeserializeOwned,
    MlPcsOpeningProof<SC>: Send + Sync,
    SC::MlChallenger: Clone,
{
    type DeviceMatrix = RowMajorMatrix<SC::Val>;
    type DeviceCompressedMatrix = CompressedMatrix<Val<SC>, Val<SC>>;
    type DeviceProverData = MlPcsProverData<SC>;
    type DeviceProvingKey = SCStarkProvingKey<SC>;
    type Error = SumcheckProverError;

    fn new(machine: SCStarkMachine<SC, A, AE>) -> Self {
        Self { machine }
    }

    fn machine(&self) -> &SCStarkMachine<SC, A, AE> {
        &self.machine
    }

    fn setup(&self, program: &A::Program) -> (Self::DeviceProvingKey, SCStarkVerifyingKey<SC>) {
        self.machine().setup(program)
    }

    fn pk_to_device(&self, pk: &SCStarkProvingKey<SC>) -> Self::DeviceProvingKey {
        pk.clone()
    }

    fn pk_to_host(&self, pk: &Self::DeviceProvingKey) -> SCStarkProvingKey<SC> {
        pk.clone()
    }

    fn commit(
        &self,
        record: &A::Record,
        compressed_traces: Vec<(String, CompressedMatrix<Val<SC>>)>,
    ) -> SCShardMainData<SC, Self::DeviceCompressedMatrix, Self::DeviceProverData> {
        self.commit_with_pcs_stack_log_height(record, compressed_traces, None)
    }

    fn commit_with_pcs_stack_log_height(
        &self,
        record: &A::Record,
        mut compressed_traces: Vec<(String, CompressedMatrix<Val<SC>>)>,
        preprocessed_pcs_stack_log_height: Option<usize>,
    ) -> SCShardMainData<SC, Self::DeviceCompressedMatrix, Self::DeviceProverData> {
        // Sort traces by height (largest first), then by name for deterministic ordering.
        compressed_traces.sort_by_key(|(name, trace)| (Reverse(trace.total_height), name.clone()));
        let chip_ordering =
            compressed_traces.iter().enumerate().map(|(i, (name, _))| (name.clone(), i)).collect();

        let public_values_vec = record.public_values();
        let chips = self.machine().shard_chips_ordered(&chip_ordering).collect::<Vec<_>>();
        assert_eq!(chips.len(), compressed_traces.len(), "trace/chip inventory mismatch");
        if self.machine().contains_global_bus() {
            let public_values: &PublicValues<crate::Word<Val<SC>>, Val<SC>> =
                public_values_vec.as_slice().borrow();
            let mut derived = None;
            for (chip, (_, trace)) in chips.iter().zip(compressed_traces.iter()) {
                let extracted = chip
                    .extract_global_claim(trace)
                    .expect("canonical Global claim extraction failed before commitment");
                match chip.global_boundary_owner() {
                    Some(_) => {
                        let extracted =
                            extracted.expect("registered Global owner produced no claim");
                        assert!(
                            derived.replace(extracted).is_none(),
                            "duplicate Global boundary owner"
                        );
                    }
                    None => {
                        assert!(extracted.is_none(), "unregistered chip produced a Global claim");
                    }
                }
            }
            validate_global_claim(&public_values.global, derived.is_some())
                .expect("honest Global claim admission failed before commitment");
            if let Some(extracted) = derived {
                assert_eq!(
                    public_values.global, extracted,
                    "Global public claim differs from trace boundary before commitment"
                );
            }
        }

        let traces_refs: Vec<&CompressedMatrix<Val<SC>>> =
            compressed_traces.iter().map(|(_, trace)| trace).collect();

        let pcs = self.config().mlpcs();
        let batch_max_log_height = traces_refs
            .iter()
            .filter(|trace| trace.width() > 0)
            .map(|trace| log2_strict_usize(trace.height()))
            .max();
        if let (Some(setup_height), Some(batch_height)) =
            (preprocessed_pcs_stack_log_height, batch_max_log_height)
        {
            if batch_height > setup_height {
                tracing::warn!(
                    "main trace log height {} exceeds preprocessed PCS stack log height {}; \
                     using main trace height for this commit batch.",
                    batch_height,
                    setup_height
                );
            }
        }
        let effective_max_log_height =
            match (batch_max_log_height, preprocessed_pcs_stack_log_height) {
                (Some(batch_height), Some(setup_height)) => Some(batch_height.max(setup_height)),
                (Some(batch_height), None) => Some(batch_height),
                (None, setup_height) => setup_height,
            };
        let pcs_stack_log_height =
            self.config().mlpcs_target_stack_log_height(effective_max_log_height);
        let commit_options =
            self.config().mlpcs_commit_options_for_stack_log_height(pcs_stack_log_height);
        let (main_commit, main_data) = tracing::info_span!("commit to main traces")
            .in_scope(|| pcs.commit_with_options(traces_refs, commit_options));

        SCShardMainData {
            compressed_traces,
            main_commit,
            main_data,
            pcs_stack_log_height,
            chip_ordering,
            public_values: public_values_vec,
        }
    }

    #[allow(clippy::too_many_lines)]
    fn open(
        &self,
        pk: &SCStarkProvingKey<SC>,
        data: SCShardMainData<SC, Self::DeviceCompressedMatrix, Self::DeviceProverData>,
        challenger: &mut MlChallenger<SC>,
        num_skip_rounds: usize,
        chip_log_height_threshold: usize,
    ) -> Result<SCShardProof<SC>, Self::Error> {
        assert!(chip_log_height_threshold.is_multiple_of(num_skip_rounds));
        tracing::info!(
            "proving with generalized skip k={}, threshold={}",
            num_skip_rounds,
            chip_log_height_threshold
        );

        let chips: Vec<&Chip<Val<SC>, A>> =
            self.machine().shard_chips_ordered(&data.chip_ordering).collect::<Vec<_>>();
        let chips_ext: Vec<&Chip<Challenge<SC>, AE>> =
            self.machine().shard_chips_ext_ordered(&data.chip_ordering).collect::<Vec<_>>();

        let chip_names: Vec<String> = chips.iter().map(|c| c.name()).collect();
        let preprocessed_compressed = pk.get_preprocessed_compressed_for_chips(&chip_names);

        let config = self.machine().config();
        let log_heights: Vec<usize> =
            data.compressed_traces.iter().map(|(_, t)| log2_strict_usize(t.total_height)).collect();
        let pcs = config.mlpcs();

        challenger.observe_slice(&data.public_values[0..self.num_pv_elts()]);
        challenger.observe(data.main_commit.clone());
        let active_shape = derive_active_shape_v1(
            chips
                .iter()
                .zip(log_heights.iter())
                .map(|(chip, &log_height)| (chip.name(), chip.width(), log_height)),
        )
        .expect("honest active shape must be canonical");
        observe_active_shape_v1::<Val<SC>, _>(challenger, &active_shape);

        let local_permutation_challenges: Vec<Challenge<SC>> =
            (0..2).map(|_| challenger.sample_ext_element()).collect();
        let permutation_challenges: [Challenge<SC>; 2] =
            local_permutation_challenges.as_slice().try_into().unwrap();

        // Generate compressed permutation traces and local lookup imbalances in parallel. The
        // proof-native Global interval is authenticated by the owner trace and public claim.
        let (permutation_compressed, local_cumulative_sums): (
            Vec<CompressedMatrix<Challenge<SC>, Challenge<SC>>>,
            Vec<_>,
        ) = tracing::info_span!("generate permutation traces").in_scope(|| {
            (0..chips.len())
                .into_par_iter()
                .map(|i| {
                    let chip = chips[i];
                    let main_compressed = &data.compressed_traces[i].1;
                    let prep = preprocessed_compressed[i];
                    let (perm_c, local_sum) = chip.generate_compressed_permutation_trace(
                        prep,
                        main_compressed,
                        &local_permutation_challenges,
                    );
                    (perm_c, local_sum)
                })
                .unzip()
        });

        // Log per-chip trace dimensions and cumulative sums.
        for i in 0..chips.len() {
            let trace_width = data.compressed_traces[i].1.width();
            let log_trace_height = log_heights[i];
            let stored_height = data.compressed_traces[i].1.stored_height();
            let prep_width = preprocessed_compressed[i].map_or(0, |c| c.main.width());
            let permutation_width = permutation_compressed[i].main.width();
            tracing::info!(
                "{:<15} | Main Cols = {:<5} | Pre Cols = {:<5}  | Perm Cols = {:<5} | Rows = {:<6} (padded 2^{})",
                chips[i].name(),
                trace_width,
                prep_width,
                permutation_width,
                stored_height,
                log_trace_height,
            );
        }
        if std::env::var("DEBUG_CUM_SUM").is_ok() {
            use p3_field::Field;
            let total_local: Challenge<SC> = local_cumulative_sums.iter().copied().sum();
            eprintln!("[CUM_SUM] total_local = {total_local:?}");
            for i in 0..chips.len() {
                if !Field::is_zero(&local_cumulative_sums[i]) {
                    eprintln!(
                        "[CUM_SUM] chip[{}] {} local_sum = {:?}",
                        i,
                        chips[i].name(),
                        local_cumulative_sums[i]
                    );
                }
            }
        }

        let mut dimensions: Vec<Vec<Dimensions>> = Vec::new();
        let prep_dims: Vec<Dimensions> = (0..chips.len())
            .filter_map(|i| {
                let width = preprocessed_compressed[i].map_or(0, |c| c.main.width());
                if width > 0 {
                    Some(Dimensions { width, height: data.compressed_traces[i].1.height() })
                } else {
                    None
                }
            })
            .collect();
        let main_dims = (0..chips.len())
            .map(|i| Dimensions {
                width: data.compressed_traces[i].1.width(),
                height: data.compressed_traces[i].1.height(),
            })
            .collect::<Vec<_>>();
        dimensions.push(prep_dims);
        dimensions.push(main_dims);
        let permutation_traces_base: Vec<CompressedMatrix<Val<SC>>> = permutation_compressed
            .par_iter()
            .map(|c| {
                CompressedMatrix::from_full_matrix_no_padding(c.decompress().flatten_to_base())
            })
            .collect::<Vec<_>>();
        if any(&permutation_traces_base, |trace| trace.width() > 0) {
            let permutation_dims = (0..chips.len())
                .map(|i| Dimensions {
                    width: permutation_traces_base[i].width(),
                    height: data.compressed_traces[i].1.height(),
                })
                .collect::<Vec<_>>();
            dimensions.push(permutation_dims);
        }

        // Commit to permutation traces (if any chip has permutation columns).
        let mut permutation_commit_and_data = if any(&permutation_traces_base, |trace| {
            trace.width() > 0
        }) {
            let (permutation_commit, permutation_data) =
                tracing::info_span!("commit to permutation traces").in_scope(|| {
                    pcs.commit_with_options(
                        permutation_traces_base.iter().collect(),
                        self.config()
                            .mlpcs_commit_options_for_stack_log_height(data.pcs_stack_log_height),
                    )
                });
            challenger.observe(permutation_commit.clone());
            Some((permutation_commit, permutation_data))
        } else {
            None
        };

        for local_sum in &local_cumulative_sums {
            challenger.observe_slice(
                <Challenge<SC> as AbstractExtensionField<Val<SC>>>::as_base_slice(local_sum),
            );
        }

        // Sample the constraint-batching challenge alpha.
        let alpha: Challenge<SC> = challenger.sample_ext_element::<Challenge<SC>>();

        // Compute round structure: linear rounds (degree-1) + nonlinear rounds (skipped).
        let max_height = *log_heights.iter().max().unwrap();
        let num_rounds_linear = max_height.saturating_sub(chip_log_height_threshold);
        let num_rounds_nonlinear =
            std::cmp::min(max_height, chip_log_height_threshold) / num_skip_rounds;
        let num_rounds = num_rounds_linear + num_rounds_nonlinear;

        // Sample eq-polynomial challenges for the sumcheck protocol.
        let eq_challenges: Vec<Challenge<SC>> =
            (0..num_rounds).map(|_| challenger.sample_ext_element()).collect();
        let num_constraints: Vec<usize> = chips
            .iter()
            .map(|chip| {
                *HashMap::<String, usize>::get(&pk.constraints_map, &chip.name())
                    .expect("chip not found in constraints map")
            })
            .collect();
        let num_chips_each_round =
            compute_num_chips_each_round(&log_heights, num_skip_rounds, chip_log_height_threshold);
        let powers_of_alpha = compute_powers_of_alpha(alpha, num_constraints.clone());
        let public_values_ext = extend_pv(&data.public_values);

        let round_introduced: Vec<usize> = (0..chips.len())
            .map(|i| {
                (0..num_rounds)
                    .position(|r| num_chips_each_round[r] > i)
                    .unwrap_or(num_rounds.saturating_sub(1))
            })
            .collect();

        let chip_states: Vec<ChipState<'_, SC, A, AE>> = (0..chips.len())
            .map(|i| {
                ChipState::new(
                    i,
                    log_heights[i],
                    chips[i],
                    chips_ext[i],
                    preprocessed_compressed[i],
                    &data.compressed_traces[i].1,
                    &permutation_compressed[i],
                    local_cumulative_sums[i],
                    powers_of_alpha[i].clone(),
                    num_constraints[i],
                    round_introduced[i],
                    &permutation_challenges,
                    &data.public_values,
                )
            })
            .collect();

        let bit_expand_poly_points: Vec<Val<SC>> =
            (0..(1 << num_skip_rounds)).map(Val::<SC>::from_canonical_usize).collect();

        let use_algebraic_decomp = configured_use_algebraic_decomp();
        tracing::info!("sumcheck algebraic_decomp={}", use_algebraic_decomp);
        let (extended_sumcheck_challenges, opened_values, sumcheck_proof) = if use_algebraic_decomp
        {
            prove_sumcheck_with_decomp::<SC, A, AE, true>(
                eq_challenges,
                bit_expand_poly_points,
                chip_states,
                num_rounds,
                num_rounds_linear,
                num_skip_rounds,
                chip_log_height_threshold,
                permutation_challenges,
                &data.public_values,
                public_values_ext,
                num_chips_each_round,
                max_height,
                challenger,
            )
        } else {
            prove_sumcheck_with_decomp::<SC, A, AE, false>(
                eq_challenges,
                bit_expand_poly_points,
                chip_states,
                num_rounds,
                num_rounds_linear,
                num_skip_rounds,
                chip_log_height_threshold,
                permutation_challenges,
                &data.public_values,
                public_values_ext,
                num_chips_each_round,
                max_height,
                challenger,
            )
        };

        // Build the opening point from the extended sumcheck challenges (max height).
        let mut opening_point: Vec<Challenge<SC>> = extended_sumcheck_challenges.clone();
        opening_point.reverse();

        let preprocessed_traces_for_open = pk.get_preprocessed_traces_for_open(&chip_names);

        // Decompress main traces for PCS batch opening (deferred to after sumcheck).
        let _main_traces_full: Vec<RowMajorMatrix<Val<SC>>> =
            data.compressed_traces.iter().map(|(_, c)| c.decompress()).collect();

        // Produce the PCS batch opening proof.
        tracing::info_span!("batch open").in_scope(|| {
            // Build opened_values: Vec<Vec<Vec<EF>>> — one entry per trace group (batch),
            // each containing one Vec<EF> per matrix (chip).
            let prep_opened_values: Vec<Vec<Challenge<SC>>> = opened_values
                .iter()
                .filter(|chip| !chip.preprocessed.local.is_empty())
                .map(|chip| chip.preprocessed.to_vec_values())
                .collect();
            let main_opened_values: Vec<Vec<Challenge<SC>>> =
                opened_values.iter().map(|chip| chip.main.to_vec_values()).collect();

            let main_traces_compressed: Vec<CompressedMatrix<Val<SC>>> =
                data.compressed_traces.into_iter().map(|(_, c)| c).collect();

            // Path with permutation: 3 trace groups (preprocessed, main, permutation).
            if let Some((permutation_commit, permutation_data)) = permutation_commit_and_data.take()
            {
                let permutation_opened_values: Vec<Vec<Challenge<SC>>> =
                    opened_values.iter().map(|chip| chip.permutation.to_vec_values()).collect();
                let pcs_opened_values =
                    vec![prep_opened_values, main_opened_values, permutation_opened_values];
                let opening_proof = pcs
                    .open(
                        vec![
                            preprocessed_traces_for_open.into_iter().flatten().collect(),
                            main_traces_compressed,
                            permutation_traces_base,
                        ],
                        vec![pk.data.clone(), data.main_data, permutation_data],
                        &opening_point,
                        &pcs_opened_values,
                        challenger,
                    )
                    .expect("opening proof failed");
                Ok(SCShardProof::<SC> {
                    commitment: SCShardCommitment {
                        main_commit: data.main_commit,
                        permutation_commit: Some(permutation_commit),
                    },
                    opened_values: SCShardOpenedValues {
                        chips: opened_values.clone(),
                        _field: core::marker::PhantomData,
                    },
                    opening_proof,
                    sumcheck_proof,
                    dimensions,
                    chip_ordering: data.chip_ordering,
                    public_values: data.public_values,
                })
            } else {
                // Path without permutation: 2 trace groups only (preprocessed, main).
                let pcs_opened_values = vec![prep_opened_values, main_opened_values];
                let opening_proof = pcs
                    .open(
                        vec![
                            preprocessed_traces_for_open.into_iter().flatten().collect(),
                            main_traces_compressed,
                        ],
                        vec![pk.data.clone(), data.main_data],
                        &opening_point,
                        &pcs_opened_values,
                        challenger,
                    )
                    .expect("opening proof failed");
                Ok(SCShardProof::<SC> {
                    commitment: SCShardCommitment {
                        main_commit: data.main_commit,
                        permutation_commit: None,
                    },
                    opened_values: SCShardOpenedValues {
                        chips: opened_values,
                        _field: core::marker::PhantomData,
                    },
                    opening_proof,
                    sumcheck_proof,
                    dimensions,
                    chip_ordering: data.chip_ordering,
                    public_values: data.public_values,
                })
            }
        })
    }

    #[allow(clippy::needless_for_each)]
    fn prove(
        &self,
        pk: &SCStarkProvingKey<SC>,
        mut records: Vec<A::Record>,
        challenger: &mut SC::MlChallenger,
        opts: <A::Record as MachineRecord>::Config,
        num_skip_rounds: usize,
        chip_log_height_threshold: usize,
    ) -> Result<SCMachineProof<SC>, Self::Error>
    where
        A: for<'a> Air<DebugConstraintBuilder<'a, Val<SC>, Challenge<SC>>>,
    {
        tracing::info!(
            "proving with generalized skip k={}, threshold={}",
            num_skip_rounds,
            chip_log_height_threshold
        );

        // Generate cross-chip dependencies (e.g. memory, lookup interactions).
        self.machine().generate_dependencies(&mut records, &opts, None);

        // Observe the preprocessed commitment into the challenger.
        pk.observe_into(challenger);

        let shard_proofs = tracing::info_span!("prove_shards").in_scope(|| {
            records
                .into_par_iter()
                .map(|record| {
                    let compressed_traces = self.generate_traces(&record);
                    let shard_data_v2 = self.commit_with_pcs_stack_log_height(
                        &record,
                        compressed_traces,
                        pk.preprocessed_pcs_stack_log_height,
                    );
                    self.open(
                        pk,
                        shard_data_v2,
                        &mut challenger.clone(),
                        num_skip_rounds,
                        chip_log_height_threshold,
                    )
                })
                .collect::<Result<Vec<_>, _>>()
        })?;

        Ok(SCMachineProof { shard_proofs })
    }
}
