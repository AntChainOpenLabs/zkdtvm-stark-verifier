use std::{
    borrow::Borrow,
    cmp::Reverse,
    error::Error,
    fmt::Display,
    sync::{
        atomic::{AtomicUsize, Ordering},
        OnceLock,
    },
    time::Instant,
};

use crate::{
    core::SumcheckProtocol,
    evaluator::ConstraintFolder,
    permutation::{fused_precompute_reserved_permutation, PermutationRowBuilder},
    precompute::PrecomputeRowBuilder,
    state::{compute_eq_poly_coeffs, finalize, ChipState},
    Chip, SCStarkMachine,
};
use dt_stark::{
    air::{
        derive_active_shape_v1, observe_active_shape_v1, FullAir, MachineAir, PolyAirExtendable,
        PublicValues,
    },
    global_d11::validate_global_claim,
    septic_curve_params::compute_beta_septix,
    sumcheck::{
        config::{MlChallenger, MlCom, MlPcsOpeningProof, MlPcsProverData, SCStarkGenericConfig},
        keys::{SCMachineProvingKey, SCStarkProvingKey, SCStarkVerifyingKey},
        proof::{
            SCMachineProof, SCShardCommitment, SCShardMainData, SCShardOpenedValues, SCShardProof,
            SumcheckProof,
        },
        use_algebraic_decomp as configured_use_algebraic_decomp,
        utils::{compute_num_chips_each_round, compute_powers_of_alpha},
    },
    Challenge, MachineRecord, Val, Word,
};
use hashbrown::HashMap;
use itertools::any;
use p3_air::BaseAir;
use p3_challenger::{CanObserve, FieldChallenger};
use p3_field::{AbstractExtensionField, AbstractField, PrimeField32};
use p3_matrix::{compressed::CompressedMatrix, dense::RowMajorMatrix, Dimensions, Matrix};
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;
use pcs::{basefold::mlpcs::MlPCS, whir::profile as whir_profile};
use serde::{de::DeserializeOwned, Serialize};

type SumcheckRunOutput<SC> = (Vec<Challenge<SC>>, SumcheckProof<SC>);

pub struct SumcheckProver<SC: SCStarkGenericConfig, A: MachineAir<Val<SC>>, const D: usize>
where
    Val<SC>: PolyAirExtendable<D>,
{
    pub machine: SCStarkMachine<SC, A, D>,
}

const PK_DATA_CLONE_AUDIT_ENV: &str = "DT_PK_DATA_CLONE_AUDIT";
static PK_DATA_CLONE_AUDIT_SEQ: AtomicUsize = AtomicUsize::new(0);

fn pk_data_clone_audit_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var(PK_DATA_CLONE_AUDIT_ENV) {
        Ok(value) => value != "0" && !value.eq_ignore_ascii_case("false"),
        Err(_) => false,
    })
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1_000.0
}

fn clone_pk_data_for_open<SC>(pk: &SCStarkProvingKey<SC>, site: &str) -> MlPcsProverData<SC>
where
    SC: SCStarkGenericConfig,
    MlPcsProverData<SC>: Clone + Serialize,
{
    if !pk_data_clone_audit_enabled() {
        return pk.data.clone();
    }

    let size_start = Instant::now();
    let serialized_bytes = bincode::serialized_size(&pk.data).ok();
    let size_ms = elapsed_ms(size_start);

    let clone_start = Instant::now();
    let cloned = pk.data.clone();
    let clone_ms = elapsed_ms(clone_start);

    let seq = PK_DATA_CLONE_AUDIT_SEQ.fetch_add(1, Ordering::Relaxed);
    match serialized_bytes {
        Some(bytes) => eprintln!(
            "pk_data_clone_audit seq={seq} site={site} serialized_bytes={bytes} size_ms={size_ms:.3} clone_ms={clone_ms:.3}"
        ),
        None => eprintln!(
            "pk_data_clone_audit seq={seq} site={site} serialized_bytes=error size_ms={size_ms:.3} clone_ms={clone_ms:.3}"
        ),
    }

    cloned
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
fn prove_sumcheck_with_decomp<'a, SC, A, const D: usize, const USE_ALGEBRAIC_DECOMP: bool>(
    eq_challenges: Vec<Challenge<SC>>,
    chip_states: Vec<ChipState<'a, SC, A, D>>,
    num_rounds: usize,
    num_rounds_linear: usize,
    num_skip_rounds: usize,
    chip_log_height_threshold: usize,
    perm_alpha: Challenge<SC>,
    beta_powers: &'a [Challenge<SC>],
    beta_septix: Challenge<SC>,
    public_values: &'a [Val<SC>],
    num_chips_each_round: Vec<usize>,
    challenger: &mut MlChallenger<SC>,
) -> SumcheckRunOutput<SC>
where
    SC: SCStarkGenericConfig,
    A: MachineAir<Val<SC>>,
    A: for<'b> FullAir<ConstraintFolder<'b, Val<SC>, Val<SC>, Challenge<SC>>>,
    A: for<'b> FullAir<ConstraintFolder<'b, Val<SC>, Challenge<SC>, Challenge<SC>>>,
    Val<SC>: PolyAirExtendable<D> + PrimeField32,
{
    let mut sumcheck_protocol = SumcheckProtocol::<'_, SC, A, D, USE_ALGEBRAIC_DECOMP>::new(
        eq_challenges,
        chip_states,
        num_rounds,
        num_rounds_linear,
        num_skip_rounds,
        chip_log_height_threshold,
        perm_alpha,
        beta_powers,
        beta_septix,
        public_values,
        num_chips_each_round,
    );
    let sumcheck_start = Instant::now();
    tracing::info_span!("sumcheck").in_scope(|| sumcheck_protocol.prove(challenger));
    whir_profile::add_ms("open.polyair_sumcheck_ms", sumcheck_start.elapsed().as_millis());

    tracing::trace!("PROVER: final claim = {:?}", sumcheck_protocol.state.claim);
    for (i, cs) in sumcheck_protocol.state.chip_states.iter().enumerate() {
        tracing::trace!("  PROVER chip[{}]: claim={:?}, perm_claim={:?}, num_constraints={}, perm_last_alpha={:?}",
            i, cs.claim, cs.perm_claim, cs.num_constraints,
            cs.powers_of_alpha.last());
    }

    let mut sumcheck_challenges = sumcheck_protocol.state.sumcheck_challenges.clone();
    sumcheck_challenges.reverse();
    let sumcheck_proof = SumcheckProof { unipolys: sumcheck_protocol.unipolys };

    (sumcheck_challenges, sumcheck_proof)
}

/// Trait for sumcheck-based machine provers.
///
/// Defines the interface for proving STARK constraints via the sumcheck protocol,
/// including trace generation, commitment, opening, and full proof generation.
pub trait SCMachineProver<SC: SCStarkGenericConfig, A: MachineAir<SC::Val>, const D: usize>:
    'static + Send + Sync
where
    Val<SC>: PolyAirExtendable<D>,
{
    /// The matrix type used to store traces on the proving device.
    type DeviceMatrix;

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
    fn new(machine: SCStarkMachine<SC, A, D>) -> Self;

    /// Returns a reference to the underlying machine.
    fn machine(&self) -> &SCStarkMachine<SC, A, D>;

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
            .filter(|chip| chips_filter.is_none_or(|filter| filter.contains(&chip.name())))
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

    /// Computes the sumcheck opening proof for a single shard.
    ///
    /// Runs the sumcheck protocol over compressed matrices, then produces a PCS opening proof.
    fn open(
        &self,
        pk: &Self::DeviceProvingKey,
        data: SCShardMainData<SC, Self::DeviceCompressedMatrix, Self::DeviceProverData>,
        challenger: &mut MlChallenger<SC>,
        _num_skip_rounds: usize,
        _chip_log_height_threshold: usize,
    ) -> Result<SCShardProof<SC>, Self::Error>;

    /// Proves all shards end-to-end: generates traces, commits, and opens each shard.
    fn prove(
        &self,
        pk: &Self::DeviceProvingKey,
        records: Vec<A::Record>,
        challenger: &mut SC::MlChallenger,
        opts: <A::Record as MachineRecord>::Config,
        _num_skip_rounds: usize,
        _chip_log_height_threshold: usize,
    ) -> Result<SCMachineProof<SC>, Self::Error>;

    /// Returns the STARK configuration for the machine.
    fn config(&self) -> &SC {
        self.machine().config()
    }

    /// Returns the number of public values elements.
    fn num_pv_elts(&self) -> usize {
        self.machine().num_pv_elts()
    }

    /// Commits to the compressed main traces with an optional PCS stack log height
    /// inherited from the preprocessed proving key.
    fn commit_with_pcs_stack_log_height(
        &self,
        record: &A::Record,
        compressed_traces: Vec<(String, CompressedMatrix<Val<SC>>)>,
        _preprocessed_pcs_stack_log_height: Option<usize>,
    ) -> SCShardMainData<SC, Self::DeviceCompressedMatrix, Self::DeviceProverData> {
        self.commit(record, compressed_traces)
    }

    /// Returns an iterator over the base-field chips included in the given record.
    fn shard_chips<'a, 'b>(
        &'a self,
        record: &'b A::Record,
    ) -> impl Iterator<Item = &'b Chip<A, Val<SC>, D>>
    where
        'a: 'b,
        SC: 'b,
    {
        self.machine().shard_chips(record)
    }
}

impl<SC, A, const D: usize> SumcheckProver<SC, A, D>
where
    SC: 'static + SCStarkGenericConfig + Send + Sync,
    A: MachineAir<SC::Val>,
    Val<SC>: PolyAirExtendable<D>,
    MlCom<SC>: Send + Sync,
    MlPcsProverData<SC>: Send + Sync + Serialize + DeserializeOwned,
{
    fn commit_with_pcs_stack_log_height(
        &self,
        record: &A::Record,
        mut compressed_traces: Vec<(String, CompressedMatrix<Val<SC>>)>,
        preprocessed_pcs_stack_log_height: Option<usize>,
    ) -> SCShardMainData<SC, CompressedMatrix<Val<SC>, Val<SC>>, MlPcsProverData<SC>> {
        compressed_traces.sort_by_key(|(name, trace)| (Reverse(trace.total_height), name.clone()));
        let chip_ordering =
            compressed_traces.iter().enumerate().map(|(i, (name, _))| (name.clone(), i)).collect();

        let public_values_vec = record.public_values();
        let chips = self.machine.shard_chips_ordered(&chip_ordering).collect::<Vec<_>>();
        assert_eq!(chips.len(), compressed_traces.len(), "trace/chip inventory mismatch");
        if self.machine.contains_global_bus {
            let public_values: &PublicValues<Word<Val<SC>>, Val<SC>> =
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

        let config = &self.machine.config;
        let pcs = config.mlpcs();
        let batch_max_log_height = traces_refs
            .iter()
            .filter(|trace| trace.width() > 0)
            .map(|trace| p3_util::log2_strict_usize(trace.height()))
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
                (Some(b), Some(p)) => Some(b.max(p)),
                (Some(b), None) => Some(b),
                (None, p) => p,
            };
        let pcs_stack_log_height = config.mlpcs_target_stack_log_height(effective_max_log_height);
        let commit_options = config.mlpcs_commit_options_for_stack_log_height(pcs_stack_log_height);
        let (main_commit, main_data) = tracing::info_span!("commit to main traces")
            .in_scope(|| pcs.commit_with_options(traces_refs, commit_options));

        SCShardMainData {
            compressed_traces,
            main_commit,
            main_data,
            chip_ordering,
            public_values: public_values_vec,
            pcs_stack_log_height,
        }
    }
}

impl<SC, A, const D: usize> SCMachineProver<SC, A, D> for SumcheckProver<SC, A, D>
where
    SC: 'static + SCStarkGenericConfig + Send + Sync,
    A: MachineAir<SC::Val>,
    A: for<'a> FullAir<PrecomputeRowBuilder<'a, Val<SC>, Val<SC>, Challenge<SC>>>,
    A: for<'a> FullAir<PermutationRowBuilder<'a, Val<SC>, Challenge<SC>>>,
    A: for<'b> FullAir<ConstraintFolder<'b, Val<SC>, Val<SC>, Challenge<SC>>>,
    A: for<'b> FullAir<ConstraintFolder<'b, Val<SC>, Challenge<SC>, Challenge<SC>>>,
    Val<SC>: PolyAirExtendable<D> + PrimeField32,
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

    fn new(machine: SCStarkMachine<SC, A, D>) -> Self {
        Self { machine }
    }

    fn machine(&self) -> &SCStarkMachine<SC, A, D> {
        &self.machine
    }

    fn setup(&self, program: &A::Program) -> (Self::DeviceProvingKey, SCStarkVerifyingKey<SC>) {
        self.machine().setup(program)
    }

    fn pk_to_device(&self, pk: &SCStarkProvingKey<SC>) -> Self::DeviceProvingKey {
        SCStarkProvingKey::clone(pk)
    }

    fn pk_to_host(&self, pk: &Self::DeviceProvingKey) -> SCStarkProvingKey<SC> {
        SCStarkProvingKey::clone(pk)
    }

    fn commit(
        &self,
        record: &A::Record,
        compressed_traces: Vec<(String, CompressedMatrix<Val<SC>>)>,
    ) -> SCShardMainData<SC, Self::DeviceCompressedMatrix, Self::DeviceProverData> {
        SumcheckProver::commit_with_pcs_stack_log_height(self, record, compressed_traces, None)
    }

    fn commit_with_pcs_stack_log_height(
        &self,
        record: &A::Record,
        compressed_traces: Vec<(String, CompressedMatrix<Val<SC>>)>,
        preprocessed_pcs_stack_log_height: Option<usize>,
    ) -> SCShardMainData<SC, Self::DeviceCompressedMatrix, Self::DeviceProverData> {
        SumcheckProver::commit_with_pcs_stack_log_height(
            self,
            record,
            compressed_traces,
            preprocessed_pcs_stack_log_height,
        )
    }

    fn open(
        &self,
        pk: &Self::DeviceProvingKey,
        data: SCShardMainData<SC, Self::DeviceCompressedMatrix, Self::DeviceProverData>,
        challenger: &mut MlChallenger<SC>,
        num_skip_rounds: usize,
        chip_log_height_threshold: usize,
    ) -> Result<SCShardProof<SC>, Self::Error> {
        assert!(chip_log_height_threshold == 0);
        let chips: Vec<&Chip<A, Val<SC>, D>> =
            self.machine().shard_chips_ordered(&data.chip_ordering).collect::<Vec<_>>();

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

        let perm_alpha: Challenge<SC> = challenger.sample_ext_element();
        let perm_beta: Challenge<SC> = challenger.sample_ext_element();
        let max_beta_powers = chips.iter().map(|c| c.required_max_beta_power()).max().unwrap();
        let beta_powers = {
            let mut res = Vec::with_capacity(max_beta_powers + 1);
            let mut powers = perm_beta.powers();
            for _ in 0..(max_beta_powers + 1) {
                res.push(powers.next().unwrap());
            }
            res
        };
        #[cfg(feature = "koalabear")]
        let beta_septix = compute_beta_septix::<
            Val<SC>,
            Challenge<SC>,
            dt_stark::septic_curve_params::KoalaBearCurveParams,
        >(perm_beta);
        #[cfg(feature = "babybear")]
        let beta_septix = compute_beta_septix::<
            Val<SC>,
            Challenge<SC>,
            dt_stark::septic_curve_params::BabyBearCurveParams,
        >(perm_beta);
        let precompute_permutation_start = Instant::now();
        let timer = tracing::info_span!("generate precompute and permutation").entered();
        let (precompute_lcs, (reserved_polys, (perms, local_sums))): (
            Vec<CompressedMatrix<Challenge<SC>, Challenge<SC>>>,
            (
                Vec<CompressedMatrix<Val<SC>, Val<SC>>>,
                (Vec<CompressedMatrix<Challenge<SC>, Challenge<SC>>>, Vec<Challenge<SC>>),
            ),
        ) = (0..chips.len())
            .into_par_iter()
            .map(|i| {
                let chip = chips[i];
                let main = &data.compressed_traces[i].1;
                let prep = preprocessed_compressed[i];
                // T-K fused gather: one pass over (prep, main) builds all
                // three phase inputs instead of three full re-reads.
                let (precompute_lc, reserved_poly, perm, local_sum) =
                    fused_precompute_reserved_permutation(
                        &chip.air,
                        prep,
                        main,
                        &data.public_values[0..self.num_pv_elts()],
                        perm_alpha,
                        &beta_powers,
                        beta_septix,
                        chip.num_precompute(),
                        chip.reserved_poly(),
                        chip.logup_batch_size(),
                        chip.num_lookup(),
                    );

                (precompute_lc, (reserved_poly, (perm, local_sum)))
            })
            .unzip();
        timer.exit();
        whir_profile::add_ms(
            "open.polyair_precompute_permutation_ms",
            precompute_permutation_start.elapsed().as_millis(),
        );

        for i in 0..chips.len() {
            let trace_width = data.compressed_traces[i].1.width();
            let log_trace_height = log_heights[i];
            let stored_height = data.compressed_traces[i].1.stored_height();
            let prep_width = preprocessed_compressed[i].map_or(0, |c| c.main.width());
            let permutation_width = perms[i].main.width();
            tracing::info!(
                "{:<40} | Main Cols = {:<5} | Pre Cols = {:<5}  | Perm Cols = {:<5} | Rows = {:<6} (padded 2^{})",
                chips[i].name(),
                trace_width,
                prep_width,
                permutation_width,
                stored_height,
                log_trace_height,
            );
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

        // Flatten the EF perm traces to base while KEEPING the padding
        // compressed: the old decompress()+flatten_to_base() detour
        // materialized two full-height copies per chip and re-expanded the
        // padding region the fused gather had deliberately stored as one row.
        let perm_flatten_start = Instant::now();
        let permutation_traces_base: Vec<CompressedMatrix<Val<SC>>> =
            perms.par_iter().map(|c| c.flatten_to_base::<Val<SC>>()).collect::<Vec<_>>();
        whir_profile::add_ms(
            "open.polyair_perm_flatten_ms",
            perm_flatten_start.elapsed().as_millis(),
        );
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
            let perm_commit_options =
                self.config().mlpcs_commit_options_for_stack_log_height(data.pcs_stack_log_height);
            let permutation_commit_start = Instant::now();
            let (permutation_commit, permutation_data) =
                tracing::info_span!("commit to permutation traces").in_scope(|| {
                    pcs.commit_with_options(
                        permutation_traces_base.iter().collect(),
                        perm_commit_options,
                    )
                });
            whir_profile::add_ms(
                "open.polyair_permutation_commit_ms",
                permutation_commit_start.elapsed().as_millis(),
            );
            challenger.observe(permutation_commit.clone());
            Some((permutation_commit, permutation_data))
        } else {
            None
        };
        for local_sum in local_sums.iter() {
            challenger.observe_slice(
                <Challenge<SC> as AbstractExtensionField<Val<SC>>>::as_base_slice(local_sum),
            );
        }

        let alpha: Challenge<SC> = challenger.sample_ext_element::<Challenge<SC>>();

        let max_height = *log_heights.iter().max().unwrap();
        let num_rounds_linear = max_height.saturating_sub(chip_log_height_threshold);
        let num_rounds_nonlinear =
            std::cmp::min(max_height, chip_log_height_threshold) / num_skip_rounds;
        let num_rounds = num_rounds_linear + num_rounds_nonlinear;

        let eq_challenges: Vec<Challenge<SC>> =
            (0..num_rounds).map(|_| challenger.sample_ext_element()).collect();
        let num_constraints: Vec<usize> = chips
            .iter()
            .map(|chip| {
                *HashMap::<String, usize>::get(&pk.constraints_map, &chip.name())
                    .expect("chip not found in constraints map")
            })
            .collect();
        let num_chips_each_round = compute_num_chips_each_round(&log_heights, 1, 0);
        let powers_of_alpha = compute_powers_of_alpha(alpha, num_constraints.clone());

        let chip_states: Vec<ChipState<'_, SC, A, D>> = (0..chips.len())
            .map(|i| {
                ChipState::new(
                    i,
                    log_heights[i],
                    chips[i],
                    &reserved_polys[i],
                    &precompute_lcs[i],
                    &perms[i],
                    local_sums[i],
                    powers_of_alpha[i].clone(),
                    num_constraints[i],
                    perm_alpha,
                    &beta_powers,
                    beta_septix,
                    &data.public_values,
                )
            })
            .collect();

        let use_algebraic_decomp = configured_use_algebraic_decomp();
        tracing::info!("polyair sumcheck algebraic_decomp={}", use_algebraic_decomp);
        let (sumcheck_challenges, sumcheck_proof) = if use_algebraic_decomp {
            prove_sumcheck_with_decomp::<SC, A, D, true>(
                eq_challenges,
                chip_states,
                num_rounds,
                num_rounds_linear,
                1,
                0,
                perm_alpha,
                &beta_powers,
                beta_septix,
                &data.public_values,
                num_chips_each_round,
                challenger,
            )
        } else {
            prove_sumcheck_with_decomp::<SC, A, D, false>(
                eq_challenges,
                chip_states,
                num_rounds,
                num_rounds_linear,
                1,
                0,
                perm_alpha,
                &beta_powers,
                beta_septix,
                &data.public_values,
                num_chips_each_round,
                challenger,
            )
        };

        let finalize_start = Instant::now();
        let sumcheck_eq_poly_coeffs = compute_eq_poly_coeffs(&sumcheck_challenges);
        let opened_values = finalize::<SC>(
            &log_heights,
            &data.compressed_traces,
            &preprocessed_compressed,
            &perms,
            &sumcheck_eq_poly_coeffs,
            &local_sums,
        );
        let opening_point = sumcheck_challenges;

        let preprocessed_traces_for_open = pk.get_preprocessed_traces_for_open(&chip_names);
        whir_profile::add_ms(
            "open.polyair_finalize_opened_values_ms",
            finalize_start.elapsed().as_millis(),
        );

        // Produce the PCS batch opening proof.
        let batch_open_start = Instant::now();
        let batch_open_result = tracing::info_span!("batch open").in_scope(|| {
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
                        vec![
                            clone_pk_data_for_open(pk, "polyair_with_permutation"),
                            data.main_data,
                            permutation_data,
                        ],
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
                        vec![
                            clone_pk_data_for_open(pk, "polyair_without_permutation"),
                            data.main_data,
                        ],
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
        });
        whir_profile::add_ms(
            "open.polyair_pcs_batch_open_ms",
            batch_open_start.elapsed().as_millis(),
        );
        batch_open_result
    }

    fn prove(
        &self,
        pk: &Self::DeviceProvingKey,
        mut records: Vec<A::Record>,
        challenger: &mut <SC as SCStarkGenericConfig>::MlChallenger,
        opts: <A::Record as MachineRecord>::Config,
        num_skip_rounds: usize,
        chip_log_height_threshold: usize,
    ) -> Result<SCMachineProof<SC>, Self::Error> {
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
