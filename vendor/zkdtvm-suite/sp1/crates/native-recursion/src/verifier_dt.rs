use bincode::Options as _;
use dt_core_machine::reduce::DTReduceProof;
use dt_stark::{
    air::MachineAir,
    global_d11::{
        canonical_program_boundary_fields_v1, ProgramImageBoundaryV1, GLOBAL146_COMPOSITE_IDENTITY,
    },
    koalabear_poseidon2::{
        install_embedded_whir_config, koala_bear_poseidon2::SCKoalaBearSha256Root,
        parse_embedded_whir_json, StageJsonConfig, WhirJsonConfig,
    },
    sumcheck::keys::SCStarkVerifyingKey,
};
use p3_field::PrimeField32;
use serde::{Deserialize, Serialize};

use crate::{
    compress_dt::{checked_native_root_public_interval, root_vk_digest, verifying_keys_equal},
    config::{RootSC, DIGEST_SIZE, F},
    machine_dt::{
        native_root_verifier_machine, verify_root_recursion_shard, NativeRecursionAssemblyError,
        NativeRecursionAssemblyResult, NativeRootMachine,
    },
    native_air_dt::NativeRecursionLayer,
    statement_dt::{validate_native_root_global_interval, NATIVE_PV_DT_VK_DIGEST_START},
    system_dt::RecursionNativeProgram,
};

pub const NATIVE_ROOT_VERIFIER_ARTIFACT_SCHEMA_V1: u32 = 1;

/// Frozen vk_L4 statement digest for the current KoalaBear/ext5, SHA256-root product.
///
/// The full trusted VK remains part of the artifact because this statement digest deliberately
/// does not cover every host-only VK metadata map.
///
/// Re-pinned for the v0.8 verifier release from two independent uncached RSP artifact exports;
/// both produced byte-identical authority artifacts. The provider-free verifier also recomputes
/// this digest after artifact round-trip decoding before accepting the authority.
pub const NATIVE_ROOT_VK_FROZEN_DIGEST_V1: [u32; DIGEST_SIZE] =
    [353571810, 594284398, 939220650, 1315032732, 214802506, 1865505683, 636054627, 1352906159];

const FROZEN_ROOT_STACK_LOG_HEIGHT_V1: usize = 18;
const FROZEN_ROOT_FINAL_POLY_LOG_HEIGHT_V1: usize = 6;
const FROZEN_ROOT_COMMITTED_GROUPS_V1: usize = 3;
const FROZEN_ROOT_ROUND_QUERY_COUNTS_V1: [usize; FROZEN_ROOT_COMMITTED_GROUPS_V1] = [67, 36, 24];
const FROZEN_ROOT_FOLDING_ROUNDS_PER_GROUP_V1: usize = (FROZEN_ROOT_STACK_LOG_HEIGHT_V1 -
    FROZEN_ROOT_FINAL_POLY_LOG_HEIGHT_V1) /
    FROZEN_ROOT_COMMITTED_GROUPS_V1;

/// Self-contained verifier authority for one frozen native L4 machine.
///
/// This object is a trust root, not proof-supplied input. In particular, the compact
/// `frozen_root_vk_digest` does not bind all host VK metadata, the L4 program, or the expected
/// application statement. A consumer must authenticate this complete artifact (normally by
/// compiling its bytes into the verifier) before passing it to
/// [`NativeRootVerifier::from_artifact`]. Never deserialize this authority directly from a proof,
/// request, or other runtime user input.
#[derive(Clone, Serialize, Deserialize)]
pub struct NativeRootVerifierArtifactV1 {
    pub schema_version: u32,
    pub package_version: String,
    pub global146_identity: [u8; 32],
    pub whir_config_json: String,
    pub frozen_root_vk_digest: [u32; DIGEST_SIZE],
    pub l4_program: RecursionNativeProgram<F>,
    pub trusted_l4_vk: SCStarkVerifyingKey<RootSC>,
    pub expected_core_statement_digest: [F; DIGEST_SIZE],
    pub expected_program_boundary: ProgramImageBoundaryV1<u32>,
}

impl NativeRootVerifierArtifactV1 {
    pub fn new(
        whir_config_json: String,
        l4_program: RecursionNativeProgram<F>,
        trusted_l4_vk: SCStarkVerifyingKey<RootSC>,
        expected_core_statement_digest: [F; DIGEST_SIZE],
        expected_program_boundary: ProgramImageBoundaryV1<u32>,
    ) -> Self {
        Self {
            schema_version: NATIVE_ROOT_VERIFIER_ARTIFACT_SCHEMA_V1,
            package_version: env!("CARGO_PKG_VERSION").to_string(),
            global146_identity: GLOBAL146_COMPOSITE_IDENTITY,
            whir_config_json,
            frozen_root_vk_digest: NATIVE_ROOT_VK_FROZEN_DIGEST_V1,
            l4_program,
            trusted_l4_vk,
            expected_core_statement_digest,
            expected_program_boundary,
        }
    }

    /// Re-decodes and initializes the exact serialized release bytes before an exporter writes
    /// them. This makes the producer exercise the same authority path as a WASM consumer.
    pub fn validate_serialized_roundtrip(&self, bytes: &[u8]) -> NativeRecursionAssemblyResult<()> {
        let expected = bincode::DefaultOptions::new()
            .with_little_endian()
            .with_fixint_encoding()
            .serialize(self)
            .map_err(|error| validation(format!("serialize verifier artifact: {error}")))?;
        if expected != bytes {
            return Err(validation(
                "serialized verifier artifact bytes differ from the export candidate",
            ));
        }
        let decoded = bincode::DefaultOptions::new()
            .with_little_endian()
            .with_fixint_encoding()
            .reject_trailing_bytes()
            .deserialize(bytes)
            .map_err(|error| validation(format!("round-trip verifier artifact: {error}")))?;
        NativeRootVerifier::from_artifact(decoded).map(|_| ())
    }
}

/// Provider-free verifier for one fully self-contained native root proof.
pub struct NativeRootVerifier {
    machine: NativeRootMachine,
    trusted_l4_vk: SCStarkVerifyingKey<RootSC>,
    expected_core_statement_digest: [F; DIGEST_SIZE],
    expected_program_boundary: ProgramImageBoundaryV1<u32>,
}

impl NativeRootVerifier {
    /// Validates and installs an already-authenticated artifact authority before constructing the
    /// provider-free machine.
    ///
    /// This method rejects schema/config/identity drift, but cannot make attacker-supplied program,
    /// full VK metadata, or expected-statement fields trustworthy. The caller must pin the complete
    /// serialized artifact outside this API (for example with `include_bytes!`); runtime
    /// user-controlled artifact bytes are not a supported input to this constructor.
    pub fn from_artifact(
        artifact: NativeRootVerifierArtifactV1,
    ) -> NativeRecursionAssemblyResult<Self> {
        if artifact.schema_version != NATIVE_ROOT_VERIFIER_ARTIFACT_SCHEMA_V1 {
            return Err(validation(format!(
                "unsupported native root verifier artifact schema {}",
                artifact.schema_version
            )));
        }
        if artifact.package_version != env!("CARGO_PKG_VERSION") {
            return Err(validation(format!(
                "native root verifier package version mismatch: artifact={} verifier={}",
                artifact.package_version,
                env!("CARGO_PKG_VERSION")
            )));
        }
        if artifact.global146_identity != GLOBAL146_COMPOSITE_IDENTITY {
            return Err(validation("native root verifier Global146 identity mismatch"));
        }
        if artifact.frozen_root_vk_digest != NATIVE_ROOT_VK_FROZEN_DIGEST_V1 {
            return Err(validation("native root verifier vk_L4 pin mismatch"));
        }
        if artifact.trusted_l4_vk.global146_identity != artifact.global146_identity {
            return Err(validation("trusted vk_L4 carries a different Global146 identity"));
        }
        artifact
            .trusted_l4_vk
            .owner_registry
            .validate()
            .map_err(|error| validation(format!("{error:?}")))?;
        canonical_program_boundary_fields_v1::<F>(&artifact.expected_program_boundary)
            .map_err(|error| validation(format!("{error:?}")))?;
        if artifact.l4_program.layer()? != NativeRecursionLayer::L4Root {
            return Err(validation("native root verifier artifact program is not L4"));
        }

        let parsed_config =
            parse_embedded_whir_json(&artifact.whir_config_json).map_err(validation)?;
        validate_frozen_whir_config_v1(&parsed_config)?;

        let got_root_digest =
            root_vk_digest(&artifact.trusted_l4_vk).map(|limb| limb.as_canonical_u32());
        if got_root_digest != artifact.frozen_root_vk_digest {
            return Err(validation(format!(
                "trusted vk_L4 does not match the frozen digest: actual={got_root_digest:?} expected={:?}",
                artifact.frozen_root_vk_digest,
            )));
        }

        install_embedded_whir_config(parsed_config).map_err(validation)?;
        let root_config =
            SCKoalaBearSha256Root::root_shrink_from_installed_embedded().map_err(validation)?;
        let machine = native_root_verifier_machine(&artifact.l4_program, root_config)?;
        validate_machine_vk_authority(&machine, &artifact.trusted_l4_vk)?;

        Ok(Self {
            machine,
            trusted_l4_vk: artifact.trusted_l4_vk,
            expected_core_statement_digest: artifact.expected_core_statement_digest,
            expected_program_boundary: artifact.expected_program_boundary,
        })
    }

    /// Verifies a new-format native root proof carrying every preprocessed input opening.
    pub fn verify_full(&self, reduce: &DTReduceProof<RootSC>) -> NativeRecursionAssemblyResult<()> {
        self.verify_full_with_statement(
            reduce,
            &self.expected_core_statement_digest,
            &self.expected_program_boundary,
        )
    }

    /// Verifies against the caller's application authority while reusing the fixed L4 machine.
    /// Expected values must come from a trusted application VK, never from this proof.
    pub fn verify_full_with_statement(
        &self,
        reduce: &DTReduceProof<RootSC>,
        expected_core_statement_digest: &[F; DIGEST_SIZE],
        expected_program_boundary: &ProgramImageBoundaryV1<u32>,
    ) -> NativeRecursionAssemblyResult<()> {
        canonical_program_boundary_fields_v1::<F>(expected_program_boundary)
            .map_err(|error| validation(format!("{error:?}")))?;
        if !verifying_keys_equal(&self.trusted_l4_vk, &reduce.vk) {
            return Err(validation("presented root vk differs from the trusted vk_L4"));
        }
        require_safe_root_polyair_shape(&self.machine, &self.trusted_l4_vk, reduce)?;
        require_full_root_input_opening(reduce)?;

        let public = &reduce.proof.public_values;
        let (global_start, global_end) = checked_native_root_public_interval(public)?;
        verify_root_recursion_shard(&self.machine, &self.trusted_l4_vk, &reduce.proof)?;

        for idx in 0..DIGEST_SIZE {
            if public[NATIVE_PV_DT_VK_DIGEST_START + idx] !=
                expected_core_statement_digest[idx]
            {
                return Err(validation(
                    "root dt_vk digest does not match the expected core statement",
                ));
            }
        }
        validate_native_root_global_interval(
            expected_program_boundary,
            global_start,
            global_end,
        )
        .map_err(validation)?;
        Ok(())
    }

    pub fn trusted_l4_vk(&self) -> &SCStarkVerifyingKey<RootSC> {
        &self.trusted_l4_vk
    }
}

fn validation(error: impl std::fmt::Display) -> NativeRecursionAssemblyError {
    NativeRecursionAssemblyError::Validation(error.to_string())
}

fn validate_machine_vk_authority(
    machine: &NativeRootMachine,
    vk: &SCStarkVerifyingKey<RootSC>,
) -> NativeRecursionAssemblyResult<()> {
    if machine.global_boundary_registry != vk.owner_registry {
        return Err(validation("trusted vk_L4 owner registry does not match the frozen L4 machine"));
    }
    if vk.constraints_map.len() != machine.chips().len() ||
        machine.chips().iter().any(|chip| !vk.constraints_map.contains_key(&chip.name()))
    {
        return Err(validation(
            "trusted vk_L4 constraint inventory does not match the frozen L4 machine",
        ));
    }
    if vk.chip_information.len() != vk.chip_ordering.len() ||
        vk.chip_information
            .iter()
            .enumerate()
            .any(|(index, (name, _))| vk.chip_ordering.get(name) != Some(&index))
    {
        return Err(validation("trusted vk_L4 preprocessed-chip inventory is not canonical"));
    }
    Ok(())
}

pub(crate) fn require_safe_root_polyair_shape(
    machine: &NativeRootMachine,
    vk: &SCStarkVerifyingKey<RootSC>,
    reduce: &DTReduceProof<RootSC>,
) -> NativeRecursionAssemblyResult<()> {
    let proof = &reduce.proof;
    let opened = &proof.opened_values.chips;
    let selected = machine.shard_chips_ordered(&proof.chip_ordering).collect::<Vec<_>>();
    let constrained =
        selected.iter().filter(|chip| vk.constraints_map.contains_key(&chip.name())).count();
    let log_heights = opened.iter().map(|chip| chip.log_height).collect::<Vec<_>>();
    let eval_lengths =
        proof.sumcheck_proof.unipolys.iter().map(|poly| poly.evals.len()).collect::<Vec<_>>();
    validate_root_polyair_lengths(
        &log_heights,
        &eval_lengths,
        proof.chip_ordering.len(),
        selected.len(),
        constrained,
    )?;
    validate_permutation_commitment_presence(
        proof.commitment.permutation_commit.is_some(),
        &opened.iter().map(|chip| chip.permutation.local.len()).collect::<Vec<_>>(),
    )?;

    for (chip, opening) in selected.iter().zip(opened) {
        let name = chip.name();
        if opening.preprocessed.local.len() != chip.preprocessed_width() ||
            opening.main.local.len() != p3_air::BaseAir::width(*chip) ||
            opening.permutation.local.len() != chip.perm_width()
        {
            return Err(validation(format!(
                "root chip {name} opening widths do not match the frozen machine",
            )));
        }
        if opening.preprocessed.local.is_empty() {
            continue;
        }
        let trusted_index = vk.chip_ordering.get(&name).ok_or_else(|| {
            validation(format!("root chip {name} has no trusted preprocessed VK entry"))
        })?;
        let (trusted_name, trusted_dimensions) =
            vk.chip_information.get(*trusted_index).ok_or_else(|| {
                validation(format!("root chip {name} trusted preprocessed index is out of range"))
            })?;
        let height = 1usize.checked_shl(opening.log_height as u32).ok_or_else(|| {
            validation(format!("root chip {name} log height exceeds the platform width"))
        })?;
        if trusted_name != &name ||
            trusted_dimensions.width != opening.preprocessed.local.len() ||
            trusted_dimensions.height != height
        {
            return Err(validation(format!(
                "root chip {name} preprocessed dimensions differ from trusted vk_L4",
            )));
        }
    }
    Ok(())
}

fn validate_root_polyair_lengths(
    log_heights: &[usize],
    eval_lengths: &[usize],
    chip_ordering_len: usize,
    selected_chip_count: usize,
    constrained_chip_count: usize,
) -> NativeRecursionAssemblyResult<()> {
    if log_heights.is_empty() {
        return Err(validation("root proof has no opened chips"));
    }
    if chip_ordering_len != log_heights.len() || selected_chip_count != log_heights.len() {
        return Err(validation("root proof chip ordering does not select exactly its opened chips"));
    }
    if constrained_chip_count != selected_chip_count {
        return Err(validation("root proof selects a chip absent from trusted vk_L4 constraints"));
    }
    let max_log_height = log_heights[0];
    if max_log_height >= 31 ||
        log_heights.iter().any(|height| *height >= 31) ||
        log_heights.windows(2).any(|pair| pair[0] < pair[1])
    {
        return Err(validation(
            "root proof chip log heights are not verifier-safe descending heights below 31",
        ));
    }
    if eval_lengths.len() != max_log_height {
        return Err(validation(format!(
            "root outer sumcheck has {} rounds, expected {max_log_height}",
            eval_lengths.len(),
        )));
    }
    if eval_lengths.iter().any(|length| *length < 2) {
        return Err(validation(
            "root outer sumcheck contains a univariate with fewer than two evaluations",
        ));
    }
    Ok(())
}

fn validate_permutation_commitment_presence(
    has_commitment: bool,
    permutation_widths: &[usize],
) -> NativeRecursionAssemblyResult<()> {
    let has_openings = permutation_widths.iter().any(|width| *width != 0);
    if has_commitment != has_openings {
        return Err(validation(
            "root permutation commitment presence does not match its opened columns",
        ));
    }
    Ok(())
}

/// Enforces the frozen product's complete L4 opening shape.
///
/// In particular, this rejects the former wire optimization which omitted the first stacked
/// input-opening batch. Verification never reconstructs proof bytes from proving-key data.
pub(crate) fn require_full_root_input_opening(
    reduce: &DTReduceProof<RootSC>,
) -> NativeRecursionAssemblyResult<()> {
    let proof = &reduce.proof;
    let opening = &proof.opening_proof;
    if opening.stack_log_height != Some(FROZEN_ROOT_STACK_LOG_HEIGHT_V1) {
        return Err(validation("root proof does not carry the frozen stacked opening"));
    }
    if opening.iopp_pruned.is_some() || !opening.iopp_queries.is_empty() {
        return Err(validation("root proof mixes legacy and per-round IOPP openings"));
    }
    if opening.grinding_batching_witness.len() != 2 || !opening.grinding_query_witness.is_empty() {
        return Err(validation("root proof carries non-canonical grinding witnesses"));
    }
    let reduction = opening
        .stacking_reduction
        .as_ref()
        .ok_or_else(|| validation("root proof is missing the stacking reduction"))?;
    let reduction_coeff_lengths =
        reduction.sumcheck.uni_polys.iter().map(|poly| poly.coeffs.len()).collect::<Vec<_>>();
    require_degree_two_rounds(
        "root stacking-reduction sumcheck",
        &reduction_coeff_lengths,
        FROZEN_ROOT_STACK_LOG_HEIGHT_V1,
    )?;
    let whir_coeff_lengths = opening
        .sumcheck_transcript
        .uni_polys
        .iter()
        .map(|poly| poly.coeffs.len())
        .collect::<Vec<_>>();
    require_degree_two_rounds(
        "root WHIR sumcheck",
        &whir_coeff_lengths,
        FROZEN_ROOT_STACK_LOG_HEIGHT_V1 - FROZEN_ROOT_FINAL_POLY_LOG_HEIGHT_V1,
    )?;
    if opening.iopp_oracles.len() != FROZEN_ROOT_COMMITTED_GROUPS_V1 ||
        opening.ood_values.len() != FROZEN_ROOT_COMMITTED_GROUPS_V1 - 1 ||
        opening.final_poly.len() != 1usize << FROZEN_ROOT_FINAL_POLY_LOG_HEIGHT_V1
    {
        return Err(validation("root proof WHIR commitment schedule differs from the frozen shape"));
    }
    let round_iopp = opening
        .round_iopp
        .as_ref()
        .ok_or_else(|| validation("root proof is missing the per-round IOPP opening"))?;
    if !round_iopp.rounds.is_empty() || round_iopp.pruned.is_none() {
        return Err(validation(
            "root proof does not carry the frozen pruned per-round IOPP opening",
        ));
    }
    require_exact_group_lengths(
        "root per-round query witnesses",
        &round_iopp.query_witnesses.iter().map(Vec::len).collect::<Vec<_>>(),
        FROZEN_ROOT_COMMITTED_GROUPS_V1,
        2,
    )?;
    require_exact_group_lengths(
        "root per-round folding witnesses",
        &round_iopp.folding_witnesses.iter().map(Vec::len).collect::<Vec<_>>(),
        FROZEN_ROOT_COMMITTED_GROUPS_V1,
        2 * FROZEN_ROOT_FOLDING_ROUNDS_PER_GROUP_V1,
    )?;
    let round_pruned = round_iopp.pruned.as_ref().expect("checked above");
    if round_pruned.rounds.len() != FROZEN_ROOT_COMMITTED_GROUPS_V1 {
        return Err(validation("root proof pruned IOPP group count differs from the frozen shape"));
    }
    for (round, expected_queries) in
        round_pruned.rounds.iter().zip(FROZEN_ROOT_ROUND_QUERY_COUNTS_V1)
    {
        if round.query_to_unique_slot.len() != expected_queries ||
            round.opened_rows.is_empty() ||
            round.opened_rows.len() > expected_queries
        {
            return Err(validation(
                "root proof pruned IOPP query shape differs from the frozen schedule",
            ));
        }
    }
    if !opening.query_openings.per_query.is_empty() {
        return Err(validation("root proof mixes standard and pruned input openings"));
    }
    let pruned = opening
        .query_openings
        .pruned
        .as_ref()
        .ok_or_else(|| validation("root proof is missing pruned input openings"))?;
    require_full_root_input_opening_batches(
        proof.dimensions.len(),
        proof.commitment.permutation_commit.is_some(),
        pruned.round_pruned.len(),
        pruned.round_opened_values.len(),
        pruned.query_to_unique_slot.len(),
    )?;
    for (opened_values, query_to_unique_slot) in
        pruned.round_opened_values.iter().zip(&pruned.query_to_unique_slot)
    {
        if query_to_unique_slot.len() != FROZEN_ROOT_ROUND_QUERY_COUNTS_V1[0] ||
            opened_values.is_empty() ||
            opened_values.len() > FROZEN_ROOT_ROUND_QUERY_COUNTS_V1[0]
        {
            return Err(validation(
                "root proof pruned input-opening query shape differs from the frozen schedule",
            ));
        }
    }
    Ok(())
}

fn require_full_root_input_opening_batches(
    dimensions_len: usize,
    has_permutation_commitment: bool,
    pruned_len: usize,
    opened_len: usize,
    query_to_unique_slot_len: usize,
) -> NativeRecursionAssemblyResult<()> {
    let expected_batches = 2 + usize::from(has_permutation_commitment);
    if dimensions_len != expected_batches ||
        pruned_len != expected_batches ||
        opened_len != expected_batches ||
        query_to_unique_slot_len != expected_batches
    {
        return Err(validation(format!(
            "root proof does not carry every input-opening batch: dimensions={dimensions_len} \
             pruned={pruned_len} opened={opened_len} q2u={query_to_unique_slot_len} \
             expected={expected_batches}",
        )));
    }
    Ok(())
}

fn require_degree_two_rounds(
    label: &str,
    coefficient_lengths: &[usize],
    expected_rounds: usize,
) -> NativeRecursionAssemblyResult<()> {
    if coefficient_lengths.len() != expected_rounds ||
        coefficient_lengths.iter().any(|length| *length != 3)
    {
        return Err(validation(format!("{label} is not {expected_rounds} degree-two rounds",)));
    }
    Ok(())
}

fn require_exact_group_lengths(
    label: &str,
    lengths: &[usize],
    expected_groups: usize,
    expected_each: usize,
) -> NativeRecursionAssemblyResult<()> {
    if lengths.len() != expected_groups || lengths.iter().any(|length| *length != expected_each) {
        return Err(validation(format!(
            "{label} is not {expected_groups} groups of length {expected_each}",
        )));
    }
    Ok(())
}

fn validate_frozen_whir_config_v1(config: &WhirJsonConfig) -> NativeRecursionAssemblyResult<()> {
    if config != &frozen_whir_config_v1() {
        return Err(validation(
            "embedded WHIR configuration differs from the frozen verifier-v1 configuration",
        ));
    }
    Ok(())
}

fn frozen_whir_config_v1() -> WhirJsonConfig {
    let non_stacking = |log_blowup, num_queries| StageJsonConfig {
        log_blowup: Some(log_blowup),
        num_queries: Some(num_queries),
        grinding_bits_query: Some(20),
        grinding_bits_batching: Some(10),
        grinding_bits_folding: None,
        log_final_poly_len: None,
        num_committed_groups: None,
        round_query_counts: None,
        stack_log_height: None,
        stacking: Some(false),
        path_pruning: Some(false),
    };
    WhirJsonConfig {
        num_skip_rounds: Some(1),
        chip_log_height_threshold: Some(0),
        use_algebraic_decomp: Some(true),
        core: Some(non_stacking(1, 261)),
        compress: Some(non_stacking(2, 160)),
        shrink: Some(non_stacking(3, 131)),
        root_shrink: Some(StageJsonConfig {
            log_blowup: Some(4),
            num_queries: None,
            grinding_bits_query: Some(20),
            grinding_bits_batching: Some(20),
            grinding_bits_folding: Some(20),
            log_final_poly_len: Some(6),
            num_committed_groups: Some(3),
            round_query_counts: Some(FROZEN_ROOT_ROUND_QUERY_COUNTS_V1.to_vec()),
            stack_log_height: Some(FROZEN_ROOT_STACK_LOG_HEIGHT_V1),
            stacking: Some(true),
            path_pruning: Some(true),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PRODUCT_WHIR_JSON: &str = include_str!("../../../../whir_config_koalabear_ext5.json");

    #[test]
    fn frozen_v1_accepts_the_product_whir_json() {
        let parsed = parse_embedded_whir_json(PRODUCT_WHIR_JSON).unwrap();
        validate_frozen_whir_config_v1(&parsed).unwrap();
    }

    #[test]
    fn frozen_v1_rejects_query_schedule_drift() {
        let mut parsed = parse_embedded_whir_json(PRODUCT_WHIR_JSON).unwrap();
        parsed.root_shrink.as_mut().unwrap().round_query_counts = Some(vec![67, 36, 23]);
        assert!(validate_frozen_whir_config_v1(&parsed).is_err());
    }

    #[test]
    fn embedded_config_install_is_idempotent_and_rejects_drift() {
        let parsed = parse_embedded_whir_json(PRODUCT_WHIR_JSON).unwrap();
        install_embedded_whir_config(parsed.clone()).unwrap();
        install_embedded_whir_config(parsed.clone()).unwrap();
        let mut drifted = parsed;
        drifted.root_shrink.as_mut().unwrap().stack_log_height = Some(17);
        assert!(install_embedded_whir_config(drifted).is_err());
    }

    #[test]
    fn root_polyair_preflight_rejects_panic_shapes() {
        assert!(validate_root_polyair_lengths(&[], &[], 0, 0, 0).is_err());
        assert!(validate_root_polyair_lengths(&[2], &[2, 2], 0, 1, 1).is_err());
        assert!(validate_root_polyair_lengths(&[2], &[2, 2], 1, 0, 0).is_err());
        assert!(validate_root_polyair_lengths(&[2], &[2, 2], 1, 1, 0).is_err());
        assert!(validate_root_polyair_lengths(&[2], &[2], 1, 1, 1).is_err());
        assert!(validate_root_polyair_lengths(&[2], &[2, 1], 1, 1, 1).is_err());
        assert!(validate_root_polyair_lengths(&[31], &vec![2; 31], 1, 1, 1).is_err());
        assert!(validate_root_polyair_lengths(&[3, 4], &vec![2; 3], 2, 2, 2).is_err());
        validate_root_polyair_lengths(&[3, 2], &[2, 2, 2], 2, 2, 2).unwrap();

        validate_permutation_commitment_presence(true, &[0, 2]).unwrap();
        validate_permutation_commitment_presence(false, &[0, 0]).unwrap();
        assert!(validate_permutation_commitment_presence(false, &[0, 2]).is_err());
        assert!(validate_permutation_commitment_presence(true, &[0, 0]).is_err());
    }

    #[test]
    fn frozen_whir_preflight_rejects_missing_or_wrong_degree_groups() {
        require_degree_two_rounds("test", &[3, 3], 2).unwrap();
        assert!(require_degree_two_rounds("test", &[3], 2).is_err());
        assert!(require_degree_two_rounds("test", &[3, 2], 2).is_err());

        require_exact_group_lengths("test", &[8, 8, 8], 3, 8).unwrap();
        assert!(require_exact_group_lengths("test", &[8, 8], 3, 8).is_err());
        assert!(require_exact_group_lengths("test", &[8, 8, 6], 3, 8).is_err());
    }

    #[test]
    fn full_root_input_batch_preflight_rejects_legacy_elision() {
        for (has_permutation_commitment, full_batches) in [(false, 2), (true, 3)] {
            require_full_root_input_opening_batches(
                full_batches,
                has_permutation_commitment,
                full_batches,
                full_batches,
                full_batches,
            )
            .unwrap();

            let elided_batches = full_batches - 1;
            let error = require_full_root_input_opening_batches(
                full_batches,
                has_permutation_commitment,
                elided_batches,
                elided_batches,
                elided_batches,
            )
            .expect_err("an elided first input-opening batch must fail closed");
            assert!(error.to_string().contains("does not carry every input-opening batch"));
        }
    }

    #[test]
    fn full_root_input_batch_preflight_rejects_inconsistent_batch_vectors() {
        for lengths in [[2, 3, 3], [3, 2, 3], [3, 3, 2], [4, 3, 3]] {
            assert!(require_full_root_input_opening_batches(
                3, true, lengths[0], lengths[1], lengths[2],
            )
            .is_err());
        }
        assert!(require_full_root_input_opening_batches(2, true, 3, 3, 3).is_err());
    }
}
