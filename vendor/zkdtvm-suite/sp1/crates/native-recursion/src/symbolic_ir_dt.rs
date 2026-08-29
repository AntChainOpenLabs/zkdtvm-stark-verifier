use dt_stark::air::{InteractionScope, PairCol, PolyAirExtendable};
use p3_field::AbstractField;
use serde::{
    de::Error as _, ser::SerializeStruct, Deserialize, Deserializer, Serialize, Serializer,
};
use std::{
    ops::Deref,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

use crate::Instant;

use crate::{
    config::{D_EF, EF, F},
    symbolic_expr_adapter_dt::{
        classify_extension_constant, op_mix, CanonicalExtensionConstant, RecursionAdapterError,
        RecursionOpMix, RecursionPolyAirLeaf, RecursionPolyAirNode, RecursionPolyAirOp,
    },
    symbolic_expr_fixed_dt::{
        RecursionChildRole, RecursionFixedSymbolicChip, RecursionFixedSymbolicProgram,
    },
};

static NEXT_PROGRAM_AUTHORITY_IDENTITY: AtomicU64 = AtomicU64::new(1);

fn next_program_authority_identity() -> Result<u64, RecursionPolyAirProgramError> {
    NEXT_PROGRAM_AUTHORITY_IDENTITY
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |identity| identity.checked_add(1))
        .map_err(|_| {
            RecursionPolyAirProgramError::InvalidProgram(
                "recursion program authority identity exhausted".to_string(),
            )
        })
}

/// Mutable construction and serialization form. It never crosses into proving code directly:
/// callers must pass it through [`RecursionPolyAirVerifierProgram::try_from_dto`], which validates
/// every allocation/index bound before compiling the static plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecursionPolyAirVerifierProgramDto {
    pub version: u32,
    pub role: RecursionChildRole,
    pub artifact_digest: [F; crate::config::DIGEST_SIZE],
    pub chips: Vec<RecursionPolyAirChipIr>,
    pub max_required_beta_power: usize,
}

/// The IR and its static plan have one owner and therefore one lifetime. The fields are readable
/// through `Deref`, but no mutable reference to this value is ever exposed.
#[derive(Debug)]
pub struct FrozenConstraintProgram {
    pub version: u32,
    pub role: RecursionChildRole,
    pub artifact_digest: [F; crate::config::DIGEST_SIZE],
    pub chips: Vec<RecursionPolyAirChipIr>,
    pub max_required_beta_power: usize,
    pub(crate) authority_identity: u64,
    pub(crate) constraint_static_plan:
        Arc<crate::constraint_replay_dt::trace::ConstraintProgramPlan>,
    pub(crate) verified_child_layouts: Box<[crate::child_views::VerifiedChildLayout]>,
}

/// Immutable handle used by AIRs, finalized records and tracegen input. Cloning only clones this
/// `Arc`; editing a DTO and freezing it necessarily creates a new IR/plan authority.
#[derive(Debug, Clone)]
pub struct RecursionPolyAirVerifierProgram {
    frozen: Arc<FrozenConstraintProgram>,
}

impl Deref for RecursionPolyAirVerifierProgram {
    type Target = FrozenConstraintProgram;

    fn deref(&self) -> &Self::Target {
        &self.frozen
    }
}

impl PartialEq for RecursionPolyAirVerifierProgram {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.frozen, &other.frozen)
    }
}

impl Eq for RecursionPolyAirVerifierProgram {}

impl Serialize for RecursionPolyAirVerifierProgram {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Keep the existing five-field wire order. Runtime authority/plan state is deliberately
        // absent, so decoding always validates and creates a fresh frozen authority.
        let mut state = serializer.serialize_struct("RecursionPolyAirVerifierProgram", 5)?;
        state.serialize_field("version", &self.version)?;
        state.serialize_field("role", &self.role)?;
        state.serialize_field("artifact_digest", &self.artifact_digest)?;
        state.serialize_field("chips", &self.chips)?;
        state.serialize_field("max_required_beta_power", &self.max_required_beta_power)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for RecursionPolyAirVerifierProgram {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let dto = RecursionPolyAirVerifierProgramDto::deserialize(deserializer)?;
        Self::try_from_dto(dto).map_err(|err| D::Error::custom(format!("{err:?}")))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecursionPolyAirChipIr {
    pub static_chip_id: usize,
    pub chip_name: String,
    pub widths: RecursionPolyAirWidths,
    pub commit_scope: InteractionScope,
    pub logup_batch_size: usize,
    pub reserved_poly: Vec<PairCol>,
    pub derived_roots: Vec<RecursionPolyAirDerivedRoot>,
    pub gate_roots: Vec<RecursionPolyAirConstraintRoot>,
    pub lookup_multiplicity_roots: Vec<RecursionPolyAirLookupRoot>,
    pub node_table: Vec<RecursionPolyAirNode>,
    pub num_constraints_from_builder: usize,
    pub cost_ledger: RecursionD0CostLedger,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecursionPolyAirWidths {
    pub preprocessed: usize,
    pub main: usize,
    pub public: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecursionPolyAirDerivedRoot {
    BetaPower { power: usize },
    BetaSeptix,
    ReservedPoly { index: usize, source: PairCol },
    PrecomputeLc { index: usize, root_node_id: u32 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecursionEvaluatedDerivedRoot {
    pub root: RecursionPolyAirDerivedRoot,
    pub value: EF,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecursionPolyAirConstraintRoot {
    pub static_chip_id: usize,
    pub gate_idx: usize,
    pub root_node_id: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecursionPolyAirLookupRoot {
    pub lookup_idx: usize,
    pub root_node_id: u32,
    pub is_send: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecursionD0CostLedger {
    pub node_count: usize,
    pub op_mix: RecursionOpMix,
    pub gate_count: usize,
    pub precompute_root_count: usize,
    pub derived_root_count: usize,
    pub expected_node_bus_rows: usize,
    pub expected_wide_unroll_rows: usize,
    pub expected_wide_unroll_width: usize,
    pub internal_recursion_interactions_node_bus: usize,
    pub internal_recursion_interactions_wide_unroll: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecursionStaticChipBinding {
    pub proof_idx: usize,
    pub chip_idx: usize,
    pub static_chip_id: usize,
}

#[derive(Debug, Clone)]
pub struct RecursionPolyAirEnv<'a> {
    pub proof_idx: usize,
    pub chip_idx: usize,
    pub opened_preprocessed: &'a [EF],
    pub opened_main: &'a [EF],
    pub public_values: &'a [F],
    pub constraint_alpha: EF,
    pub perm_alpha: EF,
    pub perm_beta: EF,
    pub beta_powers: &'a [EF],
    pub beta_septix: EF,
    pub precomputed_lc: &'a [EF],
    pub reserved_poly: &'a [EF],
    pub is_first_row: EF,
    pub is_last_row: EF,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecursionPolyAirLookupBatchEval {
    pub batch_idx: usize,
    pub denominator: EF,
    pub numerator: EF,
    pub permutation_value: EF,
    pub constraint_value: EF,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecursionPolyAirChipEval {
    /// One case-local topological evaluation. Every downstream constraint consumer indexes this
    /// arena instead of replaying the symbolic DAG.
    pub node_values: Vec<EF>,
    pub precomputed_lc: Vec<EF>,
    pub reserved_poly: Vec<EF>,
    pub gate_values: Vec<EF>,
    pub signed_lookup_multiplicities: Vec<EF>,
    pub lookup_batches: Vec<RecursionPolyAirLookupBatchEval>,
    pub accumulator: EF,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RecursionPolyAirNodeEvalProfile {
    pub precompute_nodes: usize,
    pub remaining_nodes: usize,
    pub precompute_us: u64,
    pub remaining_us: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecursionPolyAirProgramError {
    Adapter(RecursionAdapterError),
    InvalidProgram(String),
}

impl From<RecursionAdapterError> for RecursionPolyAirProgramError {
    fn from(value: RecursionAdapterError) -> Self {
        Self::Adapter(value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecursionPolyAirEvaluationError {
    InvalidNodeId { node_id: u32 },
    MissingNodeValue { node_id: u32 },
    PreprocessedIndexOutOfRange { index: usize, len: usize },
    MainIndexOutOfRange { index: usize, len: usize },
    PublicIndexOutOfRange { index: usize, len: usize },
    BetaPowerIndexOutOfRange { index: usize, len: usize },
    PrecomputedIndexOutOfRange { index: usize, len: usize },
    ReservedPolyIndexOutOfRange { index: usize, len: usize },
    ChipIndexOutOfRange { chip_idx: usize, len: usize },
    ChipNameMismatch { expected: String, actual: String },
    WidthMismatch { field: &'static str, expected: usize, actual: usize },
    ConstraintCountMismatch { expected: usize, actual: usize },
    PrecomputeRootIndexOutOfRange { index: usize, len: usize },
    LookupBatchSizeZero,
    LookupDenominatorPartitionMismatch { precompute_roots: usize, lookup_roots: usize },
    PermutationIndexOutOfRange { index: usize, len: usize },
    NonCanonicalExtensionConstant,
}

impl RecursionPolyAirVerifierProgram {
    pub fn try_new(
        version: u32,
        role: RecursionChildRole,
        artifact_digest: [F; crate::config::DIGEST_SIZE],
        chips: Vec<RecursionPolyAirChipIr>,
        max_required_beta_power: usize,
    ) -> Result<Self, RecursionPolyAirProgramError> {
        Self::try_from_dto(RecursionPolyAirVerifierProgramDto {
            version,
            role,
            artifact_digest,
            chips,
            max_required_beta_power,
        })
    }

    pub fn try_from_dto(
        dto: RecursionPolyAirVerifierProgramDto,
    ) -> Result<Self, RecursionPolyAirProgramError> {
        validate_constraint_program_dto(&dto)
            .map_err(RecursionPolyAirProgramError::InvalidProgram)?;
        let verified_child_layouts = crate::child_views::VerifiedChildLayout::compile_all(&dto)
            .map_err(RecursionPolyAirProgramError::InvalidProgram)?;
        let constraint_static_plan =
            crate::constraint_replay_dt::trace::ConstraintProgramPlan::compile(&dto)
                .map_err(RecursionPolyAirProgramError::InvalidProgram)?;
        Ok(Self {
            frozen: Arc::new(FrozenConstraintProgram {
                version: dto.version,
                role: dto.role,
                artifact_digest: dto.artifact_digest,
                chips: dto.chips,
                max_required_beta_power: dto.max_required_beta_power,
                authority_identity: next_program_authority_identity()?,
                constraint_static_plan: Arc::new(constraint_static_plan),
                verified_child_layouts,
            }),
        })
    }

    pub fn to_dto(&self) -> RecursionPolyAirVerifierProgramDto {
        RecursionPolyAirVerifierProgramDto {
            version: self.version,
            role: self.role,
            artifact_digest: self.artifact_digest,
            chips: self.chips.clone(),
            max_required_beta_power: self.max_required_beta_power,
        }
    }

    pub(crate) fn authority_identity(&self) -> u64 {
        self.authority_identity
    }

    #[cfg(test)]
    pub(crate) fn shares_authority_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.frozen, &other.frozen)
    }

    pub(crate) fn verified_child_layout(
        &self,
        static_chip_id_offset: usize,
    ) -> Option<&crate::child_views::VerifiedChildLayout> {
        self.verified_child_layouts
            .iter()
            .find(|layout| layout.static_chip_id_offset() == static_chip_id_offset)
    }

    pub fn compile(
        fixed: &RecursionFixedSymbolicProgram,
    ) -> Result<Self, RecursionPolyAirProgramError> {
        let chips = fixed
            .chips
            .iter()
            .map(RecursionPolyAirChipIr::compile)
            .collect::<Result<Vec<_>, _>>()?;
        Self::try_from_dto(RecursionPolyAirVerifierProgramDto {
            version: fixed.version,
            role: fixed.role,
            artifact_digest: fixed.artifact_digest,
            chips,
            max_required_beta_power: fixed.max_required_beta_power,
        })
    }
}

pub(crate) const CONSTRAINT_PROGRAM_SCHEMA_VERSION: u32 =
    dt_stark::global_d11::GLOBAL146_CONSTRAINT_PROGRAM_SCHEMA_VERSION;
pub(crate) const CONSTRAINT_PROGRAM_LOGUP_BATCH_SIZE: usize = 2;
const MAX_CONSTRAINT_STATIC_CHIP_ID: usize = 255;
const MAX_CONSTRAINT_CHIPS: usize = 256;
const MAX_CONSTRAINT_NODES_PER_CHIP: usize = 1 << 24;
const MAX_CONSTRAINT_NODES_TOTAL: usize = 1 << 26;
const MAX_CONSTRAINT_WIDTH: usize = 1 << 20;
const MAX_CONSTRAINT_ROOTS_PER_CHIP: usize = 1 << 24;

/// Bounded validation required before `ConstraintProgramPlan::compile` is allowed to reserve or
/// index from externally supplied counts. Small validation workspaces are themselves bounded and
/// use fallible allocation where their size follows the decoded program.
pub(crate) fn validate_constraint_program_dto(
    program: &RecursionPolyAirVerifierProgramDto,
) -> Result<(), String> {
    use crate::constraint_replay_dt::CONSTRAINT_MAX_BETA_POWERS;
    use std::collections::BTreeSet;

    if program.version != CONSTRAINT_PROGRAM_SCHEMA_VERSION {
        return Err(format!(
            "unsupported constraint program version {}, expected {}",
            program.version, CONSTRAINT_PROGRAM_SCHEMA_VERSION
        ));
    }
    if program.chips.len() > MAX_CONSTRAINT_CHIPS {
        return Err(format!(
            "constraint program has {} chips, maximum is {MAX_CONSTRAINT_CHIPS}",
            program.chips.len()
        ));
    }
    if !program.chips.windows(2).all(|pair| pair[0].static_chip_id < pair[1].static_chip_id) {
        return Err("constraint program static chip ids must be unique and sorted".to_string());
    }

    let mut total_nodes = 0usize;
    let mut max_beta_power = 0usize;
    let mut names_by_segment = [BTreeSet::new(), BTreeSet::new()];
    for chip in &program.chips {
        if chip.static_chip_id > MAX_CONSTRAINT_STATIC_CHIP_ID {
            return Err(format!(
                "static chip id {} is outside supported base-0/base-128 segments",
                chip.static_chip_id
            ));
        }
        let segment = usize::from(chip.static_chip_id >= 128);
        if !names_by_segment[segment].insert(chip.chip_name.as_str()) {
            return Err(format!(
                "duplicate chip name {:?} in replay segment {}",
                chip.chip_name,
                segment * 128
            ));
        }
        if chip.node_table.len() > MAX_CONSTRAINT_NODES_PER_CHIP {
            return Err(format!(
                "constraint chip {} has {} nodes, maximum is {MAX_CONSTRAINT_NODES_PER_CHIP}",
                chip.static_chip_id,
                chip.node_table.len()
            ));
        }
        total_nodes = total_nodes
            .checked_add(chip.node_table.len())
            .ok_or_else(|| "constraint node count overflow".to_string())?;
        if total_nodes > MAX_CONSTRAINT_NODES_TOTAL {
            return Err(format!(
                "constraint program has {total_nodes} nodes, maximum is {MAX_CONSTRAINT_NODES_TOTAL}"
            ));
        }
        let width_sum = chip
            .widths
            .preprocessed
            .checked_add(chip.widths.main)
            .and_then(|sum| sum.checked_add(chip.widths.public))
            .ok_or_else(|| format!("constraint chip {} width sum overflow", chip.static_chip_id))?;
        if chip.widths.preprocessed > MAX_CONSTRAINT_WIDTH ||
            chip.widths.main > MAX_CONSTRAINT_WIDTH ||
            chip.widths.public > MAX_CONSTRAINT_WIDTH ||
            width_sum > MAX_CONSTRAINT_WIDTH
        {
            return Err(format!(
                "constraint chip {} widths exceed supported bound {MAX_CONSTRAINT_WIDTH}",
                chip.static_chip_id
            ));
        }
        if chip.logup_batch_size != CONSTRAINT_PROGRAM_LOGUP_BATCH_SIZE {
            return Err(format!(
                "constraint chip {} has logup batch size {}, shipped native recursion requires {}",
                chip.static_chip_id, chip.logup_batch_size, CONSTRAINT_PROGRAM_LOGUP_BATCH_SIZE,
            ));
        }
        for source in &chip.reserved_poly {
            validate_pair_col(*source, chip)?;
        }
        if chip.gate_roots.len() > MAX_CONSTRAINT_ROOTS_PER_CHIP ||
            chip.lookup_multiplicity_roots.len() > MAX_CONSTRAINT_ROOTS_PER_CHIP ||
            chip.derived_roots.len() > MAX_CONSTRAINT_ROOTS_PER_CHIP ||
            chip.reserved_poly.len() > MAX_CONSTRAINT_WIDTH ||
            chip.cost_ledger.precompute_root_count > chip.node_table.len()
        {
            return Err(format!("constraint chip {} root count exceeds bound", chip.static_chip_id));
        }
        let mut precompute_roots = Vec::new();
        precompute_roots.try_reserve_exact(chip.cost_ledger.precompute_root_count).map_err(
            |_| {
                format!(
                    "constraint chip {} precompute root allocation rejected",
                    chip.static_chip_id
                )
            },
        )?;
        precompute_roots.resize(chip.cost_ledger.precompute_root_count, None);
        let mut reserved_seen = vec![false; chip.reserved_poly.len()];
        let chip_max_beta_power = chip.max_beta_power_from_roots()?;
        if chip_max_beta_power >= CONSTRAINT_MAX_BETA_POWERS {
            return Err(format!("beta power {chip_max_beta_power} exceeds supported bound"));
        }
        let mut beta_seen = vec![
            false;
            chip_max_beta_power
                .checked_add(1)
                .ok_or_else(|| "beta root count overflow".to_string())?
        ];
        let mut beta_septix = 0usize;
        for root in &chip.derived_roots {
            match root {
                RecursionPolyAirDerivedRoot::BetaPower { power } => {
                    if *power >= CONSTRAINT_MAX_BETA_POWERS {
                        return Err(format!("beta power {power} exceeds supported bound"));
                    }
                    let seen = beta_seen.get_mut(*power).ok_or_else(|| {
                        format!("constraint chip {} beta roots are not dense", chip.static_chip_id)
                    })?;
                    if core::mem::replace(seen, true) {
                        return Err(format!("duplicate beta power root {power}"));
                    }
                    max_beta_power = max_beta_power.max(*power);
                }
                RecursionPolyAirDerivedRoot::BetaSeptix => beta_septix += 1,
                RecursionPolyAirDerivedRoot::ReservedPoly { index, source } => {
                    if chip.reserved_poly.get(*index) != Some(source) ||
                        reserved_seen
                            .get_mut(*index)
                            .is_none_or(|seen| core::mem::replace(seen, true))
                    {
                        return Err(format!("invalid reserved root index {index}"));
                    }
                }
                RecursionPolyAirDerivedRoot::PrecomputeLc { index, root_node_id } => {
                    validate_root_node(*root_node_id, chip, "precompute")?;
                    let slot = precompute_roots.get_mut(*index).ok_or_else(|| {
                        format!("precompute root index {index} is outside declared count")
                    })?;
                    if slot.replace(*root_node_id).is_some() {
                        return Err(format!("duplicate precompute root index {index}"));
                    }
                }
            }
        }
        if beta_seen.iter().any(|seen| !seen) ||
            reserved_seen.iter().any(|seen| !seen) ||
            precompute_roots.iter().any(Option::is_none) ||
            beta_septix != 1
        {
            return Err(format!(
                "constraint chip {} has missing reserved/precompute/challenge roots",
                chip.static_chip_id
            ));
        }
        if chip.lookup_multiplicity_roots.len() > precompute_roots.len() {
            return Err(format!(
                "constraint chip {} has more lookup roots than precompute roots",
                chip.static_chip_id
            ));
        }
        for (expected, root) in chip.gate_roots.iter().enumerate() {
            if root.static_chip_id != chip.static_chip_id || root.gate_idx != expected {
                return Err(format!(
                    "constraint chip {} has malformed gate roots",
                    chip.static_chip_id
                ));
            }
            validate_root_node(root.root_node_id, chip, "gate")?;
        }
        for (expected, root) in chip.lookup_multiplicity_roots.iter().enumerate() {
            if root.lookup_idx != expected {
                return Err(format!(
                    "constraint chip {} has non-dense lookup roots",
                    chip.static_chip_id
                ));
            }
            validate_root_node(root.root_node_id, chip, "lookup")?;
        }

        for (expected_id, node) in chip.node_table.iter().enumerate() {
            if usize::try_from(node.node_id).ok() != Some(expected_id) {
                return Err(format!(
                    "constraint chip {} node ids are not dense at position {expected_id}",
                    chip.static_chip_id
                ));
            }
            validate_node_op(chip, node)?;
        }
        validate_cost_ledger(chip)?;
    }
    if program.max_required_beta_power != max_beta_power ||
        program.max_required_beta_power >= CONSTRAINT_MAX_BETA_POWERS
    {
        return Err(format!(
            "program max beta power {} does not match bounded IR maximum {max_beta_power}",
            program.max_required_beta_power
        ));
    }
    Ok(())
}

trait ChipValidationExt {
    fn max_beta_power_from_roots(&self) -> Result<usize, String>;
}

impl ChipValidationExt for RecursionPolyAirChipIr {
    fn max_beta_power_from_roots(&self) -> Result<usize, String> {
        self.derived_roots
            .iter()
            .filter_map(|root| match root {
                RecursionPolyAirDerivedRoot::BetaPower { power } => Some(*power),
                _ => None,
            })
            .max()
            .ok_or_else(|| {
                format!("constraint chip {} has no beta-power roots", self.static_chip_id)
            })
    }
}

fn validate_root_node(
    node_id: u32,
    chip: &RecursionPolyAirChipIr,
    kind: &str,
) -> Result<(), String> {
    if usize::try_from(node_id).ok().is_none_or(|id| id >= chip.node_table.len()) {
        return Err(format!(
            "constraint chip {} {kind} root node {node_id} is out of bounds",
            chip.static_chip_id
        ));
    }
    Ok(())
}

fn validate_pair_col(source: PairCol, chip: &RecursionPolyAirChipIr) -> Result<(), String> {
    let valid = match source {
        PairCol::Prep(index) => index < chip.widths.preprocessed,
        PairCol::Main(index) => index < chip.widths.main,
    };
    if !valid {
        return Err(format!(
            "constraint chip {} reserved source {source:?} is out of bounds",
            chip.static_chip_id
        ));
    }
    Ok(())
}

fn validate_node_op(
    chip: &RecursionPolyAirChipIr,
    node: &RecursionPolyAirNode,
) -> Result<(), String> {
    let node_id =
        usize::try_from(node.node_id).map_err(|_| "node id conversion failed".to_string())?;
    let operand = |id: u32| -> Result<(), String> {
        if usize::try_from(id).ok().is_none_or(|id| id >= node_id) {
            Err(format!(
                "constraint chip {} node {} has non-topological operand {id}",
                chip.static_chip_id, node.node_id
            ))
        } else {
            Ok(())
        }
    };
    match &node.op {
        RecursionPolyAirOp::Leaf(leaf) => match leaf {
            RecursionPolyAirLeaf::Preprocessed { col } if *col >= chip.widths.preprocessed => {
                Err(format!("preprocessed leaf {col} is out of bounds"))
            }
            RecursionPolyAirLeaf::Main { col } if *col >= chip.widths.main => {
                Err(format!("main leaf {col} is out of bounds"))
            }
            RecursionPolyAirLeaf::Public { index } if *index >= chip.widths.public => {
                Err(format!("public leaf {index} is out of bounds"))
            }
            RecursionPolyAirLeaf::BetaPower { power }
                if *power >= crate::constraint_replay_dt::CONSTRAINT_MAX_BETA_POWERS =>
            {
                Err(format!("beta-power leaf {power} is out of bounds"))
            }
            RecursionPolyAirLeaf::ReservedPoly { index } => {
                chip.reserved_poly
                    .get(*index)
                    .ok_or_else(|| format!("reserved-poly leaf {index} has no reserved source"))?;
                Ok(())
            }
            RecursionPolyAirLeaf::Precomputed { index } => {
                let root_node = chip
                    .derived_roots
                    .iter()
                    .find_map(|root| match root {
                        RecursionPolyAirDerivedRoot::PrecomputeLc {
                            index: root_index,
                            root_node_id,
                        } if root_index == index => Some(*root_node_id),
                        _ => None,
                    })
                    .ok_or_else(|| format!("precomputed leaf {index} has no root"))?;
                operand(root_node)
            }
            _ => Ok(()),
        },
        RecursionPolyAirOp::ConstBase(_) => Ok(()),
        RecursionPolyAirOp::ConstExt(value) => {
            if classify_extension_constant(value) == CanonicalExtensionConstant::Theta {
                Ok(())
            } else {
                Err(format!(
                    "constraint chip {} node {} uses non-canonical extension constant",
                    chip.static_chip_id, node.node_id
                ))
            }
        }
        RecursionPolyAirOp::Add { lhs, rhs } |
        RecursionPolyAirOp::Sub { lhs, rhs } |
        RecursionPolyAirOp::Mul { lhs, rhs } => {
            operand(*lhs)?;
            operand(*rhs)
        }
        RecursionPolyAirOp::FusedMulAdd { lhs, rhs, addend, .. } => {
            operand(*lhs)?;
            operand(*rhs)?;
            operand(*addend)
        }
        RecursionPolyAirOp::Neg { .. } => Err(format!(
            "constraint chip {} node {} uses unsupported program-table op {:?}",
            chip.static_chip_id, node.node_id, node.op
        )),
    }
}

fn validate_cost_ledger(chip: &RecursionPolyAirChipIr) -> Result<(), String> {
    let mut mix = RecursionOpMix::default();
    for node in &chip.node_table {
        mix.observe(&node.op);
    }
    let ledger = chip.cost_ledger;
    let expected_wide_width = chip
        .node_table
        .len()
        .checked_mul(D_EF)
        .ok_or_else(|| "constraint wide-unroll width overflow".to_string())?;
    if ledger.node_count != chip.node_table.len() ||
        ledger.op_mix != mix ||
        ledger.gate_count != chip.gate_roots.len() ||
        ledger.derived_root_count != chip.derived_roots.len() ||
        ledger.expected_node_bus_rows != chip.node_table.len() ||
        ledger.expected_wide_unroll_rows != 1 ||
        ledger.expected_wide_unroll_width != expected_wide_width ||
        ledger.internal_recursion_interactions_node_bus != chip.node_table.len() ||
        ledger.internal_recursion_interactions_wide_unroll != 0
    {
        return Err(format!("constraint chip {} has inconsistent cost ledger", chip.static_chip_id));
    }
    Ok(())
}

impl RecursionPolyAirChipIr {
    pub fn compile(
        fixed: &RecursionFixedSymbolicChip,
    ) -> Result<Self, RecursionPolyAirProgramError> {
        let adapted_roots = &fixed.adapted_roots;
        let mut derived_roots = Vec::new();

        for power in 0..=fixed.builder_snapshot.required_max_beta_power {
            derived_roots.push(RecursionPolyAirDerivedRoot::BetaPower { power });
        }
        derived_roots.push(RecursionPolyAirDerivedRoot::BetaSeptix);
        for (index, source) in fixed.reserved_poly.iter().copied().enumerate() {
            derived_roots.push(RecursionPolyAirDerivedRoot::ReservedPoly { index, source });
        }

        for root in &adapted_roots.precompute_roots {
            derived_roots.push(RecursionPolyAirDerivedRoot::PrecomputeLc {
                index: root.root_index,
                root_node_id: root.root_node_id,
            });
        }

        let mut lookup_multiplicity_roots = Vec::new();
        for (root, lookup) in adapted_roots
            .lookup_multiplicity_roots
            .iter()
            .zip(fixed.builder_snapshot.lookup_is_send.iter())
        {
            lookup_multiplicity_roots.push(RecursionPolyAirLookupRoot {
                lookup_idx: root.root_index,
                root_node_id: root.root_node_id,
                is_send: *lookup,
            });
        }

        let mut gate_roots = Vec::new();
        for root in &adapted_roots.gate_roots {
            gate_roots.push(RecursionPolyAirConstraintRoot {
                static_chip_id: fixed.static_chip_id,
                gate_idx: root.root_index,
                root_node_id: root.root_node_id,
            });
        }

        let widths = RecursionPolyAirWidths {
            preprocessed: fixed.preprocessed_width,
            main: fixed.main_width,
            public: fixed.public_width,
        };
        let cost_ledger = RecursionD0CostLedger::from_parts(
            &adapted_roots.node_table,
            gate_roots.len(),
            fixed.builder_snapshot.precompute_root_count,
            derived_roots.len(),
        );

        Ok(Self {
            static_chip_id: fixed.static_chip_id,
            chip_name: fixed.chip_name.clone(),
            widths,
            commit_scope: fixed.commit_scope,
            logup_batch_size: fixed.logup_batch_size,
            reserved_poly: fixed.reserved_poly.clone(),
            derived_roots,
            gate_roots,
            lookup_multiplicity_roots,
            node_table: adapted_roots.node_table.clone(),
            num_constraints_from_builder: fixed.num_constraints_from_builder,
            cost_ledger,
        })
    }

    pub fn bind_observed_chip(
        &self,
        proof_idx: usize,
        chip_idx: usize,
        chip_name: &str,
        widths: RecursionPolyAirWidths,
        constraints_count: usize,
    ) -> Result<RecursionStaticChipBinding, RecursionPolyAirEvaluationError> {
        if self.chip_name != chip_name {
            return Err(RecursionPolyAirEvaluationError::ChipNameMismatch {
                expected: self.chip_name.clone(),
                actual: chip_name.to_string(),
            });
        }
        check_width("preprocessed", self.widths.preprocessed, widths.preprocessed)?;
        check_width("main", self.widths.main, widths.main)?;
        check_width("public", self.widths.public, widths.public)?;
        if self.num_constraints_from_builder != constraints_count {
            return Err(RecursionPolyAirEvaluationError::ConstraintCountMismatch {
                expected: self.num_constraints_from_builder,
                actual: constraints_count,
            });
        }
        Ok(RecursionStaticChipBinding { proof_idx, chip_idx, static_chip_id: self.static_chip_id })
    }
}

impl RecursionD0CostLedger {
    fn from_parts(
        node_table: &[RecursionPolyAirNode],
        gate_count: usize,
        precompute_root_count: usize,
        derived_root_count: usize,
    ) -> Self {
        let node_count = node_table.len();
        Self {
            node_count,
            op_mix: op_mix(node_table),
            gate_count,
            precompute_root_count,
            derived_root_count,
            expected_node_bus_rows: node_count,
            expected_wide_unroll_rows: 1,
            expected_wide_unroll_width: node_count * D_EF,
            internal_recursion_interactions_node_bus: node_count,
            internal_recursion_interactions_wide_unroll: 0,
        }
    }
}

pub fn evaluate_node_table(
    chip: &RecursionPolyAirChipIr,
    env: &RecursionPolyAirEnv<'_>,
) -> Result<Vec<EF>, RecursionPolyAirEvaluationError> {
    evaluate_node_prefix(chip, env, chip.node_table.len())
}

pub fn evaluate_derived_roots(
    chip: &RecursionPolyAirChipIr,
    env: &RecursionPolyAirEnv<'_>,
) -> Result<Vec<RecursionEvaluatedDerivedRoot>, RecursionPolyAirEvaluationError> {
    let node_count = max_precompute_root_node(chip).map_or(0, |node_id| node_id as usize + 1);
    let node_values = evaluate_node_prefix(chip, env, node_count)?;
    chip.derived_roots
        .iter()
        .map(|root| {
            Ok(RecursionEvaluatedDerivedRoot {
                root: root.clone(),
                value: evaluate_derived_root(root, &node_values, env)?,
            })
        })
        .collect()
}

pub fn evaluate_precomputed_lc(
    chip: &RecursionPolyAirChipIr,
    env: &RecursionPolyAirEnv<'_>,
) -> Result<Vec<EF>, RecursionPolyAirEvaluationError> {
    let expected = chip.cost_ledger.precompute_root_count;
    let node_count = max_precompute_root_node(chip).map_or(0, |node_id| node_id as usize + 1);
    let node_values = evaluate_node_prefix(chip, env, node_count)?;
    let mut precomputed_lc = vec![EF::zero(); expected];
    for root in &chip.derived_roots {
        if let RecursionPolyAirDerivedRoot::PrecomputeLc { index, root_node_id } = root {
            if *index >= expected {
                return Err(RecursionPolyAirEvaluationError::PrecomputeRootIndexOutOfRange {
                    index: *index,
                    len: expected,
                });
            }
            precomputed_lc[*index] = node_value(&node_values, *root_node_id)?;
        }
    }
    Ok(precomputed_lc)
}

pub fn evaluate_reserved_poly_values(
    chip: &RecursionPolyAirChipIr,
    env: &RecursionPolyAirEnv<'_>,
) -> Result<Vec<EF>, RecursionPolyAirEvaluationError> {
    chip.reserved_poly.iter().map(|source| evaluate_pair_col(*source, env)).collect()
}

pub fn evaluate_signed_lookup_multiplicities(
    chip: &RecursionPolyAirChipIr,
    env: &RecursionPolyAirEnv<'_>,
) -> Result<Vec<EF>, RecursionPolyAirEvaluationError> {
    let node_values = evaluate_node_table(chip, env)?;
    chip.lookup_multiplicity_roots
        .iter()
        .map(|root| {
            let value = node_value(&node_values, root.root_node_id)?;
            Ok(if root.is_send { value } else { -value })
        })
        .collect()
}

pub fn evaluate_lookup_batches(
    chip: &RecursionPolyAirChipIr,
    env: &RecursionPolyAirEnv<'_>,
    permutation_local: &[EF],
) -> Result<Vec<RecursionPolyAirLookupBatchEval>, RecursionPolyAirEvaluationError> {
    let multiplicities = evaluate_signed_lookup_multiplicities(chip, env)?;
    evaluate_lookup_batches_from_multiplicities(chip, env, permutation_local, &multiplicities)
}

pub fn evaluate_lookup_batches_from_multiplicities(
    chip: &RecursionPolyAirChipIr,
    env: &RecursionPolyAirEnv<'_>,
    permutation_local: &[EF],
    multiplicities: &[EF],
) -> Result<Vec<RecursionPolyAirLookupBatchEval>, RecursionPolyAirEvaluationError> {
    let precompute_len = env.precomputed_lc.len();
    let lookup_len = multiplicities.len();
    if precompute_len < lookup_len {
        return Err(RecursionPolyAirEvaluationError::LookupDenominatorPartitionMismatch {
            precompute_roots: precompute_len,
            lookup_roots: lookup_len,
        });
    }
    if chip.cost_ledger.precompute_root_count < chip.lookup_multiplicity_roots.len() {
        return Err(RecursionPolyAirEvaluationError::LookupDenominatorPartitionMismatch {
            precompute_roots: chip.cost_ledger.precompute_root_count,
            lookup_roots: chip.lookup_multiplicity_roots.len(),
        });
    }
    let batch_size = chip.logup_batch_size;
    if batch_size == 0 && !multiplicities.is_empty() {
        return Err(RecursionPolyAirEvaluationError::LookupBatchSizeZero);
    }

    let mut batches = Vec::new();
    for (batch_idx, (values, multiplicities)) in env.precomputed_lc[..lookup_len]
        .chunks(batch_size.max(1))
        .zip(multiplicities.chunks(batch_size.max(1)))
        .enumerate()
    {
        let denominator = values.iter().copied().product::<EF>();
        let mut numerator = EF::zero();
        for (i, multiplicity) in multiplicities.iter().copied().enumerate() {
            let all_but_current = values
                .iter()
                .enumerate()
                .filter_map(|(j, value)| (i != j).then_some(*value))
                .product::<EF>();
            numerator += all_but_current * multiplicity;
        }
        let permutation_value = get_index(
            permutation_local,
            batch_idx,
            RecursionPolyAirEvaluationError::PermutationIndexOutOfRange {
                index: batch_idx,
                len: permutation_local.len(),
            },
        )?;
        let constraint_value = numerator - denominator * permutation_value;
        batches.push(RecursionPolyAirLookupBatchEval {
            batch_idx,
            denominator,
            numerator,
            permutation_value,
            constraint_value,
        });
    }
    Ok(batches)
}

pub fn evaluate_chip_replay(
    chip: &RecursionPolyAirChipIr,
    base_env: &RecursionPolyAirEnv<'_>,
    permutation_local: &[EF],
) -> Result<RecursionPolyAirChipEval, RecursionPolyAirEvaluationError> {
    evaluate_chip_replay_profiled(chip, base_env, permutation_local).map(|(replay, _)| replay)
}

pub fn evaluate_chip_replay_profiled(
    chip: &RecursionPolyAirChipIr,
    base_env: &RecursionPolyAirEnv<'_>,
    permutation_local: &[EF],
) -> Result<
    (RecursionPolyAirChipEval, RecursionPolyAirNodeEvalProfile),
    RecursionPolyAirEvaluationError,
> {
    let (node_values, precomputed_lc, reserved_poly, node_profile) =
        evaluate_chip_node_arena_profiled(chip, base_env)?;
    let env = RecursionPolyAirEnv {
        proof_idx: base_env.proof_idx,
        chip_idx: base_env.chip_idx,
        opened_preprocessed: base_env.opened_preprocessed,
        opened_main: base_env.opened_main,
        public_values: base_env.public_values,
        constraint_alpha: base_env.constraint_alpha,
        perm_alpha: base_env.perm_alpha,
        perm_beta: base_env.perm_beta,
        beta_powers: base_env.beta_powers,
        beta_septix: base_env.beta_septix,
        precomputed_lc: &precomputed_lc,
        reserved_poly: &reserved_poly,
        is_first_row: base_env.is_first_row,
        is_last_row: base_env.is_last_row,
    };
    let gate_values = chip
        .gate_roots
        .iter()
        .map(|root| node_value(&node_values, root.root_node_id))
        .collect::<Result<Vec<_>, _>>()?;
    let signed_lookup_multiplicities = chip
        .lookup_multiplicity_roots
        .iter()
        .map(|root| {
            let value = node_value(&node_values, root.root_node_id)?;
            Ok(if root.is_send { value } else { -value })
        })
        .collect::<Result<Vec<_>, RecursionPolyAirEvaluationError>>()?;
    let lookup_batches = evaluate_lookup_batches_from_multiplicities(
        chip,
        &env,
        permutation_local,
        &signed_lookup_multiplicities,
    )?;
    let mut accumulator = fold_gate_values(&gate_values, env.constraint_alpha);
    for batch in &lookup_batches {
        accumulator = accumulator * env.constraint_alpha + batch.constraint_value;
    }
    Ok((
        RecursionPolyAirChipEval {
            node_values,
            precomputed_lc,
            reserved_poly,
            gate_values,
            signed_lookup_multiplicities,
            lookup_batches,
            accumulator,
        },
        node_profile,
    ))
}

pub fn evaluate_chip_node_values(
    chip: &RecursionPolyAirChipIr,
    base_env: &RecursionPolyAirEnv<'_>,
) -> Result<Vec<EF>, RecursionPolyAirEvaluationError> {
    evaluate_chip_node_arena(chip, base_env).map(|(node_values, _, _)| node_values)
}

/// Evaluates the precompute prefix and the remaining symbolic DAG into one contiguous arena.
///
/// The old path evaluated the prefix, then restarted from node zero for every root consumer. The
/// compiler orders every precompute root before any `Precomputed` leaf that consumes it, so the
/// prefix can remain in the arena and evaluation resumes at the exact dependency barrier.
fn evaluate_chip_node_arena(
    chip: &RecursionPolyAirChipIr,
    base_env: &RecursionPolyAirEnv<'_>,
) -> Result<(Vec<EF>, Vec<EF>, Vec<EF>), RecursionPolyAirEvaluationError> {
    evaluate_chip_node_arena_profiled(chip, base_env).map(
        |(node_values, precomputed_lc, reserved_poly, _)| {
            (node_values, precomputed_lc, reserved_poly)
        },
    )
}

pub(crate) fn evaluate_chip_node_arena_profiled(
    chip: &RecursionPolyAirChipIr,
    base_env: &RecursionPolyAirEnv<'_>,
) -> Result<
    (Vec<EF>, Vec<EF>, Vec<EF>, RecursionPolyAirNodeEvalProfile),
    RecursionPolyAirEvaluationError,
> {
    let prefix_len = max_precompute_root_node(chip).map_or(0, |node_id| node_id as usize + 1);
    let precompute_started = Instant::now();
    let mut node_values = evaluate_node_prefix(chip, base_env, prefix_len)?;
    let precompute_us = elapsed_us(precompute_started);
    let mut precomputed_lc = vec![EF::zero(); chip.cost_ledger.precompute_root_count];
    for root in &chip.derived_roots {
        if let RecursionPolyAirDerivedRoot::PrecomputeLc { index, root_node_id } = root {
            if *index >= precomputed_lc.len() {
                return Err(RecursionPolyAirEvaluationError::PrecomputeRootIndexOutOfRange {
                    index: *index,
                    len: precomputed_lc.len(),
                });
            }
            precomputed_lc[*index] = node_value(&node_values, *root_node_id)?;
        }
    }
    let reserved_poly = evaluate_reserved_poly_values(chip, base_env)?;
    let env = RecursionPolyAirEnv {
        proof_idx: base_env.proof_idx,
        chip_idx: base_env.chip_idx,
        opened_preprocessed: base_env.opened_preprocessed,
        opened_main: base_env.opened_main,
        public_values: base_env.public_values,
        constraint_alpha: base_env.constraint_alpha,
        perm_alpha: base_env.perm_alpha,
        perm_beta: base_env.perm_beta,
        beta_powers: base_env.beta_powers,
        beta_septix: base_env.beta_septix,
        precomputed_lc: &precomputed_lc,
        reserved_poly: &reserved_poly,
        is_first_row: base_env.is_first_row,
        is_last_row: base_env.is_last_row,
    };
    let remaining_started = Instant::now();
    for node in chip.node_table.iter().skip(prefix_len) {
        let expected_id = node_values.len() as u32;
        if node.node_id != expected_id {
            return Err(RecursionPolyAirEvaluationError::InvalidNodeId { node_id: node.node_id });
        }
        node_values.push(evaluate_node_op(&node.op, &node_values, &env)?);
    }
    let remaining_us = elapsed_us(remaining_started);
    Ok((
        node_values,
        precomputed_lc,
        reserved_poly,
        RecursionPolyAirNodeEvalProfile {
            precompute_nodes: prefix_len,
            remaining_nodes: chip.node_table.len().saturating_sub(prefix_len),
            precompute_us,
            remaining_us,
        },
    ))
}

fn elapsed_us(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX)
}

fn evaluate_node_prefix(
    chip: &RecursionPolyAirChipIr,
    env: &RecursionPolyAirEnv<'_>,
    count: usize,
) -> Result<Vec<EF>, RecursionPolyAirEvaluationError> {
    let mut values = Vec::with_capacity(chip.node_table.len());
    for node in chip.node_table.iter().take(count) {
        let expected_id = values.len() as u32;
        if node.node_id != expected_id {
            return Err(RecursionPolyAirEvaluationError::InvalidNodeId { node_id: node.node_id });
        }
        let value = evaluate_node_op(&node.op, &values, env)?;
        values.push(value);
    }
    Ok(values)
}

fn evaluate_derived_root(
    root: &RecursionPolyAirDerivedRoot,
    node_values: &[EF],
    env: &RecursionPolyAirEnv<'_>,
) -> Result<EF, RecursionPolyAirEvaluationError> {
    let value = match root {
        RecursionPolyAirDerivedRoot::BetaPower { power } => get_index(
            env.beta_powers,
            *power,
            RecursionPolyAirEvaluationError::BetaPowerIndexOutOfRange {
                index: *power,
                len: env.beta_powers.len(),
            },
        )?,
        RecursionPolyAirDerivedRoot::BetaSeptix => env.beta_septix,
        RecursionPolyAirDerivedRoot::ReservedPoly { source, .. } => {
            evaluate_pair_col(*source, env)?
        }
        RecursionPolyAirDerivedRoot::PrecomputeLc { root_node_id, .. } => {
            node_value(node_values, *root_node_id)?
        }
    };
    Ok(value)
}

fn evaluate_pair_col(
    source: PairCol,
    env: &RecursionPolyAirEnv<'_>,
) -> Result<EF, RecursionPolyAirEvaluationError> {
    match source {
        PairCol::Prep(index) => get_index(
            env.opened_preprocessed,
            index,
            RecursionPolyAirEvaluationError::PreprocessedIndexOutOfRange {
                index,
                len: env.opened_preprocessed.len(),
            },
        ),
        PairCol::Main(index) => get_index(
            env.opened_main,
            index,
            RecursionPolyAirEvaluationError::MainIndexOutOfRange {
                index,
                len: env.opened_main.len(),
            },
        ),
    }
}

fn max_precompute_root_node(chip: &RecursionPolyAirChipIr) -> Option<u32> {
    chip.derived_roots
        .iter()
        .filter_map(|root| match root {
            RecursionPolyAirDerivedRoot::PrecomputeLc { root_node_id, .. } => Some(*root_node_id),
            _ => None,
        })
        .max()
}

pub fn evaluate_gate_roots(
    chip: &RecursionPolyAirChipIr,
    env: &RecursionPolyAirEnv<'_>,
) -> Result<Vec<EF>, RecursionPolyAirEvaluationError> {
    let node_values = evaluate_node_table(chip, env)?;
    chip.gate_roots.iter().map(|root| node_value(&node_values, root.root_node_id)).collect()
}

pub fn fold_gate_values(values: &[EF], constraint_alpha: EF) -> EF {
    let mut acc = EF::zero();
    for value in values {
        acc = acc * constraint_alpha + *value;
    }
    acc
}

fn evaluate_node_op(
    op: &RecursionPolyAirOp,
    values: &[EF],
    env: &RecursionPolyAirEnv<'_>,
) -> Result<EF, RecursionPolyAirEvaluationError> {
    let value = match op {
        RecursionPolyAirOp::Leaf(leaf) => evaluate_leaf(leaf, env)?,
        RecursionPolyAirOp::ConstBase(value) => F::ef_from_base(*value),
        RecursionPolyAirOp::ConstExt(value) => {
            if classify_extension_constant(value) != CanonicalExtensionConstant::Theta {
                return Err(RecursionPolyAirEvaluationError::NonCanonicalExtensionConstant);
            }
            *value
        }
        RecursionPolyAirOp::Add { lhs, rhs } => {
            node_value(values, *lhs)? + node_value(values, *rhs)?
        }
        RecursionPolyAirOp::Sub { lhs, rhs } => {
            node_value(values, *lhs)? - node_value(values, *rhs)?
        }
        RecursionPolyAirOp::Neg { input } => -node_value(values, *input)?,
        RecursionPolyAirOp::Mul { lhs, rhs } => {
            node_value(values, *lhs)? * node_value(values, *rhs)?
        }
        RecursionPolyAirOp::FusedMulAdd { lhs, rhs, addend, sign } => {
            let product = node_value(values, *lhs)? * node_value(values, *rhs)?;
            let addend = node_value(values, *addend)?;
            if *sign {
                product - addend
            } else {
                product + addend
            }
        }
    };
    Ok(value)
}

fn evaluate_leaf(
    leaf: &RecursionPolyAirLeaf,
    env: &RecursionPolyAirEnv<'_>,
) -> Result<EF, RecursionPolyAirEvaluationError> {
    let value = match leaf {
        RecursionPolyAirLeaf::Preprocessed { col } => get_index(
            env.opened_preprocessed,
            *col,
            RecursionPolyAirEvaluationError::PreprocessedIndexOutOfRange {
                index: *col,
                len: env.opened_preprocessed.len(),
            },
        )?,
        RecursionPolyAirLeaf::Main { col } => get_index(
            env.opened_main,
            *col,
            RecursionPolyAirEvaluationError::MainIndexOutOfRange {
                index: *col,
                len: env.opened_main.len(),
            },
        )?,
        RecursionPolyAirLeaf::Public { index } => {
            let public = get_index(
                env.public_values,
                *index,
                RecursionPolyAirEvaluationError::PublicIndexOutOfRange {
                    index: *index,
                    len: env.public_values.len(),
                },
            )?;
            F::ef_from_base(public)
        }
        RecursionPolyAirLeaf::PermAlpha => env.perm_alpha,
        RecursionPolyAirLeaf::BetaPower { power } => get_index(
            env.beta_powers,
            *power,
            RecursionPolyAirEvaluationError::BetaPowerIndexOutOfRange {
                index: *power,
                len: env.beta_powers.len(),
            },
        )?,
        RecursionPolyAirLeaf::BetaSeptix => env.beta_septix,
        RecursionPolyAirLeaf::Precomputed { index, .. } => get_index(
            env.precomputed_lc,
            *index,
            RecursionPolyAirEvaluationError::PrecomputedIndexOutOfRange {
                index: *index,
                len: env.precomputed_lc.len(),
            },
        )?,
        RecursionPolyAirLeaf::ReservedPoly { index, .. } => get_index(
            env.reserved_poly,
            *index,
            RecursionPolyAirEvaluationError::ReservedPolyIndexOutOfRange {
                index: *index,
                len: env.reserved_poly.len(),
            },
        )?,
        RecursionPolyAirLeaf::IsFirstRow => env.is_first_row,
        RecursionPolyAirLeaf::IsLastRow => env.is_last_row,
    };
    Ok(value)
}

fn node_value(values: &[EF], node_id: u32) -> Result<EF, RecursionPolyAirEvaluationError> {
    values
        .get(node_id as usize)
        .copied()
        .ok_or(RecursionPolyAirEvaluationError::MissingNodeValue { node_id })
}

fn get_index<T: Copy>(
    values: &[T],
    index: usize,
    error: RecursionPolyAirEvaluationError,
) -> Result<T, RecursionPolyAirEvaluationError> {
    values.get(index).copied().ok_or(error)
}

fn check_width(
    field: &'static str,
    expected: usize,
    actual: usize,
) -> Result<(), RecursionPolyAirEvaluationError> {
    if expected != actual {
        return Err(RecursionPolyAirEvaluationError::WidthMismatch { field, expected, actual });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use dt_stark::air::FullAirBuilder;
    use p3_field::AbstractField;
    use polyair::symbolic::{Rotation, SymbolicAirBuilder, SymbolicExpression, SymbolicVar};

    use super::*;
    use crate::{
        config::DIGEST_SIZE,
        symbolic_expr_fixed_dt::{RecursionChildRole, RecursionFixedSymbolicChip},
    };

    fn bounded_program_dto() -> RecursionPolyAirVerifierProgramDto {
        let mut builder = SymbolicAirBuilder::<F, D_EF>::new_empty();
        builder.with_main_width(4);
        builder.with_public_width(1);
        builder.width_max_beta_power(1);
        builder.reserved_poly_output.push(PairCol::Main(0));
        builder.retain_precomputed(SymbolicExpression::VARiable(SymbolicVar::Main(0)));
        builder.send(SymbolicExpression::VARiable(SymbolicVar::Main(2)));
        builder.gate.push(
            SymbolicExpression::VARiable(SymbolicVar::BetaPowers(1)) +
                SymbolicExpression::VARiable(SymbolicVar::Public(0)),
        );
        let fixed_chip = RecursionFixedSymbolicChip::from_symbolic_builder(
            0,
            "bounded".to_string(),
            InteractionScope::Local,
            CONSTRAINT_PROGRAM_LOGUP_BATCH_SIZE,
            1,
            &builder,
        )
        .expect("bounded fixed chip");
        let fixed = RecursionFixedSymbolicProgram::new(
            CONSTRAINT_PROGRAM_SCHEMA_VERSION,
            RecursionChildRole::Core,
            vec![fixed_chip],
            [F::zero(); DIGEST_SIZE],
        )
        .expect("bounded fixed program");
        RecursionPolyAirVerifierProgram::compile(&fixed).expect("bounded frozen program").to_dto()
    }

    #[test]
    fn malformed_programs_fail_before_plan_compile_without_panicking() {
        let valid = bounded_program_dto();
        let mut fixtures = Vec::<(&str, RecursionPolyAirVerifierProgramDto)>::new();

        let mut oversized_chip = valid.clone();
        oversized_chip.chips[0].static_chip_id = 256;
        fixtures.push(("oversized static chip id", oversized_chip));

        let mut duplicate = valid.clone();
        let mut duplicate_chip = duplicate.chips[0].clone();
        duplicate_chip.chip_name = "duplicate".to_string();
        duplicate.chips.push(duplicate_chip);
        fixtures.push(("duplicate static chip id", duplicate));

        let mut unsorted = valid.clone();
        let mut second_segment = unsorted.chips[0].clone();
        second_segment.static_chip_id = 128;
        unsorted.chips.insert(0, second_segment);
        fixtures.push(("unsorted segment id", unsorted));

        let mut non_dense_node = valid.clone();
        non_dense_node.chips[0].node_table[0].node_id = 1;
        fixtures.push(("non-dense node id", non_dense_node));

        let mut invalid_operand = valid.clone();
        let operand_node = invalid_operand.chips[0]
            .node_table
            .iter_mut()
            .find(|node| matches!(node.op, RecursionPolyAirOp::Add { .. }))
            .expect("bounded fixture has an add node");
        operand_node.op = RecursionPolyAirOp::Add { lhs: u32::MAX, rhs: 0 };
        fixtures.push(("out-of-bounds operand", invalid_operand));

        let mut missing_reserved = valid.clone();
        missing_reserved.chips[0]
            .derived_roots
            .retain(|root| !matches!(root, RecursionPolyAirDerivedRoot::ReservedPoly { .. }));
        fixtures.push(("missing reserved root", missing_reserved));

        let mut missing_precompute = valid.clone();
        missing_precompute.chips[0]
            .derived_roots
            .retain(|root| !matches!(root, RecursionPolyAirDerivedRoot::PrecomputeLc { .. }));
        fixtures.push(("missing precompute root", missing_precompute));

        let mut invalid_root = valid.clone();
        invalid_root.chips[0].gate_roots[0].root_node_id = u32::MAX;
        fixtures.push(("out-of-bounds root node", invalid_root));

        let mut zero_batch = valid.clone();
        zero_batch.chips[0].logup_batch_size = 0;
        fixtures.push(("zero batch size", zero_batch));

        let mut overflow_count = valid.clone();
        overflow_count.chips[0].cost_ledger.precompute_root_count = usize::MAX;
        fixtures.push(("overflow row count", overflow_count));

        let mut unsupported_op = valid.clone();
        unsupported_op.chips[0].node_table[0].op = RecursionPolyAirOp::Neg { input: 0 };
        fixtures.push(("unsupported op", unsupported_op));

        let mut unsupported_schema = valid.clone();
        unsupported_schema.version += 1;
        fixtures.push(("unsupported schema version", unsupported_schema));

        let mut stale_schema = valid;
        stale_schema.version = CONSTRAINT_PROGRAM_SCHEMA_VERSION - 1;
        fixtures.push(("stale schema version", stale_schema));

        for (name, fixture) in fixtures {
            let result =
                std::panic::catch_unwind(|| RecursionPolyAirVerifierProgram::try_from_dto(fixture));
            assert!(result.is_ok(), "{name} panicked");
            assert!(result.expect("checked above").is_err(), "{name} was accepted");
        }
    }

    #[test]
    fn frozen_child_layout_indexes_subsets_and_dual_segments_and_rejects_static_drift() {
        fn rebase_chip(
            mut chip: RecursionPolyAirChipIr,
            static_chip_id: usize,
            name: &str,
        ) -> RecursionPolyAirChipIr {
            chip.static_chip_id = static_chip_id;
            chip.chip_name = name.to_string();
            for root in &mut chip.gate_roots {
                root.static_chip_id = static_chip_id;
            }
            chip
        }

        let seed = bounded_program_dto();
        let chip = seed.chips[0].clone();
        let mut dto = seed;
        dto.chips = vec![
            rebase_chip(chip.clone(), 0, "alpha"),
            rebase_chip(chip.clone(), 1, "beta"),
            rebase_chip(chip, 128, "gamma"),
        ];
        let program = RecursionPolyAirVerifierProgram::try_from_dto(dto)
            .expect("dual-segment child authority freezes");
        let base = program.verified_child_layout(0).expect("base-0 child layout");
        let mixed = program.verified_child_layout(128).expect("base-128 child layout");

        assert_eq!(base.static_chip_id("alpha"), Some(0));
        assert_eq!(base.static_chip_id("beta"), Some(1));
        assert_eq!(base.static_chip_id("gamma"), None);
        assert_eq!(mixed.static_chip_id("gamma"), Some(128));
        assert_eq!(mixed.static_chip_id("alpha"), None);
        assert_eq!(
            ["beta"].into_iter().filter_map(|name| base.find_chip(name)).count(),
            1,
            "a proof-local chip subset uses the frozen name index without fixing one shape"
        );
        assert_eq!(base.air_authority(), crate::child_views::NativeAirAuthority::PublicMetadata);

        let exact_machine = base.chips().to_vec();
        base.validate_machine_metadata(&exact_machine).expect("identical cold machine authority");

        let mut unknown_name = exact_machine.clone();
        unknown_name[0].name = "unknown".to_string();
        assert!(matches!(
            base.validate_machine_metadata(&unknown_name),
            Err(crate::child_views::NativeChildViewError::MetadataChipMissing { .. })
        ));

        let mut width_drift = exact_machine;
        width_drift[0].main_width += 1;
        assert!(matches!(
            base.validate_machine_metadata(&width_drift),
            Err(crate::child_views::NativeChildViewError::MetadataWidthMismatch { .. })
        ));
    }

    #[test]
    fn compiles_and_evaluates_gate_roots() {
        let mut builder = SymbolicAirBuilder::<F, D_EF>::new_empty();
        builder.with_main_width(1);
        builder.with_public_width(1);
        builder.width_max_beta_power(1);
        builder.reserved_poly_output.push(PairCol::Main(0));
        builder.gate.push(
            SymbolicExpression::VARiable(SymbolicVar::BetaPowers(1)) *
                SymbolicExpression::VARiable(SymbolicVar::Public(0)),
        );
        let fixed_chip = RecursionFixedSymbolicChip::from_symbolic_builder(
            0,
            "chip".to_string(),
            InteractionScope::Local,
            CONSTRAINT_PROGRAM_LOGUP_BATCH_SIZE,
            1,
            &builder,
        )
        .unwrap();
        let fixed = RecursionFixedSymbolicProgram::new(
            CONSTRAINT_PROGRAM_SCHEMA_VERSION,
            RecursionChildRole::Core,
            vec![fixed_chip],
            [F::zero(); DIGEST_SIZE],
        )
        .unwrap();
        let program = RecursionPolyAirVerifierProgram::compile(&fixed).unwrap();
        let chip = &program.chips[0];

        let beta_power = F::ef_from_base(F::from_canonical_u32(3));
        let public = F::from_canonical_u32(5);
        let reserved = F::ef_from_base(F::from_canonical_u32(7));
        let opened_main = [reserved];
        let beta_powers = [EF::one(), beta_power];
        let public_values = [public];
        let reserved_poly = [reserved];
        let env = RecursionPolyAirEnv {
            proof_idx: 0,
            chip_idx: 0,
            opened_preprocessed: &[],
            opened_main: &opened_main,
            public_values: &public_values,
            constraint_alpha: EF::one(),
            perm_alpha: EF::zero(),
            perm_beta: EF::zero(),
            beta_powers: &beta_powers,
            beta_septix: EF::zero(),
            precomputed_lc: &[],
            reserved_poly: &reserved_poly,
            is_first_row: EF::zero(),
            is_last_row: EF::zero(),
        };

        let derived_values = evaluate_derived_roots(chip, &env).unwrap();
        assert!(derived_values.iter().any(|derived| {
            derived.root == RecursionPolyAirDerivedRoot::BetaPower { power: 1 } &&
                derived.value == beta_power
        }));
        assert!(derived_values.iter().any(|derived| {
            derived.root ==
                RecursionPolyAirDerivedRoot::ReservedPoly { index: 0, source: PairCol::Main(0) } &&
                derived.value == reserved
        }));

        let gate_values = evaluate_gate_roots(chip, &env).unwrap();
        assert_eq!(gate_values, vec![beta_power * F::ef_from_base(public)]);
        assert_eq!(fold_gate_values(&gate_values, EF::from_canonical_u32(11)), gate_values[0]);
        assert_eq!(chip.cost_ledger.internal_recursion_interactions_wide_unroll, 0);
    }

    #[test]
    fn replays_lookup_batches_after_gate_roots() {
        let mut builder = SymbolicAirBuilder::<F, D_EF>::new_empty();
        builder.with_main_width(4);
        builder.width_max_beta_power(1);
        builder.reserved_poly_output.push(PairCol::Main(0));
        builder.retain_precomputed(SymbolicExpression::VARiable(SymbolicVar::Main(0)));
        builder.retain_precomputed(SymbolicExpression::VARiable(SymbolicVar::Main(1)));
        builder.send(SymbolicExpression::VARiable(SymbolicVar::Main(2)));
        builder.recv(SymbolicExpression::VARiable(SymbolicVar::Main(3)));
        builder
            .gate
            .push(SymbolicExpression::VARiable(SymbolicVar::ReservedPoly(0, Rotation::Local)));

        let fixed_chip = RecursionFixedSymbolicChip::from_symbolic_builder(
            0,
            "chip".to_string(),
            InteractionScope::Local,
            2,
            3,
            &builder,
        )
        .unwrap();
        let fixed = RecursionFixedSymbolicProgram::new(
            CONSTRAINT_PROGRAM_SCHEMA_VERSION,
            RecursionChildRole::Core,
            vec![fixed_chip],
            [F::zero(); DIGEST_SIZE],
        )
        .unwrap();
        let chip = &RecursionPolyAirVerifierProgram::compile(&fixed).unwrap().chips[0];

        let main = [
            EF::from_canonical_u32(5),
            EF::from_canonical_u32(7),
            EF::from_canonical_u32(11),
            EF::from_canonical_u32(13),
        ];
        let public_values = [];
        let beta_powers = [EF::one(), EF::from_canonical_u32(3)];
        let base_env = RecursionPolyAirEnv {
            proof_idx: 0,
            chip_idx: 0,
            opened_preprocessed: &[],
            opened_main: &main,
            public_values: &public_values,
            constraint_alpha: EF::from_canonical_u32(17),
            perm_alpha: EF::zero(),
            perm_beta: EF::zero(),
            beta_powers: &beta_powers,
            beta_septix: EF::zero(),
            precomputed_lc: &[],
            reserved_poly: &[],
            is_first_row: EF::zero(),
            is_last_row: EF::zero(),
        };
        let (replay, profile) =
            evaluate_chip_replay_profiled(chip, &base_env, &[EF::from_canonical_u32(19)]).unwrap();
        assert_eq!(profile.precompute_nodes + profile.remaining_nodes, chip.node_table.len());

        assert_eq!(replay.precomputed_lc, vec![main[0], main[1]]);
        assert_eq!(replay.signed_lookup_multiplicities, vec![main[2], -main[3]]);
        let expected_numerator = main[1] * main[2] - main[0] * main[3];
        let expected_denominator = main[0] * main[1];
        let expected_lookup_constraint =
            expected_numerator - expected_denominator * EF::from_canonical_u32(19);
        assert_eq!(replay.lookup_batches[0].numerator, expected_numerator);
        assert_eq!(replay.lookup_batches[0].denominator, expected_denominator);
        assert_eq!(replay.lookup_batches[0].constraint_value, expected_lookup_constraint);
        assert_eq!(
            replay.accumulator,
            main[0] * EF::from_canonical_u32(17) + expected_lookup_constraint
        );
    }

    #[test]
    fn binding_rejects_constraint_count_mismatch() {
        let mut builder = SymbolicAirBuilder::<F, D_EF>::new_empty();
        builder.gate.push(SymbolicExpression::from(F::one()));
        let fixed_chip = RecursionFixedSymbolicChip::from_symbolic_builder(
            0,
            "chip".to_string(),
            InteractionScope::Local,
            1,
            4,
            &builder,
        )
        .unwrap();
        let chip = RecursionPolyAirChipIr::compile(&fixed_chip).unwrap();
        let widths = RecursionPolyAirWidths { preprocessed: 0, main: 0, public: 0 };

        let err = chip.bind_observed_chip(0, 0, "chip", widths, 3).unwrap_err();
        assert_eq!(
            err,
            RecursionPolyAirEvaluationError::ConstraintCountMismatch { expected: 4, actual: 3 }
        );
    }
}
