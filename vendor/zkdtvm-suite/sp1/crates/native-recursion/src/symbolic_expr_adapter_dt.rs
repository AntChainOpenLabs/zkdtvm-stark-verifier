use std::collections::HashMap;

use p3_field::{AbstractExtensionField, AbstractField, PrimeField32};
use polyair::symbolic::{SymbolicExpression, SymbolicVar};
use serde::{Deserialize, Serialize};

use crate::config::{D_EF, EF, F};

pub type RecursionSymbolicExpression = SymbolicExpression<SymbolicVar, F, D_EF>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecursionSymbolicExprAdapter {
    pub version: u32,
}

impl Default for RecursionSymbolicExprAdapter {
    fn default() -> Self {
        Self { version: 2 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecursionRootKind {
    PrecomputeLc,
    Gate,
    LookupMultiplicity,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecursionAdaptedRoot {
    pub kind: RecursionRootKind,
    pub root_index: usize,
    pub root_node_id: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecursionPolyAirLeaf {
    Preprocessed { col: usize },
    Main { col: usize },
    Public { index: usize },
    PermAlpha,
    BetaPower { power: usize },
    BetaSeptix,
    Precomputed { index: usize },
    ReservedPoly { index: usize },
    IsFirstRow,
    IsLastRow,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecursionPolyAirOp {
    Leaf(RecursionPolyAirLeaf),
    ConstBase(F),
    ConstExt(EF),
    Add { lhs: u32, rhs: u32 },
    Sub { lhs: u32, rhs: u32 },
    Neg { input: u32 },
    Mul { lhs: u32, rhs: u32 },
    FusedMulAdd { lhs: u32, rhs: u32, addend: u32, sign: bool },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecursionPolyAirNode {
    pub node_id: u32,
    pub op: RecursionPolyAirOp,
    pub degree_multiple: u16,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecursionOpMix {
    pub leaves: usize,
    pub const_base: usize,
    pub const_ext: usize,
    pub adds: usize,
    pub subs: usize,
    pub negs: usize,
    pub muls: usize,
}

impl RecursionOpMix {
    pub fn observe(&mut self, op: &RecursionPolyAirOp) {
        match op {
            RecursionPolyAirOp::Leaf(_) => self.leaves += 1,
            RecursionPolyAirOp::ConstBase(_) => self.const_base += 1,
            RecursionPolyAirOp::ConstExt(_) => self.const_ext += 1,
            RecursionPolyAirOp::Add { .. } => self.adds += 1,
            RecursionPolyAirOp::Sub { .. } => self.subs += 1,
            RecursionPolyAirOp::Neg { .. } => self.negs += 1,
            RecursionPolyAirOp::Mul { .. } => self.muls += 1,
            RecursionPolyAirOp::FusedMulAdd { .. } => {}
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecursionAdapterError {
    NodeCountOverflow,
    DegreeMultipleTooLarge { degree_multiple: usize },
    UnsupportedExtensionConstant,
    UnsupportedTransitionSelector,
    RawTraceLeafInGateRoot { root_index: usize, leaf: RecursionPolyAirLeaf },
    MissingChildNode,
    InvalidNodeId { node_id: u32 },
    DeletedRootNode { node_id: u32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CanonicalExtensionConstant {
    Base(F),
    Theta,
    Unsupported,
}

pub(crate) fn classify_extension_constant(value: &EF) -> CanonicalExtensionConstant {
    let limbs = <EF as AbstractExtensionField<F>>::as_base_slice(value);
    if limbs[1..].iter().all(|limb| *limb == F::zero()) {
        CanonicalExtensionConstant::Base(limbs[0])
    } else if *value == <EF as AbstractExtensionField<F>>::monomial(1) {
        CanonicalExtensionConstant::Theta
    } else {
        CanonicalExtensionConstant::Unsupported
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecursionAdaptedRoots {
    pub roots: Vec<RecursionAdaptedRoot>,
    pub node_table: Vec<RecursionPolyAirNode>,
    pub op_mix: RecursionOpMix,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecursionAdaptedRootStreams {
    pub precompute_roots: Vec<RecursionAdaptedRoot>,
    pub lookup_multiplicity_roots: Vec<RecursionAdaptedRoot>,
    pub gate_roots: Vec<RecursionAdaptedRoot>,
    pub node_table: Vec<RecursionPolyAirNode>,
    pub op_mix: RecursionOpMix,
}

#[derive(Debug, Default)]
struct RecursionHashConsState {
    node_table: Vec<RecursionPolyAirNode>,
    ptr_cache: HashMap<usize, u32>,
    struct_cache: HashMap<Vec<u8>, u32>,
    progress: RecursionAdapterProgress,
}

#[derive(Debug)]
struct RecursionAdapterProgress {
    interval: usize,
    next_node_count: usize,
    stream: RecursionRootKind,
    root_index: usize,
    max_depth: usize,
}

impl Default for RecursionAdapterProgress {
    fn default() -> Self {
        let interval = crate::env_var("NATIVE_RECURSION_ADAPTER_PROGRESS")
            .ok()
            .and_then(
                |value| {
                    if value == "1" {
                        Some(1_000_000)
                    } else {
                        value.parse::<usize>().ok()
                    }
                },
            )
            .unwrap_or(0);
        Self {
            interval,
            next_node_count: interval,
            stream: RecursionRootKind::PrecomputeLc,
            root_index: 0,
            max_depth: 0,
        }
    }
}

impl RecursionAdapterProgress {
    fn enter_root(&mut self, stream: RecursionRootKind, root_index: usize) {
        self.stream = stream;
        self.root_index = root_index;
    }

    fn observe_depth(&mut self, depth: usize) {
        self.max_depth = self.max_depth.max(depth);
    }

    fn maybe_report(&mut self, node_count: usize, depth: usize) {
        self.observe_depth(depth);
        if self.interval == 0 {
            return;
        }
        while node_count >= self.next_node_count {
            eprintln!(
                "ADAPTER_PROGRESS stream={:?} root={} nodes={} depth={} max_depth={}",
                self.stream, self.root_index, node_count, depth, self.max_depth
            );
            self.next_node_count = self.next_node_count.saturating_add(self.interval);
        }
    }
}

impl RecursionSymbolicExprAdapter {
    pub fn adapt_roots(
        &self,
        kind: RecursionRootKind,
        expressions: &[RecursionSymbolicExpression],
    ) -> Result<RecursionAdaptedRoots, RecursionAdapterError> {
        let mut state = RecursionHashConsState::default();
        let roots = self.adapt_stream_into(kind, expressions, &mut state)?;
        let op_mix = op_mix(&state.node_table);
        Ok(RecursionAdaptedRoots { roots, node_table: state.node_table, op_mix })
    }

    pub fn adapt_root_streams(
        &self,
        precompute_roots: &[RecursionSymbolicExpression],
        lookup_multiplicity_roots: &[RecursionSymbolicExpression],
        gate_roots: &[RecursionSymbolicExpression],
    ) -> Result<RecursionAdaptedRootStreams, RecursionAdapterError> {
        let mut state = RecursionHashConsState::default();
        let precompute_roots =
            self.adapt_stream_into(RecursionRootKind::PrecomputeLc, precompute_roots, &mut state)?;
        let lookup_multiplicity_roots = self.adapt_stream_into(
            RecursionRootKind::LookupMultiplicity,
            lookup_multiplicity_roots,
            &mut state,
        )?;
        let gate_roots = self.adapt_stream_into(RecursionRootKind::Gate, gate_roots, &mut state)?;
        let op_mix = op_mix(&state.node_table);
        Ok(RecursionAdaptedRootStreams {
            precompute_roots,
            lookup_multiplicity_roots,
            gate_roots,
            node_table: state.node_table,
            op_mix,
        })
    }

    pub fn adapt_root(
        &self,
        kind: RecursionRootKind,
        root_index: usize,
        expression: &RecursionSymbolicExpression,
        node_table: &mut Vec<RecursionPolyAirNode>,
    ) -> Result<RecursionAdaptedRoot, RecursionAdapterError> {
        let mut ptr_cache = HashMap::<usize, u32>::new();
        let mut struct_cache = struct_cache_for_nodes(node_table);
        let root_node_id = self.hash_cons_expression(
            kind,
            root_index,
            expression,
            &mut ptr_cache,
            &mut struct_cache,
            node_table,
            &mut RecursionAdapterProgress::default(),
            0,
        )?;
        Ok(RecursionAdaptedRoot { kind, root_index, root_node_id })
    }

    pub fn canonical_bytes_for_root_streams(&self, roots: &RecursionAdaptedRootStreams) -> Vec<u8> {
        let mut out = Vec::new();
        put_u32(&mut out, self.version);
        put_usize(&mut out, roots.node_table.len());
        for node in &roots.node_table {
            encode_node(&mut out, node);
        }
        encode_root_slice(&mut out, &roots.precompute_roots);
        encode_root_slice(&mut out, &roots.lookup_multiplicity_roots);
        encode_root_slice(&mut out, &roots.gate_roots);
        out
    }

    fn adapt_stream_into(
        &self,
        kind: RecursionRootKind,
        expressions: &[RecursionSymbolicExpression],
        state: &mut RecursionHashConsState,
    ) -> Result<Vec<RecursionAdaptedRoot>, RecursionAdapterError> {
        let mut roots = Vec::with_capacity(expressions.len());
        for (root_index, expression) in expressions.iter().enumerate() {
            state.progress.enter_root(kind, root_index);
            let root_node_id = self.hash_cons_expression(
                kind,
                root_index,
                expression,
                &mut state.ptr_cache,
                &mut state.struct_cache,
                &mut state.node_table,
                &mut state.progress,
                0,
            )?;
            roots.push(RecursionAdaptedRoot { kind, root_index, root_node_id });
        }
        Ok(roots)
    }

    fn hash_cons_expression(
        &self,
        kind: RecursionRootKind,
        root_index: usize,
        expression: &RecursionSymbolicExpression,
        ptr_cache: &mut HashMap<usize, u32>,
        struct_cache: &mut HashMap<Vec<u8>, u32>,
        node_table: &mut Vec<RecursionPolyAirNode>,
        progress: &mut RecursionAdapterProgress,
        depth: usize,
    ) -> Result<u32, RecursionAdapterError> {
        progress.observe_depth(depth);
        let ptr = expr_ptr(expression);
        if let Some(node_id) = ptr_cache.get(&ptr).copied() {
            let op = &node_table
                .get(node_id as usize)
                .ok_or(RecursionAdapterError::InvalidNodeId { node_id })?
                .op;
            self.check_root_leaf_partition(kind, root_index, op)?;
            return Ok(node_id);
        }

        let op = match expression {
            SymbolicExpression::VARiable(var) => RecursionPolyAirOp::Leaf(convert_leaf(*var)?),
            SymbolicExpression::Constant(value) => RecursionPolyAirOp::ConstBase(*value),
            SymbolicExpression::ConstantExt(value) => match classify_extension_constant(value) {
                CanonicalExtensionConstant::Base(value) => RecursionPolyAirOp::ConstBase(value),
                CanonicalExtensionConstant::Theta => RecursionPolyAirOp::ConstExt(*value),
                CanonicalExtensionConstant::Unsupported => {
                    return Err(RecursionAdapterError::UnsupportedExtensionConstant);
                }
            },
            SymbolicExpression::Add { x, y, .. } => {
                if let (Some(lhs), Some(rhs)) =
                    (literal_base_const(x.as_ref()), literal_base_const(y.as_ref()))
                {
                    let node_id = intern_const_base(lhs + rhs, struct_cache, node_table)?;
                    ptr_cache.insert(ptr, node_id);
                    return Ok(node_id);
                }
                if literal_base_const(x.as_ref()).is_some_and(|value| value == F::zero()) {
                    let node_id = self.hash_cons_expression(
                        kind,
                        root_index,
                        y.as_ref(),
                        ptr_cache,
                        struct_cache,
                        node_table,
                        progress,
                        depth + 1,
                    )?;
                    ptr_cache.insert(ptr, node_id);
                    return Ok(node_id);
                }
                if literal_base_const(y.as_ref()).is_some_and(|value| value == F::zero()) {
                    let node_id = self.hash_cons_expression(
                        kind,
                        root_index,
                        x.as_ref(),
                        ptr_cache,
                        struct_cache,
                        node_table,
                        progress,
                        depth + 1,
                    )?;
                    ptr_cache.insert(ptr, node_id);
                    return Ok(node_id);
                }
                let lhs = self.hash_cons_expression(
                    kind,
                    root_index,
                    x.as_ref(),
                    ptr_cache,
                    struct_cache,
                    node_table,
                    progress,
                    depth + 1,
                )?;
                let rhs = self.hash_cons_expression(
                    kind,
                    root_index,
                    y.as_ref(),
                    ptr_cache,
                    struct_cache,
                    node_table,
                    progress,
                    depth + 1,
                )?;
                if let (Some(lhs_value), Some(rhs_value)) =
                    (node_base_const(node_table, lhs)?, node_base_const(node_table, rhs)?)
                {
                    let node_id =
                        intern_const_base(lhs_value + rhs_value, struct_cache, node_table)?;
                    ptr_cache.insert(ptr, node_id);
                    return Ok(node_id);
                }
                if node_base_const(node_table, lhs)?.is_some_and(|value| value == F::zero()) {
                    ptr_cache.insert(ptr, rhs);
                    return Ok(rhs);
                }
                if node_base_const(node_table, rhs)?.is_some_and(|value| value == F::zero()) {
                    ptr_cache.insert(ptr, lhs);
                    return Ok(lhs);
                }
                let (lhs, rhs) = sorted_pair(lhs, rhs);
                RecursionPolyAirOp::Add { lhs, rhs }
            }
            SymbolicExpression::Sub { x, y, .. } => {
                if let (Some(lhs), Some(rhs)) =
                    (literal_base_const(x.as_ref()), literal_base_const(y.as_ref()))
                {
                    let node_id = intern_const_base(lhs - rhs, struct_cache, node_table)?;
                    ptr_cache.insert(ptr, node_id);
                    return Ok(node_id);
                }
                if literal_base_const(y.as_ref()).is_some_and(|value| value == F::zero()) {
                    let node_id = self.hash_cons_expression(
                        kind,
                        root_index,
                        x.as_ref(),
                        ptr_cache,
                        struct_cache,
                        node_table,
                        progress,
                        depth + 1,
                    )?;
                    ptr_cache.insert(ptr, node_id);
                    return Ok(node_id);
                }
                let lhs = self.hash_cons_expression(
                    kind,
                    root_index,
                    x.as_ref(),
                    ptr_cache,
                    struct_cache,
                    node_table,
                    progress,
                    depth + 1,
                )?;
                let rhs = self.hash_cons_expression(
                    kind,
                    root_index,
                    y.as_ref(),
                    ptr_cache,
                    struct_cache,
                    node_table,
                    progress,
                    depth + 1,
                )?;
                if let Some(rhs_value) = node_base_const(node_table, rhs)? {
                    if rhs_value == F::zero() {
                        ptr_cache.insert(ptr, lhs);
                        return Ok(lhs);
                    }
                }
                if let (Some(lhs_value), Some(rhs_value)) =
                    (node_base_const(node_table, lhs)?, node_base_const(node_table, rhs)?)
                {
                    let node_id =
                        intern_const_base(lhs_value - rhs_value, struct_cache, node_table)?;
                    ptr_cache.insert(ptr, node_id);
                    return Ok(node_id);
                }
                RecursionPolyAirOp::Sub { lhs, rhs }
            }
            SymbolicExpression::Neg { x, .. } => {
                if let Some(value) = literal_base_const(x.as_ref()) {
                    let node_id = intern_const_base(-value, struct_cache, node_table)?;
                    ptr_cache.insert(ptr, node_id);
                    return Ok(node_id);
                }
                if let SymbolicExpression::Neg { x: inner, .. } = x.as_ref() {
                    let node_id = self.hash_cons_expression(
                        kind,
                        root_index,
                        inner.as_ref(),
                        ptr_cache,
                        struct_cache,
                        node_table,
                        progress,
                        depth + 1,
                    )?;
                    ptr_cache.insert(ptr, node_id);
                    return Ok(node_id);
                }
                let lhs = intern_const_base(F::zero(), struct_cache, node_table)?;
                let rhs = self.hash_cons_expression(
                    kind,
                    root_index,
                    x.as_ref(),
                    ptr_cache,
                    struct_cache,
                    node_table,
                    progress,
                    depth + 1,
                )?;
                RecursionPolyAirOp::Sub { lhs, rhs }
            }
            SymbolicExpression::Mul { x, y, .. } => {
                if let (Some(lhs), Some(rhs)) =
                    (literal_base_const(x.as_ref()), literal_base_const(y.as_ref()))
                {
                    let node_id = intern_const_base(lhs * rhs, struct_cache, node_table)?;
                    ptr_cache.insert(ptr, node_id);
                    return Ok(node_id);
                }
                if literal_base_const(x.as_ref()).is_some_and(|value| value == F::zero()) ||
                    literal_base_const(y.as_ref()).is_some_and(|value| value == F::zero())
                {
                    let node_id = intern_const_base(F::zero(), struct_cache, node_table)?;
                    ptr_cache.insert(ptr, node_id);
                    return Ok(node_id);
                }
                if literal_base_const(x.as_ref()).is_some_and(|value| value == F::one()) {
                    let node_id = self.hash_cons_expression(
                        kind,
                        root_index,
                        y.as_ref(),
                        ptr_cache,
                        struct_cache,
                        node_table,
                        progress,
                        depth + 1,
                    )?;
                    ptr_cache.insert(ptr, node_id);
                    return Ok(node_id);
                }
                if literal_base_const(y.as_ref()).is_some_and(|value| value == F::one()) {
                    let node_id = self.hash_cons_expression(
                        kind,
                        root_index,
                        x.as_ref(),
                        ptr_cache,
                        struct_cache,
                        node_table,
                        progress,
                        depth + 1,
                    )?;
                    ptr_cache.insert(ptr, node_id);
                    return Ok(node_id);
                }
                let lhs = self.hash_cons_expression(
                    kind,
                    root_index,
                    x.as_ref(),
                    ptr_cache,
                    struct_cache,
                    node_table,
                    progress,
                    depth + 1,
                )?;
                let rhs = self.hash_cons_expression(
                    kind,
                    root_index,
                    y.as_ref(),
                    ptr_cache,
                    struct_cache,
                    node_table,
                    progress,
                    depth + 1,
                )?;
                if let (Some(lhs_value), Some(rhs_value)) =
                    (node_base_const(node_table, lhs)?, node_base_const(node_table, rhs)?)
                {
                    let node_id =
                        intern_const_base(lhs_value * rhs_value, struct_cache, node_table)?;
                    ptr_cache.insert(ptr, node_id);
                    return Ok(node_id);
                }
                if node_base_const(node_table, lhs)?.is_some_and(|value| value == F::zero()) ||
                    node_base_const(node_table, rhs)?.is_some_and(|value| value == F::zero())
                {
                    let node_id = intern_const_base(F::zero(), struct_cache, node_table)?;
                    ptr_cache.insert(ptr, node_id);
                    return Ok(node_id);
                }
                if node_base_const(node_table, lhs)?.is_some_and(|value| value == F::one()) {
                    ptr_cache.insert(ptr, rhs);
                    return Ok(rhs);
                }
                if node_base_const(node_table, rhs)?.is_some_and(|value| value == F::one()) {
                    ptr_cache.insert(ptr, lhs);
                    return Ok(lhs);
                }
                let (lhs, rhs) = sorted_pair(lhs, rhs);
                RecursionPolyAirOp::Mul { lhs, rhs }
            }
        };
        self.check_root_leaf_partition(kind, root_index, &op)?;

        let key = op_key(&op);
        if let Some(node_id) = struct_cache.get(&key).copied() {
            ptr_cache.insert(ptr, node_id);
            return Ok(node_id);
        }

        let degree_multiple = degree_multiple_u16(expression)?;
        let node_id = u32::try_from(node_table.len())
            .map_err(|_| RecursionAdapterError::NodeCountOverflow)?;
        node_table.push(RecursionPolyAirNode { node_id, op, degree_multiple });
        struct_cache.insert(key, node_id);
        ptr_cache.insert(ptr, node_id);
        progress.maybe_report(node_table.len(), depth);
        Ok(node_id)
    }

    fn check_root_leaf_partition(
        &self,
        kind: RecursionRootKind,
        root_index: usize,
        op: &RecursionPolyAirOp,
    ) -> Result<(), RecursionAdapterError> {
        if kind != RecursionRootKind::Gate {
            return Ok(());
        }
        if let RecursionPolyAirOp::Leaf(leaf @ RecursionPolyAirLeaf::Preprocessed { .. }) |
        RecursionPolyAirOp::Leaf(leaf @ RecursionPolyAirLeaf::Main { .. }) = op
        {
            return Err(RecursionAdapterError::RawTraceLeafInGateRoot {
                root_index,
                leaf: leaf.clone(),
            });
        }
        Ok(())
    }
}

pub fn op_mix(nodes: &[RecursionPolyAirNode]) -> RecursionOpMix {
    let mut mix = RecursionOpMix::default();
    for node in nodes {
        mix.observe(&node.op);
    }
    mix
}

pub fn fuse_mul_add_nodes(
    mut roots: RecursionAdaptedRootStreams,
) -> Result<RecursionAdaptedRootStreams, RecursionAdapterError> {
    let node_count = roots.node_table.len();
    if node_count == 0 {
        return Ok(roots);
    }

    let fanouts = fusion_fanouts(&roots);
    let mut deleted = vec![false; node_count];
    let mut fused = vec![None; node_count];

    for node in &roots.node_table {
        let node_idx = node.node_id as usize;
        let candidate = match node.op {
            RecursionPolyAirOp::Add { lhs, rhs } => {
                fused_candidate(&roots.node_table, &fanouts, lhs, rhs, false)
                    .or_else(|| fused_candidate(&roots.node_table, &fanouts, rhs, lhs, false))
            }
            RecursionPolyAirOp::Sub { lhs, rhs } => {
                fused_candidate(&roots.node_table, &fanouts, lhs, rhs, true)
            }
            _ => None,
        };
        if let Some(candidate) = candidate {
            deleted[candidate.mul_node_id as usize] = true;
            fused[node_idx] = Some(candidate);
        }
    }

    if fused.iter().all(Option::is_none) {
        return Ok(roots);
    }

    let mut remap = vec![None; node_count];
    let mut new_nodes =
        Vec::with_capacity(node_count - deleted.iter().filter(|value| **value).count());
    let old_node_table = roots.node_table;
    for node in &old_node_table {
        let old_idx = node.node_id as usize;
        if deleted[old_idx] {
            continue;
        }
        let node_id =
            u32::try_from(new_nodes.len()).map_err(|_| RecursionAdapterError::NodeCountOverflow)?;
        let op = if let Some(candidate) = fused[old_idx] {
            RecursionPolyAirOp::FusedMulAdd {
                lhs: remapped_node(&remap, candidate.lhs)?,
                rhs: remapped_node(&remap, candidate.rhs)?,
                addend: remapped_node(&remap, candidate.addend)?,
                sign: candidate.sign,
            }
        } else {
            remap_op(&node.op, &remap)?
        };
        remap[old_idx] = Some(node_id);
        new_nodes.push(RecursionPolyAirNode { node_id, op, degree_multiple: node.degree_multiple });
    }

    remap_roots(&mut roots.precompute_roots, &remap)?;
    remap_roots(&mut roots.lookup_multiplicity_roots, &remap)?;
    remap_roots(&mut roots.gate_roots, &remap)?;
    roots.node_table = new_nodes;
    roots.op_mix = op_mix(&roots.node_table);
    Ok(roots)
}

#[derive(Debug, Clone, Copy)]
struct FusedCandidate {
    mul_node_id: u32,
    lhs: u32,
    rhs: u32,
    addend: u32,
    sign: bool,
}

fn fused_candidate(
    nodes: &[RecursionPolyAirNode],
    fanouts: &[usize],
    mul_node_id: u32,
    addend: u32,
    sign: bool,
) -> Option<FusedCandidate> {
    let mul_idx = mul_node_id as usize;
    if fanouts.get(mul_idx).copied()? != 1 {
        return None;
    }
    let RecursionPolyAirOp::Mul { lhs, rhs } = nodes.get(mul_idx)?.op else {
        return None;
    };
    Some(FusedCandidate { mul_node_id, lhs, rhs, addend, sign })
}

fn fusion_fanouts(roots: &RecursionAdaptedRootStreams) -> Vec<usize> {
    let mut fanouts = vec![0usize; roots.node_table.len()];
    for node in &roots.node_table {
        for child in op_children(&node.op) {
            bump_fanout(&mut fanouts, child);
        }
        if let RecursionPolyAirOp::Leaf(RecursionPolyAirLeaf::Precomputed { index }) = node.op {
            if let Some(root) = roots.precompute_roots.iter().find(|root| root.root_index == index)
            {
                bump_fanout(&mut fanouts, root.root_node_id);
            }
        }
    }
    for root in roots
        .precompute_roots
        .iter()
        .chain(roots.lookup_multiplicity_roots.iter())
        .chain(roots.gate_roots.iter())
    {
        bump_fanout(&mut fanouts, root.root_node_id);
    }
    fanouts
}

fn op_children(op: &RecursionPolyAirOp) -> impl Iterator<Item = u32> + '_ {
    let mut children = [None; 3];
    match op {
        RecursionPolyAirOp::Add { lhs, rhs } |
        RecursionPolyAirOp::Sub { lhs, rhs } |
        RecursionPolyAirOp::Mul { lhs, rhs } => {
            children[0] = Some(*lhs);
            children[1] = Some(*rhs);
        }
        RecursionPolyAirOp::Neg { input } => {
            children[0] = Some(*input);
        }
        RecursionPolyAirOp::FusedMulAdd { lhs, rhs, addend, .. } => {
            children[0] = Some(*lhs);
            children[1] = Some(*rhs);
            children[2] = Some(*addend);
        }
        _ => {}
    }
    children.into_iter().flatten()
}

fn bump_fanout(fanouts: &mut [usize], node_id: u32) {
    if let Some(value) = fanouts.get_mut(node_id as usize) {
        *value += 1;
    }
}

fn remap_roots(
    roots: &mut [RecursionAdaptedRoot],
    remap: &[Option<u32>],
) -> Result<(), RecursionAdapterError> {
    for root in roots {
        root.root_node_id = remapped_node(remap, root.root_node_id)?;
    }
    Ok(())
}

fn remap_op(
    op: &RecursionPolyAirOp,
    remap: &[Option<u32>],
) -> Result<RecursionPolyAirOp, RecursionAdapterError> {
    let op = match op {
        RecursionPolyAirOp::Leaf(leaf) => RecursionPolyAirOp::Leaf(leaf.clone()),
        RecursionPolyAirOp::ConstBase(value) => RecursionPolyAirOp::ConstBase(*value),
        RecursionPolyAirOp::ConstExt(value) => RecursionPolyAirOp::ConstExt(*value),
        RecursionPolyAirOp::Add { lhs, rhs } => RecursionPolyAirOp::Add {
            lhs: remapped_node(remap, *lhs)?,
            rhs: remapped_node(remap, *rhs)?,
        },
        RecursionPolyAirOp::Sub { lhs, rhs } => RecursionPolyAirOp::Sub {
            lhs: remapped_node(remap, *lhs)?,
            rhs: remapped_node(remap, *rhs)?,
        },
        RecursionPolyAirOp::Neg { input } => {
            RecursionPolyAirOp::Neg { input: remapped_node(remap, *input)? }
        }
        RecursionPolyAirOp::Mul { lhs, rhs } => RecursionPolyAirOp::Mul {
            lhs: remapped_node(remap, *lhs)?,
            rhs: remapped_node(remap, *rhs)?,
        },
        RecursionPolyAirOp::FusedMulAdd { lhs, rhs, addend, sign } => {
            RecursionPolyAirOp::FusedMulAdd {
                lhs: remapped_node(remap, *lhs)?,
                rhs: remapped_node(remap, *rhs)?,
                addend: remapped_node(remap, *addend)?,
                sign: *sign,
            }
        }
    };
    Ok(op)
}

fn remapped_node(remap: &[Option<u32>], node_id: u32) -> Result<u32, RecursionAdapterError> {
    remap
        .get(node_id as usize)
        .copied()
        .flatten()
        .ok_or(RecursionAdapterError::DeletedRootNode { node_id })
}

fn literal_base_const(expr: &RecursionSymbolicExpression) -> Option<F> {
    match expr {
        SymbolicExpression::Constant(value) => Some(*value),
        SymbolicExpression::ConstantExt(value) => base_from_ext_const(value),
        _ => None,
    }
}

fn node_base_const(
    node_table: &[RecursionPolyAirNode],
    node_id: u32,
) -> Result<Option<F>, RecursionAdapterError> {
    let op = &node_table
        .get(node_id as usize)
        .ok_or(RecursionAdapterError::InvalidNodeId { node_id })?
        .op;
    let value = match op {
        RecursionPolyAirOp::ConstBase(value) => Some(*value),
        RecursionPolyAirOp::ConstExt(value) => base_from_ext_const(value),
        _ => None,
    };
    Ok(value)
}

fn base_from_ext_const(value: &EF) -> Option<F> {
    match classify_extension_constant(value) {
        CanonicalExtensionConstant::Base(value) => Some(value),
        CanonicalExtensionConstant::Theta | CanonicalExtensionConstant::Unsupported => None,
    }
}

fn intern_const_base(
    value: F,
    struct_cache: &mut HashMap<Vec<u8>, u32>,
    node_table: &mut Vec<RecursionPolyAirNode>,
) -> Result<u32, RecursionAdapterError> {
    let op = RecursionPolyAirOp::ConstBase(value);
    let key = op_key(&op);
    if let Some(node_id) = struct_cache.get(&key).copied() {
        return Ok(node_id);
    }
    let node_id =
        u32::try_from(node_table.len()).map_err(|_| RecursionAdapterError::NodeCountOverflow)?;
    node_table.push(RecursionPolyAirNode { node_id, op, degree_multiple: 0 });
    struct_cache.insert(key, node_id);
    Ok(node_id)
}

fn sorted_pair(lhs: u32, rhs: u32) -> (u32, u32) {
    if lhs <= rhs {
        (lhs, rhs)
    } else {
        (rhs, lhs)
    }
}

pub fn encode_node(out: &mut Vec<u8>, node: &RecursionPolyAirNode) {
    put_u32(out, node.node_id);
    put_u16(out, node.degree_multiple);
    encode_op(out, &node.op);
}

pub fn encode_root(out: &mut Vec<u8>, root: &RecursionAdaptedRoot) {
    put_u8(out, root_kind_tag(root.kind));
    put_usize(out, root.root_index);
    put_u32(out, root.root_node_id);
}

pub fn encode_root_slice(out: &mut Vec<u8>, roots: &[RecursionAdaptedRoot]) {
    put_usize(out, roots.len());
    for root in roots {
        encode_root(out, root);
    }
}

pub fn encode_op(out: &mut Vec<u8>, op: &RecursionPolyAirOp) {
    match op {
        RecursionPolyAirOp::Leaf(leaf) => {
            put_u8(out, 0);
            encode_leaf(out, leaf);
        }
        RecursionPolyAirOp::ConstBase(value) => {
            put_u8(out, 1);
            encode_f(out, value);
        }
        RecursionPolyAirOp::ConstExt(value) => {
            put_u8(out, 2);
            encode_ef(out, value);
        }
        RecursionPolyAirOp::Add { lhs, rhs } => {
            put_u8(out, 3);
            put_u32(out, *lhs);
            put_u32(out, *rhs);
        }
        RecursionPolyAirOp::Sub { lhs, rhs } => {
            put_u8(out, 4);
            put_u32(out, *lhs);
            put_u32(out, *rhs);
        }
        RecursionPolyAirOp::Neg { input } => {
            put_u8(out, 5);
            put_u32(out, *input);
        }
        RecursionPolyAirOp::Mul { lhs, rhs } => {
            put_u8(out, 6);
            put_u32(out, *lhs);
            put_u32(out, *rhs);
        }
        RecursionPolyAirOp::FusedMulAdd { lhs, rhs, addend, sign } => {
            put_u8(out, 7);
            put_u32(out, *lhs);
            put_u32(out, *rhs);
            put_u32(out, *addend);
            put_u8(out, u8::from(*sign));
        }
    }
}

pub fn op_key(op: &RecursionPolyAirOp) -> Vec<u8> {
    let mut out = Vec::new();
    encode_op(&mut out, op);
    out
}

pub fn encode_leaf(out: &mut Vec<u8>, leaf: &RecursionPolyAirLeaf) {
    match leaf {
        RecursionPolyAirLeaf::Preprocessed { col } => {
            put_u8(out, 0);
            put_usize(out, *col);
        }
        RecursionPolyAirLeaf::Main { col } => {
            put_u8(out, 1);
            put_usize(out, *col);
        }
        RecursionPolyAirLeaf::Public { index } => {
            put_u8(out, 2);
            put_usize(out, *index);
        }
        RecursionPolyAirLeaf::PermAlpha => put_u8(out, 3),
        RecursionPolyAirLeaf::BetaPower { power } => {
            put_u8(out, 4);
            put_usize(out, *power);
        }
        RecursionPolyAirLeaf::BetaSeptix => put_u8(out, 5),
        RecursionPolyAirLeaf::Precomputed { index } => {
            put_u8(out, 6);
            put_usize(out, *index);
        }
        RecursionPolyAirLeaf::ReservedPoly { index } => {
            put_u8(out, 7);
            put_usize(out, *index);
        }
        RecursionPolyAirLeaf::IsFirstRow => put_u8(out, 9),
        RecursionPolyAirLeaf::IsLastRow => put_u8(out, 10),
    }
}

pub fn put_u8(out: &mut Vec<u8>, value: u8) {
    out.push(value);
}

pub fn put_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}

pub fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

pub fn put_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

pub fn put_usize(out: &mut Vec<u8>, value: usize) {
    put_u64(out, value as u64);
}

pub fn encode_f(out: &mut Vec<u8>, value: &F) {
    put_u32(out, PrimeField32::as_canonical_u32(value));
}

pub fn encode_ef(out: &mut Vec<u8>, value: &EF) {
    for limb in <EF as AbstractExtensionField<F>>::as_base_slice(value) {
        encode_f(out, limb);
    }
}

pub fn root_kind_tag(kind: RecursionRootKind) -> u8 {
    match kind {
        RecursionRootKind::PrecomputeLc => 0,
        RecursionRootKind::Gate => 1,
        RecursionRootKind::LookupMultiplicity => 2,
    }
}

fn convert_leaf(var: SymbolicVar) -> Result<RecursionPolyAirLeaf, RecursionAdapterError> {
    let leaf = match var {
        SymbolicVar::Preprocessed(col) => RecursionPolyAirLeaf::Preprocessed { col },
        SymbolicVar::Main(col) => RecursionPolyAirLeaf::Main { col },
        SymbolicVar::Public(index) => RecursionPolyAirLeaf::Public { index },
        SymbolicVar::Alpha => RecursionPolyAirLeaf::PermAlpha,
        SymbolicVar::BetaPowers(power) => RecursionPolyAirLeaf::BetaPower { power },
        SymbolicVar::BetaSeptix => RecursionPolyAirLeaf::BetaSeptix,
        SymbolicVar::Precomputed(index, _) => RecursionPolyAirLeaf::Precomputed { index },
        SymbolicVar::ReservedPoly(index, _) => RecursionPolyAirLeaf::ReservedPoly { index },
        SymbolicVar::IsFirstRow => RecursionPolyAirLeaf::IsFirstRow,
        SymbolicVar::IsLastRow => RecursionPolyAirLeaf::IsLastRow,
        // NATIVE_REC_TODO_DELETE: support IsTransition after transition-selector
        // replay fixtures are implemented.
        SymbolicVar::IsTransition => {
            return Err(RecursionAdapterError::UnsupportedTransitionSelector)
        }
    };
    Ok(leaf)
}

fn degree_multiple_u16(expr: &RecursionSymbolicExpression) -> Result<u16, RecursionAdapterError> {
    let degree_multiple = degree_multiple(expr);
    u16::try_from(degree_multiple)
        .map_err(|_| RecursionAdapterError::DegreeMultipleTooLarge { degree_multiple })
}

fn degree_multiple(expr: &RecursionSymbolicExpression) -> usize {
    match expr {
        SymbolicExpression::VARiable(var) => match var {
            SymbolicVar::Preprocessed(_) |
            SymbolicVar::Main(_) |
            SymbolicVar::Precomputed(_, _) |
            SymbolicVar::ReservedPoly(_, _) |
            SymbolicVar::IsFirstRow |
            SymbolicVar::IsLastRow |
            SymbolicVar::IsTransition => 1,
            SymbolicVar::Public(_) |
            SymbolicVar::Alpha |
            SymbolicVar::BetaPowers(_) |
            SymbolicVar::BetaSeptix => 0,
        },
        SymbolicExpression::Constant(_) | SymbolicExpression::ConstantExt(_) => 0,
        SymbolicExpression::Add { degree_multiple, .. } |
        SymbolicExpression::Sub { degree_multiple, .. } |
        SymbolicExpression::Neg { degree_multiple, .. } |
        SymbolicExpression::Mul { degree_multiple, .. } => *degree_multiple,
    }
}

fn expr_ptr(expr: &RecursionSymbolicExpression) -> usize {
    expr as *const RecursionSymbolicExpression as usize
}

fn struct_cache_for_nodes(nodes: &[RecursionPolyAirNode]) -> HashMap<Vec<u8>, u32> {
    let mut cache = HashMap::with_capacity(nodes.len());
    for node in nodes {
        cache.insert(op_key(&node.op), node.node_id);
    }
    cache
}

#[cfg(test)]
mod tests {
    use super::*;
    use dt_stark::air::FullAirBuilder;
    use p3_field::{AbstractExtensionField, AbstractField};

    fn public_leaf(index: usize) -> RecursionSymbolicExpression {
        SymbolicExpression::VARiable(SymbolicVar::Public(index))
    }

    #[test]
    fn structurally_identical_roots_share_nodes() {
        let adapter = RecursionSymbolicExprAdapter::default();
        let mut nodes = Vec::new();
        let a = public_leaf(0) + F::one();
        let b = public_leaf(0) + F::one();

        let root_a =
            adapter.adapt_root(RecursionRootKind::PrecomputeLc, 0, &a, &mut nodes).unwrap();
        let root_b =
            adapter.adapt_root(RecursionRootKind::PrecomputeLc, 1, &b, &mut nodes).unwrap();

        assert_eq!(root_a.root_node_id, root_b.root_node_id);
        assert_eq!(nodes.len(), 3);
    }

    #[test]
    fn neg_lowers_to_existing_sub_opcode() {
        let adapter = RecursionSymbolicExprAdapter::default();
        let expression = -public_leaf(0);
        let adapted = adapter.adapt_roots(RecursionRootKind::PrecomputeLc, &[expression]).unwrap();

        assert_eq!(adapted.op_mix.negs, 0);
        assert_eq!(adapted.op_mix.subs, 1);
        let root = adapted.roots[0].root_node_id;
        let (lhs, rhs) = match &adapted.node_table[root as usize].op {
            RecursionPolyAirOp::Sub { lhs, rhs } => (*lhs, *rhs),
            _ => panic!("negation must lower to subtraction"),
        };
        assert!(matches!(
            adapted.node_table[lhs as usize].op,
            RecursionPolyAirOp::ConstBase(value) if value == F::zero()
        ));
        assert!(matches!(
            adapted.node_table[rhs as usize].op,
            RecursionPolyAirOp::Leaf(RecursionPolyAirLeaf::Public { index: 0 })
        ));
    }

    #[test]
    fn gate_roots_reject_raw_main_leaves() {
        let adapter = RecursionSymbolicExprAdapter::default();
        let mut nodes = Vec::new();
        let expr = SymbolicExpression::VARiable(SymbolicVar::Main(0));

        let err = adapter.adapt_root(RecursionRootKind::Gate, 7, &expr, &mut nodes).unwrap_err();

        assert!(matches!(err, RecursionAdapterError::RawTraceLeafInGateRoot { root_index: 7, .. }));
    }

    #[test]
    fn rejects_transition_selector() {
        let adapter = RecursionSymbolicExprAdapter::default();
        let mut nodes = Vec::new();
        let expr = SymbolicExpression::VARiable(SymbolicVar::IsTransition);

        let err =
            adapter.adapt_root(RecursionRootKind::PrecomputeLc, 0, &expr, &mut nodes).unwrap_err();

        assert_eq!(err, RecursionAdapterError::UnsupportedTransitionSelector);
    }

    #[test]
    fn extension_constants_are_canonical_and_theta_only() {
        let adapter = RecursionSymbolicExprAdapter::default();
        let theta = <EF as AbstractExtensionField<F>>::monomial(1);
        let base = <EF as AbstractExtensionField<F>>::from_base_fn(|idx| {
            if idx == 0 {
                F::from_canonical_u32(9)
            } else {
                F::zero()
            }
        });

        assert_eq!(
            classify_extension_constant(&base),
            CanonicalExtensionConstant::Base(F::from_canonical_u32(9))
        );
        assert_eq!(classify_extension_constant(&theta), CanonicalExtensionConstant::Theta);

        let base_root = adapter
            .adapt_roots(RecursionRootKind::PrecomputeLc, &[SymbolicExpression::from_ext(base)])
            .expect("base-valued extension constant canonicalizes");
        assert!(matches!(
            base_root.node_table[base_root.roots[0].root_node_id as usize].op,
            RecursionPolyAirOp::ConstBase(value) if value == F::from_canonical_u32(9)
        ));

        let theta_root = adapter
            .adapt_roots(RecursionRootKind::PrecomputeLc, &[SymbolicExpression::from_ext(theta)])
            .expect("theta is the one supported extension constant");
        assert!(matches!(
            theta_root.node_table[theta_root.roots[0].root_node_id as usize].op,
            RecursionPolyAirOp::ConstExt(value) if value == theta
        ));

        let other = theta * theta;
        assert_eq!(classify_extension_constant(&other), CanonicalExtensionConstant::Unsupported);
        assert_eq!(
            adapter
                .adapt_roots(
                    RecursionRootKind::PrecomputeLc,
                    &[SymbolicExpression::from_ext(other)],
                )
                .unwrap_err(),
            RecursionAdapterError::UnsupportedExtensionConstant
        );
    }

    #[test]
    fn symbolic_pack_uses_protocol_theta_ast() {
        let limbs = (0..D_EF)
            .map(|idx| SymbolicExpression::VARiable(SymbolicVar::Main(idx)))
            .collect::<Vec<_>>();
        let packed =
            <polyair::symbolic::SymbolicAirBuilder<F, D_EF> as FullAirBuilder>::pack_ext_limbs(
                &limbs,
            );
        let adapted = RecursionSymbolicExprAdapter::default()
            .adapt_roots(RecursionRootKind::PrecomputeLc, &[packed])
            .expect("symbolic packing is accepted by the native adapter");
        assert_eq!(adapted.op_mix.const_ext, 1);
        assert!(adapted.node_table.iter().any(|node| {
            matches!(
                node.op,
                RecursionPolyAirOp::Mul { .. } | RecursionPolyAirOp::FusedMulAdd { .. }
            )
        }));
        assert!(adapted.node_table.iter().any(|node| {
            matches!(
                node.op,
                RecursionPolyAirOp::ConstExt(value)
                    if value == <EF as AbstractExtensionField<F>>::monomial(1)
            )
        }));
    }
}
