//! The pure count-known tree planners (`MinNodesEvenV1` and
//! `BandAwareEvenV1`) and their immutable `TreePlan` output — S1 of the
//! final-route implementation
//! (`docs/sol-final-op.md` V3 §6-§7).
//!
//! Pure: no I/O, no proof data, no timing. Topology is a function of
//! declared count, arity, policy, worker, and integer band-table inputs only.
//!
//! Policy summary (V3 §6.2):
//! - Lift layer: `p = ceil(C/A)` nodes (minimum), spans split evenly (sizes differ by at most one,
//!   ceil-sized groups first).
//! - L2 rounds while the frontier exceeds `A`: final round (`ceil(C/A) <= A`): the minimum number
//!   of reducers `q = max(1, ceil((C-A)/(A-1)))`, reducers filled to `A` first, the overflow tail
//!   carried; intermediate round: maximal progress `next = ceil(C/A)` with the minimum reducers for
//!   that width `q = ceil((C-next)/(A-1))`.
//! - Carries forward the same owning proof at zero cost; output bindings are path-compressed past
//!   carries to the eventual real parent.
//!
//! Deliberately NOT globally proof-minimal across multi-round trees
//! (V3 §6.3; the C=123 frontier golden below pins the accepted trade).

use std::num::{NonZeroU32, NonZeroU8};

use crate::native_backend::NATIVE_MAX_NODE_ARITY;

/// Policy version tag carried by every plan this module produces.
pub const TREE_POLICY_MIN_NODES_EVEN_V1: u32 = 1;
pub const TREE_POLICY_BAND_AWARE_EVEN_V1: u32 = 2;

/// Inclusive arity breakpoints for the three node kinds affected by lift
/// splitting. Breakpoints are strictly increasing and terminate at the keyed
/// arity cap. Their ordinal is the integer cost band; no live timing enters
/// planning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArityBandTable {
    pub lift: Vec<u8>,
    pub l2: Vec<u8>,
    pub l3: Vec<u8>,
}

impl ArityBandTable {
    /// Current calibrated native shape. Warm-cache native RSP runs found a
    /// persistent roughly-2x Lift prove-cost step from k=9 to k=10 under both
    /// W=1 and W=2. The dominant height stays at 2^21, but Poseidon2 and eight
    /// other chips each rise one padded band, certifying an aggregate Lift
    /// breakpoint at 9. L3 k=2 remains below k=3; L2 has no certified arity
    /// cliff inside the keyed range.
    pub fn current_native(arity_cap: u8) -> Result<Self, PlanError> {
        check_arity_cap(arity_cap)?;
        let lift = if arity_cap <= 9 { vec![arity_cap] } else { vec![9, arity_cap] };
        let l3 = if arity_cap <= 2 { vec![arity_cap] } else { vec![2, arity_cap] };
        Ok(Self { lift, l2: vec![arity_cap], l3 })
    }

    /// A single-band table is the conservative fallback and must reproduce
    /// `MinNodesEvenV1` exactly.
    pub fn single_band(arity_cap: u8) -> Result<Self, PlanError> {
        check_arity_cap(arity_cap)?;
        Ok(Self { lift: vec![arity_cap], l2: vec![arity_cap], l3: vec![arity_cap] })
    }

    fn validate(&self, arity_cap: u8) -> Result<(), PlanError> {
        for bands in [&self.lift, &self.l2, &self.l3] {
            if bands.is_empty() ||
                bands.last().copied() != Some(arity_cap) ||
                bands.iter().any(|&band| band == 0 || band > arity_cap) ||
                bands.windows(2).any(|window| window[0] >= window[1])
            {
                return Err(PlanError::Invariant(
                    "band breakpoints must be strictly increasing and end at arity cap",
                ));
            }
        }
        Ok(())
    }

    fn band_of(&self, kind: NodeKind, arity: u32) -> Result<usize, PlanError> {
        let bands = match kind {
            NodeKind::Lift => &self.lift,
            NodeKind::L2 => &self.l2,
            NodeKind::L3 => &self.l3,
            NodeKind::L4 => {
                return Err(PlanError::Invariant("L4 has no arity band input"));
            }
        };
        bands
            .iter()
            .position(|&breakpoint| arity <= u32::from(breakpoint))
            .ok_or(PlanError::Invariant("arity outside band table"))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BandAwareEvenV1Inputs {
    pub worker_hint: NonZeroU8,
    pub bands: ArityBandTable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId {
    /// 0 = lift layer, 1.. = L2 rounds, then L3, then L4.
    pub level: u16,
    /// Ordinal within the level, in shard order.
    pub ordinal: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeKind {
    Lift,
    L2,
    L3,
    L4,
}

/// Half-open span of core shard ordinals covered by a node or child.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShardSpan {
    pub start: u32,
    pub end: u32,
}

impl ShardSpan {
    pub fn len(&self) -> u32 {
        self.end - self.start
    }

    pub fn is_empty(&self) -> bool {
        self.start >= self.end
    }
}

/// The proof class a child presents to its parent's statement/replay layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChildClass {
    Core,
    Lift,
    L2,
    L3,
}

/// Static replay-segment base the child is recorded through. `Base0` is the
/// first (u-segment) universe of the parent program; `Base128` is the mixed
/// second segment (`MIXED_SEGMENT_STATIC_CHIP_ID_OFFSET`). Only L2-class
/// children of L2/L3 parents use `Base128`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplaySegment {
    Base0,
    Base128,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SourceNodeId {
    CoreShard(u32),
    Node(NodeId),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChildBinding {
    pub source: SourceNodeId,
    pub child_class: ChildClass,
    pub span: ShardSpan,
    pub local_slot: u8,
    pub replay_segment: ReplaySegment,
}

/// Where a produced proof is consumed: its one real parent and dense slot.
/// Path-compressed: a proof carried through L2 rounds binds directly to the
/// eventual consuming node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutputBinding {
    pub parent: NodeId,
    pub parent_slot: u8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodePlan {
    pub id: NodeId,
    pub kind: NodeKind,
    pub span: ShardSpan,
    pub children: Vec<ChildBinding>,
    pub expected_arity: NonZeroU8,
    /// `None` only for the L4 root.
    pub output: Option<OutputBinding>,
}

/// A frontier element forwarded to a later round without a proof, replay, or
/// record. Recorded for auditability; routing uses the producer's
/// (path-compressed) `OutputBinding`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CarryPlan {
    pub source: SourceNodeId,
    pub span: ShardSpan,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodeAction {
    Reduce(NodePlan),
    Carry(CarryPlan),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerPlan {
    /// 0 = lift layer; 1.. = L2 rounds.
    pub depth: u16,
    /// Frontier width entering this layer (core count at depth 0).
    pub input_count: u32,
    pub actions: Vec<NodeAction>,
}

/// The immutable count-derived tree. Built purely from
/// (core_count, arity cap); contains no proof, record, timing, or device
/// state. There is no append, replan, or runtime cut mutation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TreePlan {
    pub version: u32,
    pub arity_cap: u8,
    pub core_count: u32,
    /// Present only when BandAwareEvenV1 selected a topology different from
    /// V1. A guarded fallback is byte-for-byte the ordinary V1 plan.
    pub band_aware_inputs: Option<BandAwareEvenV1Inputs>,
    /// Layer 0 is the lift layer; layers 1.. are L2 rounds (possibly none).
    pub layers: Vec<LayerPlan>,
    pub l3: NodePlan,
    pub l4: NodePlan,
}

/// Resolved route for one ready child proof: its final parent, dense slot,
/// class, and replay segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChildRoute {
    pub parent: NodeId,
    pub parent_kind: NodeKind,
    pub local_slot: u8,
    pub span: ShardSpan,
    pub child_class: ChildClass,
    pub replay_segment: ReplaySegment,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanError {
    /// `A` must satisfy `2 <= A <= NATIVE_MAX_NODE_ARITY`.
    ArityCapOutOfRange { arity_cap: u8 },
    /// Checked arithmetic failed (counts near the type boundary).
    Overflow(&'static str),
    /// An internal planner invariant failed validation; always a bug.
    Invariant(&'static str),
}

impl std::fmt::Display for PlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ArityCapOutOfRange { arity_cap } => {
                write!(f, "arity cap {arity_cap} outside 2..={NATIVE_MAX_NODE_ARITY}")
            }
            Self::Overflow(what) => write!(f, "planner arithmetic overflow: {what}"),
            Self::Invariant(what) => write!(f, "planner invariant violated: {what}"),
        }
    }
}

impl std::error::Error for PlanError {}

fn ceil_div(n: u32, d: u32) -> Result<u32, PlanError> {
    if d == 0 {
        return Err(PlanError::Overflow("ceil_div by zero"));
    }
    n.checked_add(d - 1).map(|v| v / d).ok_or(PlanError::Overflow("ceil_div"))
}

fn check_arity_cap(arity_cap: u8) -> Result<(), PlanError> {
    if arity_cap < 2 || (arity_cap as usize) > NATIVE_MAX_NODE_ARITY {
        return Err(PlanError::ArityCapOutOfRange { arity_cap });
    }
    Ok(())
}

/// Even split of `total` items over `parts` groups: sizes differ by at most
/// one, ceil-sized groups first (the lightest group last).
fn even_split(total: u32, parts: u32) -> Result<Vec<u32>, PlanError> {
    if parts == 0 || total < parts {
        return Err(PlanError::Invariant("even_split: parts must be 1..=total"));
    }
    let floor = total / parts;
    let remainder = (total % parts) as usize;
    let mut sizes = vec![floor; parts as usize];
    for size in sizes.iter_mut().take(remainder) {
        *size += 1;
    }
    Ok(sizes)
}

/// One L2 round decision over a frontier of `c` proofs (`c > a`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct L2RoundShape {
    pub reducers: u32,
    pub carries: u32,
    /// Reducer child counts, prefix-first over the frontier.
    pub sizes: Vec<u32>,
    /// Whether this round's output reaches `<= a` (the final round).
    pub is_final: bool,
}

impl L2RoundShape {
    pub fn next_frontier(&self) -> u32 {
        self.reducers + self.carries
    }
}

/// The `MinNodesEvenV1` L2 round rule (V3 §6.2). `c` must exceed `a`.
pub fn l2_round_shape(c: u32, a: u32) -> Result<L2RoundShape, PlanError> {
    if c <= a {
        return Err(PlanError::Invariant("l2_round_shape requires c > a"));
    }
    let out_min = ceil_div(c, a)?;
    let (reducers, carries, is_final) = if out_min <= a {
        // FINAL round: minimum new proofs; fill reducers to `a`, carry the
        // overflow tail. `saturating_sub` guards the c <= a*q case (e.g.
        // c=115, a=11, q=11 -> 121 > 115).
        let q = ceil_div(c - a, a - 1)?.max(1);
        let consumed = a.checked_mul(q).ok_or(PlanError::Overflow("a*q"))?;
        (q, c.saturating_sub(consumed), true)
    } else {
        // INTERMEDIATE round: maximal progress at minimal reducers.
        let q = ceil_div(c - out_min, a - 1)?;
        (q, out_min - q, false)
    };
    let reduced = c - carries;
    let sizes = even_split(reduced, reducers)?;
    let shape = L2RoundShape { reducers, carries, sizes, is_final };
    // Local invariants (validated again plan-wide later).
    if shape.sizes.iter().any(|&s| s < 2 || s > a) {
        return Err(PlanError::Invariant("l2 reducer size outside 2..=a"));
    }
    if shape.next_frontier() >= c {
        return Err(PlanError::Invariant("l2 round made no progress"));
    }
    if shape.is_final && shape.next_frontier() > a {
        return Err(PlanError::Invariant("final round exceeded arity cap"));
    }
    if !shape.is_final && shape.next_frontier() != out_min {
        return Err(PlanError::Invariant("intermediate round not maximal-progress"));
    }
    Ok(shape)
}

/// A frontier element during construction.
#[derive(Debug, Clone, Copy)]
struct FrontierItem {
    source: SourceNodeId,
    class: ChildClass,
    span: ShardSpan,
}

fn segment_for(parent: NodeKind, class: ChildClass) -> Result<ReplaySegment, PlanError> {
    match (parent, class) {
        (NodeKind::Lift, ChildClass::Core) => Ok(ReplaySegment::Base0),
        (NodeKind::L2 | NodeKind::L3, ChildClass::Lift) => Ok(ReplaySegment::Base0),
        (NodeKind::L2 | NodeKind::L3, ChildClass::L2) => Ok(ReplaySegment::Base128),
        (NodeKind::L4, ChildClass::L3) => Ok(ReplaySegment::Base0),
        _ => Err(PlanError::Invariant("illegal parent/child class pair")),
    }
}

fn bind_children(
    parent_kind: NodeKind,
    items: &[FrontierItem],
) -> Result<Vec<ChildBinding>, PlanError> {
    if items.len() > u8::MAX as usize {
        return Err(PlanError::Overflow("child slot"));
    }
    items
        .iter()
        .enumerate()
        .map(|(slot, item)| {
            Ok(ChildBinding {
                source: item.source,
                child_class: item.class,
                span: item.span,
                local_slot: slot as u8,
                replay_segment: segment_for(parent_kind, item.class)?,
            })
        })
        .collect()
}

fn span_of(items: &[FrontierItem]) -> ShardSpan {
    ShardSpan { start: items[0].span.start, end: items[items.len() - 1].span.end }
}

/// Build the `MinNodesEvenV1` plan for `core_count` shards at cap `arity_cap`.
pub fn plan_min_nodes_even_v1(
    core_count: NonZeroU32,
    arity_cap: u8,
) -> Result<TreePlan, PlanError> {
    check_arity_cap(arity_cap)?;
    let a = arity_cap as u32;
    let c = core_count.get();
    let lift_count = ceil_div(c, a)?;
    build_plan(core_count, arity_cap, lift_count, TREE_POLICY_MIN_NODES_EVEN_V1, None)
}

/// Build BandAwareEvenV1. The guarded policy returns the exact V1 plan when
/// no candidate has both a strict Lift band drop and a strict predicted
/// makespan improvement.
pub fn plan_band_aware_even_v1(
    core_count: NonZeroU32,
    arity_cap: u8,
    worker_hint: NonZeroU8,
    bands: ArityBandTable,
) -> Result<TreePlan, PlanError> {
    check_arity_cap(arity_cap)?;
    bands.validate(arity_cap)?;
    let c = core_count.get();
    let a = u32::from(arity_cap);
    let v1_lift_count = ceil_div(c, a)?;
    let lift_count = choose_band_aware_lift_count(c, a, u32::from(worker_hint.get()), &bands)?;
    if lift_count == v1_lift_count {
        return plan_min_nodes_even_v1(core_count, arity_cap);
    }
    build_plan(
        core_count,
        arity_cap,
        lift_count,
        TREE_POLICY_BAND_AWARE_EVEN_V1,
        Some(BandAwareEvenV1Inputs { worker_hint, bands }),
    )
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ParentBandEnvelope {
    l2_round_max: Vec<usize>,
    l3: usize,
}

fn l2_round_count(mut width: u32, arity_cap: u32) -> Result<usize, PlanError> {
    let mut rounds = 0usize;
    while width > arity_cap {
        width = l2_round_shape(width, arity_cap)?.next_frontier();
        rounds = rounds.checked_add(1).ok_or(PlanError::Overflow("l2 round count"))?;
    }
    Ok(rounds)
}

fn parent_band_envelope(
    mut width: u32,
    arity_cap: u32,
    bands: &ArityBandTable,
) -> Result<ParentBandEnvelope, PlanError> {
    let mut l2_round_max = Vec::new();
    while width > arity_cap {
        let shape = l2_round_shape(width, arity_cap)?;
        let max_arity =
            shape.sizes.iter().copied().max().ok_or(PlanError::Invariant("empty l2 shape"))?;
        l2_round_max.push(bands.band_of(NodeKind::L2, max_arity)?);
        width = shape.next_frontier();
    }
    Ok(ParentBandEnvelope { l2_round_max, l3: bands.band_of(NodeKind::L3, width)? })
}

fn parent_bands_preserved(candidate: &ParentBandEnvelope, v1: &ParentBandEnvelope) -> bool {
    candidate.l2_round_max.len() == v1.l2_round_max.len() &&
        candidate
            .l2_round_max
            .iter()
            .zip(&v1.l2_round_max)
            .all(|(candidate, v1)| candidate <= v1) &&
        candidate.l3 <= v1.l3
}

fn choose_band_aware_lift_count(
    core_count: u32,
    arity_cap: u32,
    worker_hint: u32,
    bands: &ArityBandTable,
) -> Result<u32, PlanError> {
    if worker_hint == 0 {
        return Err(PlanError::Invariant("worker hint must be nonzero"));
    }
    let v1_count = ceil_div(core_count, arity_cap)?;
    let v1_max_arity = ceil_div(core_count, v1_count)?;
    let v1_band = bands.band_of(NodeKind::Lift, v1_max_arity)?;
    let v1_rounds = l2_round_count(v1_count, arity_cap)?;
    let v1_parent = parent_band_envelope(v1_count, arity_cap, bands)?;
    let v1_waves = ceil_div(v1_count, worker_hint)?;
    let v1_cost = 1u128.checked_shl(v1_band as u32).ok_or(PlanError::Overflow("lift band cost"))?;
    let mut best = (v1_waves as u128 * v1_cost, v1_count);

    // Only the smallest node count entering each cheaper Lift band can win:
    // within one band, extra nodes increase waves without reducing node cost.
    for &breakpoint in bands.lift.iter().take(v1_band) {
        let candidate = ceil_div(core_count, u32::from(breakpoint))?;
        if candidate <= v1_count ||
            candidate > core_count ||
            l2_round_count(candidate, arity_cap)? != v1_rounds
        {
            continue;
        }
        let candidate_max_arity = ceil_div(core_count, candidate)?;
        let candidate_band = bands.band_of(NodeKind::Lift, candidate_max_arity)?;
        if candidate_band >= v1_band {
            continue;
        }
        let candidate_parent = parent_band_envelope(candidate, arity_cap, bands)?;
        if !parent_bands_preserved(&candidate_parent, &v1_parent) {
            continue;
        }
        let waves = ceil_div(candidate, worker_hint)?;
        let cost = 1u128
            .checked_shl(candidate_band as u32)
            .ok_or(PlanError::Overflow("lift band cost"))?;
        let makespan = waves as u128 * cost;
        if makespan < best.0 || (makespan == best.0 && candidate < best.1) {
            best = (makespan, candidate);
        }
    }
    Ok(best.1)
}

fn build_plan(
    core_count: NonZeroU32,
    arity_cap: u8,
    lift_count: u32,
    version: u32,
    band_aware_inputs: Option<BandAwareEvenV1Inputs>,
) -> Result<TreePlan, PlanError> {
    let a = arity_cap as u32;
    let c = core_count.get();
    if lift_count == 0 || lift_count > c || ceil_div(c, lift_count)? > a {
        return Err(PlanError::Invariant("invalid lift count"));
    }
    let mut layers = Vec::new();

    // Layer 0: lifts — policy-selected count, even spans, ceil-sized groups first.
    let lift_sizes = even_split(c, lift_count)?;
    let mut frontier = Vec::with_capacity(lift_count as usize);
    let mut lift_actions = Vec::with_capacity(lift_count as usize);
    let mut cursor = 0u32;
    for (ordinal, &size) in lift_sizes.iter().enumerate() {
        let id = NodeId { level: 0, ordinal: ordinal as u32 };
        let span = ShardSpan { start: cursor, end: cursor + size };
        cursor += size;
        let items: Vec<FrontierItem> = (span.start..span.end)
            .map(|shard| FrontierItem {
                source: SourceNodeId::CoreShard(shard),
                class: ChildClass::Core,
                span: ShardSpan { start: shard, end: shard + 1 },
            })
            .collect();
        let children = bind_children(NodeKind::Lift, &items)?;
        lift_actions.push(NodeAction::Reduce(NodePlan {
            id,
            kind: NodeKind::Lift,
            span,
            children,
            expected_arity: NonZeroU8::new(size as u8).ok_or(PlanError::Invariant("empty lift"))?,
            output: None,
        }));
        frontier.push(FrontierItem {
            source: SourceNodeId::Node(id),
            class: ChildClass::Lift,
            span,
        });
    }
    layers.push(LayerPlan { depth: 0, input_count: c, actions: lift_actions });

    // L2 rounds.
    let mut depth: u16 = 0;
    while frontier.len() as u32 > a {
        depth = depth.checked_add(1).ok_or(PlanError::Overflow("depth"))?;
        let width = frontier.len() as u32;
        let shape = l2_round_shape(width, a)?;
        let mut actions = Vec::with_capacity((shape.reducers + shape.carries) as usize);
        let mut next_frontier = Vec::with_capacity(shape.next_frontier() as usize);
        let mut offset = 0usize;
        for (ordinal, &size) in shape.sizes.iter().enumerate() {
            let id = NodeId { level: depth, ordinal: ordinal as u32 };
            let items = &frontier[offset..offset + size as usize];
            offset += size as usize;
            let span = span_of(items);
            let children = bind_children(NodeKind::L2, items)?;
            actions.push(NodeAction::Reduce(NodePlan {
                id,
                kind: NodeKind::L2,
                span,
                children,
                expected_arity: NonZeroU8::new(size as u8)
                    .ok_or(PlanError::Invariant("empty l2"))?,
                output: None,
            }));
            next_frontier.push(FrontierItem {
                source: SourceNodeId::Node(id),
                class: ChildClass::L2,
                span,
            });
        }
        for item in &frontier[offset..] {
            actions.push(NodeAction::Carry(CarryPlan { source: item.source, span: item.span }));
            next_frontier.push(*item);
        }
        layers.push(LayerPlan { depth, input_count: width, actions });
        if next_frontier.len() >= frontier.len() {
            return Err(PlanError::Invariant("round failed to reduce the frontier"));
        }
        frontier = next_frontier;
    }

    // L3 over the final frontier; L4 over L3.
    let l3_id = NodeId { level: depth + 1, ordinal: 0 };
    let l3_span = span_of(&frontier);
    let l3 = NodePlan {
        id: l3_id,
        kind: NodeKind::L3,
        span: l3_span,
        children: bind_children(NodeKind::L3, &frontier)?,
        expected_arity: NonZeroU8::new(frontier.len() as u8)
            .ok_or(PlanError::Invariant("empty l3"))?,
        output: None,
    };
    let l4_id = NodeId { level: depth + 2, ordinal: 0 };
    let l4 = NodePlan {
        id: l4_id,
        kind: NodeKind::L4,
        span: l3_span,
        children: vec![ChildBinding {
            source: SourceNodeId::Node(l3_id),
            child_class: ChildClass::L3,
            span: l3_span,
            local_slot: 0,
            replay_segment: ReplaySegment::Base0,
        }],
        expected_arity: NonZeroU8::new(1).expect("1 != 0"),
        output: None,
    };

    let mut plan =
        TreePlan { version, arity_cap, core_count: c, band_aware_inputs, layers, l3, l4 };
    fill_output_bindings(&mut plan)?;
    validate(&plan)?;
    Ok(plan)
}

/// Fill every produced node's one `OutputBinding` from the child bindings
/// that consume it (path compression falls out naturally: carries never
/// create bindings, so a carried proof binds straight to its real parent).
fn fill_output_bindings(plan: &mut TreePlan) -> Result<(), PlanError> {
    use std::collections::BTreeMap;
    let mut consumers: BTreeMap<NodeId, OutputBinding> = BTreeMap::new();
    let mut record = |parent: NodeId, children: &[ChildBinding]| -> Result<(), PlanError> {
        for child in children {
            if let SourceNodeId::Node(id) = child.source {
                let binding = OutputBinding { parent, parent_slot: child.local_slot };
                if consumers.insert(id, binding).is_some() {
                    return Err(PlanError::Invariant("node consumed twice"));
                }
            }
        }
        Ok(())
    };
    for layer in &plan.layers {
        for action in &layer.actions {
            if let NodeAction::Reduce(node) = action {
                record(node.id, &node.children)?;
            }
        }
    }
    record(plan.l3.id, &plan.l3.children)?;
    record(plan.l4.id, &plan.l4.children)?;

    let mut missing = 0usize;
    for layer in &mut plan.layers {
        for action in &mut layer.actions {
            if let NodeAction::Reduce(node) = action {
                match consumers.get(&node.id) {
                    Some(binding) => node.output = Some(*binding),
                    None => missing += 1,
                }
            }
        }
    }
    match consumers.get(&plan.l3.id) {
        Some(binding) => plan.l3.output = Some(*binding),
        None => missing += 1,
    }
    if missing != 0 {
        return Err(PlanError::Invariant("produced node without a consumer"));
    }
    if plan.l4.output.is_some() {
        return Err(PlanError::Invariant("l4 must not have an output binding"));
    }
    Ok(())
}

impl TreePlan {
    /// Iterate every Reduce node (lift + L2 layers + L3 + L4).
    pub fn nodes(&self) -> impl Iterator<Item = &NodePlan> {
        self.layers
            .iter()
            .flat_map(|layer| {
                layer.actions.iter().filter_map(|action| match action {
                    NodeAction::Reduce(node) => Some(node),
                    NodeAction::Carry(_) => None,
                })
            })
            .chain([&self.l3, &self.l4])
    }

    fn node(&self, id: NodeId) -> Option<&NodePlan> {
        self.nodes().find(|node| node.id == id)
    }

    /// The route of one core shard proof into its lift.
    pub fn core_route(&self, ordinal: u32) -> Option<ChildRoute> {
        let lift_layer = self.layers.first()?;
        for action in &lift_layer.actions {
            if let NodeAction::Reduce(node) = action {
                if ordinal >= node.span.start && ordinal < node.span.end {
                    let child = node
                        .children
                        .iter()
                        .find(|child| child.source == SourceNodeId::CoreShard(ordinal))?;
                    return Some(ChildRoute {
                        parent: node.id,
                        parent_kind: node.kind,
                        local_slot: child.local_slot,
                        span: child.span,
                        child_class: child.child_class,
                        replay_segment: child.replay_segment,
                    });
                }
            }
        }
        None
    }

    /// The route of one produced node proof into its (path-compressed) parent.
    /// `None` for L4 (the root has no consumer).
    pub fn node_route(&self, id: NodeId) -> Option<ChildRoute> {
        let node = self.node(id)?;
        let output = node.output?;
        let parent = self.node(output.parent)?;
        let child = parent.children.get(output.parent_slot as usize)?;
        debug_assert_eq!(child.source, SourceNodeId::Node(id));
        Some(ChildRoute {
            parent: parent.id,
            parent_kind: parent.kind,
            local_slot: child.local_slot,
            span: child.span,
            child_class: child.child_class,
            replay_segment: child.replay_segment,
        })
    }

    /// Total proof nodes (lifts + L2s + L3 + L4).
    pub fn proof_node_count(&self) -> usize {
        self.nodes().count()
    }

    /// L2 proof count across all rounds.
    pub fn l2_proof_count(&self) -> usize {
        self.layers
            .iter()
            .skip(1)
            .flat_map(|layer| &layer.actions)
            .filter(|action| matches!(action, NodeAction::Reduce(_)))
            .count()
    }
}

/// Plan-wide validation (V3 §8.2 / §13.1). Also exercised directly by tests.
pub fn validate(plan: &TreePlan) -> Result<(), PlanError> {
    let a = plan.arity_cap as u32;
    let c = plan.core_count;
    match (plan.version, &plan.band_aware_inputs) {
        (TREE_POLICY_MIN_NODES_EVEN_V1, None) => {}
        (TREE_POLICY_BAND_AWARE_EVEN_V1, Some(inputs)) => {
            inputs.bands.validate(plan.arity_cap)?;
        }
        _ => return Err(PlanError::Invariant("policy version/input mismatch")),
    }

    // Lift layer: exact coverage of 0..c in order and policy-selected node
    // count, with even spans (differ <= 1, non-increasing).
    let lift_layer = plan.layers.first().ok_or(PlanError::Invariant("no lift layer"))?;
    if lift_layer.depth != 0 || lift_layer.input_count != c {
        return Err(PlanError::Invariant("malformed lift layer header"));
    }
    let mut cursor = 0u32;
    let mut lift_sizes = Vec::new();
    for action in &lift_layer.actions {
        let node = match action {
            NodeAction::Reduce(node) => node,
            NodeAction::Carry(_) => return Err(PlanError::Invariant("carry in the lift layer")),
        };
        if node.kind != NodeKind::Lift || node.span.start != cursor || node.span.is_empty() {
            return Err(PlanError::Invariant("lift span not contiguous"));
        }
        if node.span.len() > a {
            return Err(PlanError::Invariant("lift arity above cap"));
        }
        check_node(node)?;
        lift_sizes.push(node.span.len());
        cursor = node.span.end;
    }
    if cursor != c {
        return Err(PlanError::Invariant("lifts do not cover the core span"));
    }
    let actual_lift_count = lift_sizes.len() as u32;
    let expected_lift_count = match &plan.band_aware_inputs {
        None => ceil_div(c, a)?,
        Some(inputs) => {
            choose_band_aware_lift_count(c, a, u32::from(inputs.worker_hint.get()), &inputs.bands)?
        }
    };
    if actual_lift_count != expected_lift_count {
        return Err(PlanError::Invariant("lift count differs from policy"));
    }
    if plan.version == TREE_POLICY_BAND_AWARE_EVEN_V1 {
        let v1_lift_count = ceil_div(c, a)?;
        if actual_lift_count == v1_lift_count ||
            l2_round_count(actual_lift_count, a)? != l2_round_count(v1_lift_count, a)?
        {
            return Err(PlanError::Invariant("band-aware plan did not preserve the V1 round count"));
        }
    }
    let max = *lift_sizes.iter().max().unwrap();
    let min = *lift_sizes.iter().min().unwrap();
    if max - min > 1 || lift_sizes.windows(2).any(|w| w[0] < w[1]) {
        return Err(PlanError::Invariant("lift spans not even/ceil-first"));
    }

    // L2 rounds: widths, per-round predicate, span concatenation, carries as
    // contiguous tail.
    let mut width = actual_lift_count;
    for layer in plan.layers.iter().skip(1) {
        if layer.input_count != width {
            return Err(PlanError::Invariant("layer input_count mismatch"));
        }
        if width <= a {
            return Err(PlanError::Invariant("l2 round on an admissible frontier"));
        }
        let shape = l2_round_shape(width, a)?;
        let mut reducers = 0u32;
        let mut carries = 0u32;
        let mut seen_carry = false;
        let mut cursor: Option<u32> = None;
        for action in &layer.actions {
            let (span, is_reduce) = match action {
                NodeAction::Reduce(node) => {
                    if seen_carry {
                        return Err(PlanError::Invariant("reduce after carry"));
                    }
                    if node.kind != NodeKind::L2 {
                        return Err(PlanError::Invariant("non-l2 node in l2 layer"));
                    }
                    if node.children.len() < 2 || node.children.len() as u32 > a {
                        return Err(PlanError::Invariant("l2 arity outside 2..=a"));
                    }
                    check_node(node)?;
                    (node.span, true)
                }
                NodeAction::Carry(carry) => {
                    seen_carry = true;
                    (carry.span, false)
                }
            };
            if let Some(prev_end) = cursor {
                if span.start != prev_end {
                    return Err(PlanError::Invariant("layer spans not contiguous"));
                }
            } else if span.start != 0 {
                return Err(PlanError::Invariant("layer does not start at shard 0"));
            }
            cursor = Some(span.end);
            if is_reduce {
                reducers += 1;
            } else {
                carries += 1;
            }
        }
        if cursor != Some(c) {
            return Err(PlanError::Invariant("layer does not cover the core span"));
        }
        if reducers != shape.reducers || carries != shape.carries {
            return Err(PlanError::Invariant("layer shape differs from the rule"));
        }
        width = reducers + carries;
    }

    // L3 / L4.
    if width != plan.l3.children.len() as u32 || width == 0 || width > a {
        return Err(PlanError::Invariant("l3 arity out of range"));
    }
    if plan.l3.kind != NodeKind::L3 || plan.l4.kind != NodeKind::L4 {
        return Err(PlanError::Invariant("l3/l4 kinds wrong"));
    }
    check_node(&plan.l3)?;
    check_node(&plan.l4)?;
    if plan.l3.span != (ShardSpan { start: 0, end: c }) {
        return Err(PlanError::Invariant("l3 does not cover the core span"));
    }
    if plan.l4.children.len() != 1 || plan.l4.children[0].source != SourceNodeId::Node(plan.l3.id) {
        return Err(PlanError::Invariant("l4 must consume exactly the l3"));
    }
    if plan.l4.output.is_some() {
        return Err(PlanError::Invariant("l4 has an output binding"));
    }

    // Output bindings: every non-root node consumed exactly once, and the
    // binding is consistent with the consumer's child table.
    for node in plan.nodes() {
        if node.kind == NodeKind::L4 {
            continue;
        }
        let output = node.output.ok_or(PlanError::Invariant("missing output binding"))?;
        let parent = plan.node(output.parent).ok_or(PlanError::Invariant("dangling parent id"))?;
        let child = parent
            .children
            .get(output.parent_slot as usize)
            .ok_or(PlanError::Invariant("output slot out of range"))?;
        if child.source != SourceNodeId::Node(node.id) || child.span != node.span {
            return Err(PlanError::Invariant("output binding mismatch"));
        }
    }
    Ok(())
}

fn check_node(node: &NodePlan) -> Result<(), PlanError> {
    if node.expected_arity.get() as usize != node.children.len() {
        return Err(PlanError::Invariant("expected_arity != children.len()"));
    }
    let mut cursor = node.span.start;
    for (slot, child) in node.children.iter().enumerate() {
        if child.local_slot as usize != slot {
            return Err(PlanError::Invariant("child slots not dense"));
        }
        if child.span.start != cursor || child.span.is_empty() {
            return Err(PlanError::Invariant("child spans not contiguous"));
        }
        cursor = child.span.end;
        let expected_segment = segment_for(node.kind, child.child_class)?;
        if child.replay_segment != expected_segment {
            return Err(PlanError::Invariant("wrong replay segment"));
        }
    }
    if cursor != node.span.end {
        return Err(PlanError::Invariant("children do not cover the node span"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(c: u32) -> TreePlan {
        plan_min_nodes_even_v1(NonZeroU32::new(c).unwrap(), 11).unwrap()
    }

    fn band_plan(c: u32, cap: u8, workers: u8, bands: ArityBandTable) -> TreePlan {
        plan_band_aware_even_v1(
            NonZeroU32::new(c).unwrap(),
            cap,
            NonZeroU8::new(workers).unwrap(),
            bands,
        )
        .unwrap()
    }

    fn lift_sizes(plan: &TreePlan) -> Vec<u32> {
        plan.layers[0]
            .actions
            .iter()
            .map(|action| match action {
                NodeAction::Reduce(node) => node.span.len(),
                NodeAction::Carry(_) => unreachable!(),
            })
            .collect()
    }

    fn l2_layer_shapes(plan: &TreePlan) -> Vec<(Vec<u32>, u32)> {
        plan.layers
            .iter()
            .skip(1)
            .map(|layer| {
                let mut sizes = Vec::new();
                let mut carries = 0;
                for action in &layer.actions {
                    match action {
                        NodeAction::Reduce(node) => sizes.push(node.children.len() as u32),
                        NodeAction::Carry(_) => carries += 1,
                    }
                }
                (sizes, carries)
            })
            .collect()
    }

    /// V3 §6.4 frontier-level golden transitions.
    #[test]
    fn golden_l2_round_transitions() {
        let cases: &[(u32, u32, u32, &[u32])] = &[
            // (frontier, q, carries, reducer sizes)
            (12, 1, 1, &[11]),
            (91, 8, 3, &[11, 11, 11, 11, 11, 11, 11, 11]),
            (110, 10, 0, &[11; 10]),
            (111, 10, 1, &[11; 10]),
            (115, 11, 0, &[11, 11, 11, 11, 11, 10, 10, 10, 10, 10, 10]),
            (120, 11, 0, &[11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 10]),
            (121, 11, 0, &[11; 11]),
            (122, 11, 1, &[11; 11]),
            (123, 12, 0, &[11, 11, 11, 10, 10, 10, 10, 10, 10, 10, 10, 10]),
            (132, 12, 0, &[11; 12]),
        ];
        for &(c, q, carries, sizes) in cases {
            let shape = l2_round_shape(c, 11).unwrap();
            assert_eq!(shape.reducers, q, "frontier {c}: reducers");
            assert_eq!(shape.carries, carries, "frontier {c}: carries");
            assert_eq!(shape.sizes, sizes, "frontier {c}: sizes");
        }
    }

    /// The deliberate non-minimality at frontier 123 (V3 §6.3): two rounds,
    /// 13 proofs, where a legal same-depth tree exists with 12.
    #[test]
    fn golden_accepted_nonminimal_123() {
        let shape0 = l2_round_shape(123, 11).unwrap();
        assert_eq!((shape0.reducers, shape0.carries), (12, 0));
        assert_eq!(shape0.next_frontier(), 12);
        let shape1 = l2_round_shape(12, 11).unwrap();
        assert_eq!((shape1.reducers, shape1.carries), (1, 1));
        assert_eq!(shape0.reducers + shape1.reducers, 13);
    }

    /// V3 §6.4 core-level goldens.
    #[test]
    fn golden_core_plans() {
        let p = plan(1);
        assert_eq!(lift_sizes(&p), vec![1]);
        assert_eq!(p.l3.children.len(), 1);

        let p = plan(11);
        assert_eq!(lift_sizes(&p), vec![11]);
        assert_eq!(p.l3.children.len(), 1);

        let p = plan(12);
        assert_eq!(lift_sizes(&p), vec![6, 6]);
        assert_eq!(p.l3.children.len(), 2);

        let p = plan(13);
        assert_eq!(lift_sizes(&p), vec![7, 6]);
        assert_eq!(p.l3.children.len(), 2);

        let p = plan(121);
        assert_eq!(lift_sizes(&p), vec![11; 11]);
        assert_eq!(p.l3.children.len(), 11);
        assert_eq!(p.l2_proof_count(), 0);

        let p = plan(122);
        assert_eq!(lift_sizes(&p), {
            let mut v = vec![11, 11];
            v.extend([10; 10]);
            v
        });
        assert_eq!(l2_layer_shapes(&p), vec![(vec![11], 1)]);
        assert_eq!(p.l3.children.len(), 2);
        // Mixed L3: one L2 + one carried lift.
        assert_eq!(p.l3.children[0].child_class, ChildClass::L2);
        assert_eq!(p.l3.children[1].child_class, ChildClass::Lift);
        assert_eq!(p.l3.children[1].replay_segment, ReplaySegment::Base0);

        let p = plan(1_000);
        let mut expected = vec![11u32; 90];
        expected.push(10);
        assert_eq!(lift_sizes(&p), expected);
        assert_eq!(l2_layer_shapes(&p), vec![(vec![11; 8], 3)]);
        assert_eq!(p.l3.children.len(), 11);
        assert_eq!(p.l2_proof_count(), 8);
        let classes: Vec<_> = p.l3.children.iter().map(|c| c.child_class).collect();
        assert_eq!(&classes[..8], &[ChildClass::L2; 8]);
        assert_eq!(&classes[8..], &[ChildClass::Lift; 3]);

        let p = plan(1_332);
        assert_eq!(p.layers.len(), 3); // lifts + 2 L2 rounds
        assert_eq!(l2_layer_shapes(&p), vec![(vec![11; 11], 1), (vec![11], 1)]);
        assert_eq!(p.l3.children.len(), 2);
        assert_eq!(p.l2_proof_count(), 12);

        let p = plan(1_343);
        assert_eq!(
            l2_layer_shapes(&p),
            vec![(vec![11, 11, 11, 10, 10, 10, 10, 10, 10, 10, 10, 10], 0), (vec![11], 1)]
        );
        assert_eq!(p.l2_proof_count(), 13);
    }

    #[test]
    fn rejects_bad_arity_caps() {
        let c = NonZeroU32::new(19).unwrap();
        assert!(matches!(plan_min_nodes_even_v1(c, 1), Err(PlanError::ArityCapOutOfRange { .. })));
        assert!(matches!(
            plan_min_nodes_even_v1(c, (NATIVE_MAX_NODE_ARITY + 1) as u8),
            Err(PlanError::ArityCapOutOfRange { .. })
        ));
    }

    #[test]
    fn deterministic_across_runs() {
        for c in [1u32, 13, 121, 122, 1_000, 1_343, 65_535] {
            let a = plan(c);
            let b = plan(c);
            assert_eq!(a, b, "plan for {c} not deterministic");
        }
    }

    #[test]
    fn band_aware_rsp_19_uses_three_lifts_when_parent_band_is_preserved() {
        let parent_relaxed = ArityBandTable { lift: vec![9, 11], l2: vec![11], l3: vec![11] };
        let p = band_plan(19, 11, 3, parent_relaxed);
        assert_eq!(p.version, TREE_POLICY_BAND_AWARE_EVEN_V1);
        assert_eq!(lift_sizes(&p), vec![7, 6, 6]);
        assert_eq!(p.layers.len(), 1);
        assert_eq!(p.l3.children.len(), 3);
        validate(&p).unwrap();
    }

    #[test]
    fn current_rsp_table_records_lift_breakpoint_but_preserves_parent_guard() {
        let bands = ArityBandTable::current_native(11).unwrap();
        assert_eq!(bands.lift, vec![9, 11]);
        let p = band_plan(19, 11, 3, bands);
        assert_eq!(p, plan(19));
    }

    #[test]
    fn current_native_lift_breakpoint_respects_small_arity_caps() {
        for cap in 2..=9 {
            assert_eq!(ArityBandTable::current_native(cap).unwrap().lift, vec![cap]);
        }
        assert_eq!(ArityBandTable::current_native(10).unwrap().lift, vec![9, 10]);
        assert_eq!(ArityBandTable::current_native(11).unwrap().lift, vec![9, 11]);
    }

    #[test]
    fn band_aware_guard_falls_back_to_exact_v1() {
        for workers in 1..=8 {
            for c in [1u32, 9, 11, 19, 121, 122, 1_000] {
                let v1 = plan_min_nodes_even_v1(NonZeroU32::new(c).unwrap(), 11).unwrap();
                let guarded = band_plan(c, 11, workers, ArityBandTable::single_band(11).unwrap());
                assert_eq!(guarded, v1, "c={c} workers={workers}");
            }
        }
        // At W=2 the integer power-of-two model is tied (one high-band
        // wave versus two low-band waves), and the candidate also raises
        // L3 from k=2 to k=3. The guards deliberately keep V1.
        assert_eq!(band_plan(19, 11, 2, ArityBandTable::current_native(11).unwrap()), plan(19));
    }

    #[test]
    fn band_aware_selection_properties() {
        for cap in 2u8..=11 {
            let bands = ArityBandTable::current_native(cap).unwrap();
            for c in 1..=2_000u32 {
                let v1_count = ceil_div(c, u32::from(cap)).unwrap();
                let v1_rounds = l2_round_count(v1_count, u32::from(cap)).unwrap();
                for workers in 1..=8u32 {
                    let selected =
                        choose_band_aware_lift_count(c, u32::from(cap), workers, &bands).unwrap();
                    assert!(selected >= v1_count && selected <= c);
                    assert!(ceil_div(c, selected).unwrap() <= u32::from(cap));
                    assert_eq!(
                        l2_round_count(selected, u32::from(cap)).unwrap(),
                        v1_rounds,
                        "c={c} cap={cap} workers={workers}"
                    );
                }
            }
        }

        // Build and validate representative selected plans, including every
        // V1 boundary and both sides of the calibrated Lift cliff.
        for cap in 2u8..=11 {
            let bands = ArityBandTable::current_native(cap).unwrap();
            for c in [1u32, 8, 9, 10, 11, 12, 19, 20, 120, 121, 122, 999, 2_000] {
                for workers in 1..=8 {
                    let p = band_plan(c, cap, workers, bands.clone());
                    validate(&p).unwrap_or_else(|err| panic!("c={c} cap={cap} W={workers}: {err}"));
                    assert_eq!(p.layers[0].input_count, c);
                    assert!(lift_sizes(&p).iter().all(|&arity| arity <= u32::from(cap)));
                }
            }
        }

        // Exercise selected topologies over the same representative grid
        // with a synthetic, parent-relaxed Lift cliff. The production table
        // has a certified Lift breakpoint, but its L3 guard deliberately
        // remains stricter than this selection-focused fixture.
        for cap in 3u8..=11 {
            let bands = ArityBandTable { lift: vec![cap - 2, cap], l2: vec![cap], l3: vec![cap] };
            for c in [1u32, 8, 9, 10, 11, 12, 19, 20, 120, 121, 122, 999, 2_000] {
                for workers in 1..=8 {
                    let p = band_plan(c, cap, workers, bands.clone());
                    validate(&p).unwrap_or_else(|err| {
                        panic!("synthetic c={c} cap={cap} W={workers}: {err}")
                    });
                }
            }
        }
    }

    /// Route lookups: every core shard resolves into its lift; every
    /// produced node resolves into exactly its consumer; L4 has no route.
    #[test]
    fn routes_resolve() {
        for c in [1u32, 13, 122, 1_000, 1_332] {
            let p = plan(c);
            for shard in 0..c {
                let route = p.core_route(shard).expect("core route");
                assert_eq!(route.child_class, ChildClass::Core);
                assert!(route.span.start == shard && route.span.end == shard + 1);
            }
            assert!(p.core_route(c).is_none());
            for node in p.nodes() {
                let route = p.node_route(node.id);
                if node.kind == NodeKind::L4 {
                    assert!(route.is_none());
                } else {
                    let route = route.expect("node route");
                    assert_eq!(route.span, node.span);
                }
            }
        }
    }

    /// Exhaustive validation over dense low counts, all power-of-A
    /// boundaries, and a stride sample up to the shard-endpoint namespace,
    /// for the product cap and for small forced-depth caps.
    #[test]
    fn exhaustive_properties() {
        let check = |c: u32, cap: u8| {
            let p = plan_min_nodes_even_v1(NonZeroU32::new(c).unwrap(), cap)
                .unwrap_or_else(|e| panic!("plan({c}, {cap}): {e}"));
            validate(&p).unwrap_or_else(|e| panic!("validate({c}, {cap}): {e}"));
            // Final-round minimum-proof bound.
            if let Some(last) = p.layers.iter().skip(1).last() {
                let width = last.input_count;
                let a = cap as u32;
                let reducers = last
                    .actions
                    .iter()
                    .filter(|action| matches!(action, NodeAction::Reduce(_)))
                    .count() as u32;
                let next = p.l3.children.len() as u32;
                if next <= a {
                    let bound = ((width - a) + (a - 2)) / (a - 1);
                    assert_eq!(reducers, bound.max(1), "final round q at c={c} cap={cap}");
                }
            }
        };
        for c in 1..=3_000u32 {
            check(c, 11);
        }
        let mut c = 3_000u32;
        while c <= 66_000 {
            check(c, 11);
            check(c + 1, 11);
            c += 997; // prime stride
        }
        for &boundary in &[121u32, 122, 1_331, 1_332, 14_641, 14_642, 65_535, 65_536] {
            check(boundary, 11);
        }
        // Forced-depth small caps (the cheap deep-tree technique).
        for cap in 2u8..=5 {
            for c in 1..=600u32 {
                check(c, cap);
            }
        }
    }
}
