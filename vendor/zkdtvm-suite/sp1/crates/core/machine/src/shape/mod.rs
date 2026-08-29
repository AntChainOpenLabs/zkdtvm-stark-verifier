mod shapeable;

pub use shapeable::*;

use std::{collections::BTreeMap, marker::PhantomData, str::FromStr};

use dt_core_executor::{ExecutionRecord, Instruction, Opcode, Program, RiscvAirId};
use dt_stark::{
    air::MachineAir,
    shape::{OrderedShape, Shape, ShapeCluster},
};
use hashbrown::HashMap;
use itertools::Itertools;
use num::Integer;
use p3_baby_bear::BabyBear;
use p3_field::PrimeField32;
#[cfg(feature = "koalabear")]
use p3_field::TwoAdicField;
use p3_util::log2_ceil_usize;
use serde::Deserialize;
use thiserror::Error;

use crate::{
    bytes::ByteChip,
    global::GlobalChip,
    memory::MemoryLocalChip,
    program::ProgramChip,
    riscv::RiscvAir,
    syscall::{
        chip::SyscallChip,
        precompiles::{
            keccak_dt::KeccakControllerChip,
            sha256::{ShaCompressControllerChip, ShaExtendControllerChip},
        },
    },
};

#[cfg(feature = "koalabear")]
use dt_stark::koalabear_poseidon2::koala_bear_poseidon2::Challenge as KoalaBearChallenge;

#[cfg(feature = "koalabear")]
const CORE_COMMIT_LOG_BLOWUP: usize = 1;
/// Dynamically compute and set the preprocessed shape on a `Program` based on actual sizes.
///
/// Instead of matching against preset shapes, this computes the preprocessed chip heights from
/// the actual program:
/// - Byte chip: always 2^16 = 65536 rows
/// - Program chip: next power of 2 of instruction count (minimum 16)
pub fn fix_preprocessed_shape_dynamic(program: &mut Program) {
    if program.preprocessed_shape.is_some() {
        return;
    }
    let mut shape = Shape::default();
    // Byte chip has a fixed 2^16 trace.
    shape.insert(RiscvAirId::Byte, 16);
    // Program chip: pad instruction count to next power of 2, minimum 16.
    let program_log2 =
        std::cmp::max((program.instructions.len().max(1).next_power_of_two()).ilog2() as usize, 4);
    shape.insert(RiscvAirId::Program, program_log2);
    program.preprocessed_shape = Some(shape);
}

/// Dynamically compute and set the shape of an `ExecutionRecord` based on actual event counts.
///
/// For each chip, the log2 height = ceil(log2(next_power_of_two(event_count))), minimum 4
/// (so at least 16 rows). Chips with 0 events are omitted from the shape, which means
/// `included()` will return false for them.
pub fn fix_shape_dynamic(record: &mut ExecutionRecord) {
    if record.shape.is_some() {
        return;
    }

    let mut shape = Shape::default();

    // 1. Start with preprocessed chip heights from the program.
    if let Some(prep_shape) = &record.program.preprocessed_shape {
        shape.extend(std::iter::once(prep_shape.clone()));
    } else {
        // Fallback: compute preprocessed heights if program shape was not set.
        // Byte chip: fixed 2^16 rows.
        shape.insert(RiscvAirId::Byte, 16);
        // Program chip: next power of 2 of instruction count, min 16.
        let program_log2 = std::cmp::max(
            (record.program.instructions.len().max(1).next_power_of_two()).ilog2() as usize,
            4,
        );
        shape.insert(RiscvAirId::Program, program_log2);
    }

    // 2. Core chips: compute from actual event counts (must match each chip's trace / num_rows).
    for (air_id, count) in record.core_heights() {
        if count > 0 {
            let log2_height = std::cmp::max((count.next_power_of_two()).ilog2() as usize, 4);
            shape.insert(air_id, log2_height);
        }
    }

    // 3. Memory chips (for memory init/finalize shards).
    for (air_id, count) in record.memory_heights() {
        if count > 0 {
            let log2_height = std::cmp::max((count.next_power_of_two()).ilog2() as usize, 4);
            // Only insert if not already present (core_heights may have set Global).
            if !shape.contains(&air_id) {
                shape.insert(air_id, log2_height);
            } else {
                // For Global chip, take the max of core and memory heights.
                let existing = shape.log2_height(&air_id).unwrap();
                shape.insert(air_id, std::cmp::max(existing, log2_height));
            }
        }
    }

    record.shape = Some(shape);
}

/// The set of maximal shapes.
///
/// These shapes define the "worst-case" shapes for typical shards that are proving `rv32im`
/// execution. We use a variant of a cartesian product of the allowed log heights to generate
/// smaller shapes from these ones.
const MAXIMAL_SHAPES: &[u8] = include_bytes!("maximal_shapes.json");

/// Raw shape with string keys for JSON that may still contain old RiscvAirId names (Cpu, AddSub,
/// MemoryInstrs, Jump).
#[derive(Debug, Deserialize)]
struct RawMaximalShape {
    inner: HashMap<String, usize>,
}

/// Converts a raw shape from JSON (possibly with old keys) into Shape<RiscvAirId>.
fn migrate_maximal_shape(raw: RawMaximalShape) -> Shape<RiscvAirId> {
    let mut inner = HashMap::<RiscvAirId, usize>::new();
    for (k, v) in raw.inner {
        match k.as_str() {
            "Cpu" => { /* removed: no Cpu chip */ }
            "AddSub" => {
                inner.insert(RiscvAirId::Add, v);
                inner.insert(RiscvAirId::Addi, v);
                inner.insert(RiscvAirId::Sub, v);
            }
            "MemoryInstrs" => {
                inner.insert(RiscvAirId::LoadByte, v);
                inner.insert(RiscvAirId::LoadHalf, v);
                inner.insert(RiscvAirId::LoadWord, v);
                inner.insert(RiscvAirId::StoreByte, v);
                inner.insert(RiscvAirId::StoreHalf, v);
                inner.insert(RiscvAirId::StoreWord, v);
            }
            "Jump" => {
                inner.insert(RiscvAirId::Jal, v);
                inner.insert(RiscvAirId::Jalr, v);
            }
            _ => {
                if let Ok(id) = RiscvAirId::from_str(&k) {
                    inner.insert(id, v);
                }
            }
        }
    }
    Shape { inner }
}

/// The set of tiny shapes.
///
/// These shapes are used to optimize performance for smaller programs.
const SMALL_SHAPES: &[u8] = include_bytes!("small_shapes.json");

/// Sumcheck skip-rounds parameters, read from the WHIR JSON config at runtime.
/// Defaults: NUM_SKIP_ROUNDS=4, CHIP_LOG_HEIGHT_THRESHOLD=12 (the historic values).
/// PolyAir configs typically set these to 1/0 (all-linear rounds).
fn load_whir_json_config() -> &'static dt_stark::koalabear_poseidon2::WhirJsonConfig {
    #[cfg(feature = "koalabear")]
    {
        dt_stark::koalabear_poseidon2::whir_config()
    }
    #[cfg(feature = "babybear")]
    {
        dt_stark::babybear_config()
    }
}

pub fn num_skip_rounds() -> usize {
    load_whir_json_config().num_skip_rounds()
}

pub fn chip_log_height_threshold() -> usize {
    load_whir_json_config().chip_log_height_threshold()
}

pub fn chip_height_threshold() -> usize {
    1usize << chip_log_height_threshold()
}

/// A configuration for what shapes are allowed to be used by the prover.
#[derive(Debug)]
pub struct CoreShapeConfig<F: PrimeField32> {
    partial_preprocessed_shapes: ShapeCluster<RiscvAirId>,
    partial_core_shapes: BTreeMap<usize, Vec<ShapeCluster<RiscvAirId>>>,
    partial_memory_shapes: ShapeCluster<RiscvAirId>,
    partial_precompile_shapes: HashMap<RiscvAirId, (usize, Vec<usize>)>,
    partial_small_shapes: Vec<ShapeCluster<RiscvAirId>>,
    costs: HashMap<RiscvAirId, usize>,
    _data: PhantomData<F>,
}

impl<F: PrimeField32> CoreShapeConfig<F> {
    /// Fix the preprocessed shape of the proof.
    pub fn fix_preprocessed_shape(&self, program: &mut Program) -> Result<(), CoreShapeError> {
        // If the preprocessed shape is already fixed, return an error.
        if program.preprocessed_shape.is_some() {
            return Err(CoreShapeError::PreprocessedShapeAlreadyFixed);
        }

        // Get the heights of the preprocessed chips and find a shape that fits.
        let preprocessed_heights = RiscvAir::<F>::preprocessed_heights(program);
        let preprocessed_shape = self
            .partial_preprocessed_shapes
            .find_shape(&preprocessed_heights)
            .ok_or(CoreShapeError::PreprocessedShapeError)?;

        // Set the preprocessed shape.
        program.preprocessed_shape = Some(preprocessed_shape);

        Ok(())
    }

    /// Fix the shape of the proof.
    pub fn fix_shape(&self, record: &mut ExecutionRecord) -> Result<(), CoreShapeError> {
        if record.program.preprocessed_shape.is_none() {
            return Err(CoreShapeError::PreprocessedShapeMissing);
        }
        if record.shape.is_some() {
            return Err(CoreShapeError::ShapeAlreadyFixed);
        }

        // Set the shape of the chips with prepcoded shapes to match the preprocessed shape from the
        // program.
        record.shape.clone_from(&record.program.preprocessed_shape);

        match self.find_shape(record) {
            Ok(shape) => {
                record.shape.as_mut().unwrap().extend(shape);
                self.ensure_shape_within_active_capacity(record.shape.as_ref().unwrap())?;
                Ok(())
            }
            Err(e) => {
                if matches!(e, CoreShapeError::ShapeCapacityExceeded { .. }) {
                    return Err(e);
                }
                tracing::debug!(
                    "Shard {} fixed shape fallback to dynamic: {:?}",
                    record.public_values.shard,
                    e
                );
                record.shape = None;
                fix_shape_dynamic(record);
                self.ensure_shape_within_active_capacity(record.shape.as_ref().unwrap())?;
                Ok(())
            }
        }
    }

    /// TODO move this into the executor crate
    pub fn find_shape<R: Shapeable>(
        &self,
        record: &R,
    ) -> Result<Shape<RiscvAirId>, CoreShapeError> {
        match record.kind() {
            // If this is a packed "core" record where the cpu events are alongisde the memory init
            // and finalize events, try to fix the shape using the tiny shapes.
            ShardKind::PackedCore => {
                // Get the heights of the core airs in the record.
                let mut heights = record.core_heights();
                heights.extend(record.memory_heights());

                let (cluster_index, shape, _) = self
                    .minimal_cluster_shape(self.partial_small_shapes.iter().enumerate(), &heights)
                    .ok_or_else(|| {
                        // No shape found, so return an error.
                        CoreShapeError::ShapeError(
                            heights
                                .iter()
                                .map(|(air, height)| (air.to_string(), log2_ceil_usize(*height)))
                                .collect(),
                        )
                    })?;

                let shard = record.shard();
                tracing::debug!("Shard Lifted: Index={}, Cluster={}", shard, cluster_index);
                for (air, height) in heights.iter() {
                    if shape.contains(air) {
                        tracing::debug!(
                            "Chip {:<20}: {:<3} -> {:<3}",
                            air,
                            log2_ceil_usize(*height),
                            shape.log2_height(air).unwrap(),
                        );
                    }
                }
                self.ensure_shape_within_active_capacity(&shape)?;
                Ok(shape)
            }
            ShardKind::Core => {
                // If this is a normal "core" record, try to fix the shape as such.

                // Get the heights of the core airs in the record.
                let heights = record.core_heights();

                // Try to find the smallest shape fitting within at least one of the candidate
                // shapes.
                let log2_shard_size = record.log2_shard_size();

                let (cluster_index, shape, _) = self
                    .minimal_cluster_shape(
                        self.partial_core_shapes
                            .range(log2_shard_size..)
                            .flat_map(|(_, clusters)| clusters.iter().enumerate()),
                        &heights,
                    )
                    // No shape found, so return an error.
                    .ok_or_else(|| CoreShapeError::ShapeError(record.debug_stats()))?;

                let shard = record.shard();
                tracing::debug!("Shard Lifted: Index={}, Cluster={}", shard, cluster_index);

                for (air, height) in heights.iter() {
                    if shape.contains(air) {
                        tracing::debug!(
                            "Chip {:<20}: {:<3} -> {:<3}",
                            air,
                            log2_ceil_usize(*height),
                            shape.log2_height(air).unwrap(),
                        );
                    }
                }
                self.ensure_shape_within_active_capacity(&shape)?;
                Ok(shape)
            }
            ShardKind::GlobalMemory => {
                // If the record is a does not have the CPU chip and is a global memory
                // init/finalize record, try to fix the shape as such.
                let heights = record.memory_heights();
                let shape = self
                    .partial_memory_shapes
                    .find_shape(&heights)
                    .ok_or(CoreShapeError::ShapeError(record.debug_stats()))?;
                self.ensure_shape_within_active_capacity(&shape)?;
                Ok(shape)
            }
            ShardKind::Precompile => {
                // Try to fix the shape as a precompile record.
                for (&air, (memory_events_per_row, allowed_log2_heights)) in
                    self.partial_precompile_shapes.iter()
                {
                    // Filter to check that the shard and shape air match.
                    let Some((height, num_memory_local_events, num_global_events)) =
                        record.precompile_heights().find_map(|x| (x.0 == air).then_some(x.1))
                    else {
                        continue;
                    };
                    for allowed_log2_height in allowed_log2_heights {
                        let allowed_height = 1 << allowed_log2_height;
                        if height <= allowed_height {
                            for shape in self.get_precompile_shapes(
                                air,
                                *memory_events_per_row,
                                *allowed_log2_height,
                            ) {
                                let mem_events_height = shape[2].1;
                                let global_events_height = shape[3].1;
                                if num_memory_local_events <= (1 << mem_events_height) &&
                                    num_global_events <= (1 << global_events_height)
                                {
                                    let mut actual_shape: Shape<RiscvAirId> = Shape::default();
                                    actual_shape.extend(
                                        shape
                                            .iter()
                                            .map(|x| (RiscvAirId::from_str(&x.0).unwrap(), x.1)),
                                    );
                                    self.ensure_shape_within_active_capacity(&actual_shape)?;
                                    return Ok(actual_shape);
                                }
                            }
                        }
                    }
                    tracing::error!(
                        "Cannot find shape for precompile {:?}, height {:?}, and mem events {:?}",
                        air,
                        height,
                        num_memory_local_events
                    );
                    return Err(CoreShapeError::ShapeError(record.debug_stats()));
                }
                Err(CoreShapeError::PrecompileNotIncluded(record.debug_stats()))
            }
        }
    }

    fn active_max_committed_log_height() -> Option<usize> {
        #[cfg(feature = "koalabear")]
        {
            Some(KoalaBearChallenge::TWO_ADICITY - CORE_COMMIT_LOG_BLOWUP)
        }
        #[cfg(feature = "babybear")]
        {
            None
        }
    }

    fn shape_capacity_violation(shape: &Shape<RiscvAirId>) -> Option<(String, usize, usize)> {
        let max_log_height = Self::active_max_committed_log_height()?;
        shape
            .iter()
            .find(|(_, log_height)| **log_height > max_log_height)
            .map(|(air, log_height)| (air.to_string(), *log_height, max_log_height))
    }

    fn ordered_shape_within_active_capacity(shape: &OrderedShape) -> bool {
        let Some(max_log_height) = Self::active_max_committed_log_height() else {
            return true;
        };
        shape.inner.iter().all(|(_, log_height)| *log_height <= max_log_height)
    }

    fn shape_within_active_capacity(shape: &Shape<RiscvAirId>) -> bool {
        Self::shape_capacity_violation(shape).is_none()
    }

    pub fn ensure_shape_within_active_capacity(
        &self,
        shape: &Shape<RiscvAirId>,
    ) -> Result<(), CoreShapeError> {
        if let Some((air, log_height, max_log_height)) = Self::shape_capacity_violation(shape) {
            return Err(CoreShapeError::ShapeCapacityExceeded { air, log_height, max_log_height });
        }
        Ok(())
    }

    /// Returns the area, cluster index, and shape of the minimal shape from candidates that fit a
    /// given collection of heights.
    pub fn minimal_cluster_shape<'a, N, I>(
        &self,
        indexed_shape_clusters: I,
        heights: &[(RiscvAirId, usize)],
    ) -> Option<(N, Shape<RiscvAirId>, usize)>
    where
        I: IntoIterator<Item = (N, &'a ShapeCluster<RiscvAirId>)>,
    {
        // Try to find a shape fitting within at least one of the candidate shapes.
        indexed_shape_clusters
            .into_iter()
            .filter_map(|(i, cluster)| {
                let shape = cluster.find_shape(heights)?;
                let area = self.estimate_lde_size(&shape);
                Some((i, shape, area))
            })
            .min_by_key(|x| x.2) // Find minimum by area.
    }

    // TODO: this function is atrocious, fix this
    fn get_precompile_shapes(
        &self,
        air_id: RiscvAirId,
        memory_events_per_row: usize,
        allowed_log2_height: usize,
    ) -> Vec<[(String, usize); 4]> {
        // TODO: This is a temporary fix to the shape, concretely fix this
        (1..=4 * air_id.rows_per_event())
            .rev()
            .map(|rows_per_event| {
                let num_local_mem_events =
                    ((1 << allowed_log2_height) * memory_events_per_row).div_ceil(rows_per_event);
                [
                    (air_id.to_string(), allowed_log2_height),
                    (
                        match air_id {
                            RiscvAirId::ShaExtend => {
                                RiscvAir::<F>::ShaExtendController(ShaExtendControllerChip::new())
                            }
                            RiscvAirId::ShaCompress => RiscvAir::<F>::ShaCompressController(
                                ShaCompressControllerChip::new(),
                            ),
                            RiscvAirId::KeccakPermute => {
                                RiscvAir::<F>::KeccakController(KeccakControllerChip::new())
                            }
                            _ => RiscvAir::<F>::SyscallPrecompile(SyscallChip::precompile()),
                        }
                        .name(),
                        ((1 << allowed_log2_height)
                            .div_ceil(&air_id.rows_per_event())
                            .next_power_of_two()
                            .ilog2() as usize)
                            .max(4),
                    ),
                    (
                        RiscvAir::<F>::MemoryLocal(MemoryLocalChip::new()).name(),
                        (num_local_mem_events.next_power_of_two().ilog2() as usize).max(4),
                    ),
                    (
                        RiscvAir::<F>::Global(GlobalChip).name(),
                        ((2 * num_local_mem_events +
                            (1 << allowed_log2_height).div_ceil(&air_id.rows_per_event()))
                        .next_power_of_two()
                        .ilog2() as usize)
                            .max(4),
                    ),
                ]
            })
            .filter(|shape| shape[3].1 <= 22)
            .collect::<Vec<_>>()
    }

    fn generate_all_shapes_from_allowed_log_heights(
        allowed_log_heights: impl IntoIterator<Item = (String, Vec<Option<usize>>)>,
    ) -> impl Iterator<Item = OrderedShape> {
        allowed_log_heights
            .into_iter()
            .map(|(name, heights)| heights.into_iter().map(move |height| (name.clone(), height)))
            .multi_cartesian_product()
            .map(|iter| {
                iter.into_iter()
                    .filter_map(|(name, maybe_height)| {
                        maybe_height.map(|log_height| (name, log_height))
                    })
                    .collect::<OrderedShape>()
            })
    }

    pub fn all_shapes(&self) -> impl Iterator<Item = OrderedShape> + '_ {
        let preprocessed_heights = self
            .partial_preprocessed_shapes
            .iter()
            .map(|(air, heights)| (air.to_string(), heights.clone()))
            .collect::<HashMap<_, _>>();

        let mut memory_heights = self
            .partial_memory_shapes
            .iter()
            .map(|(air, heights)| (air.to_string(), heights.clone()))
            .collect::<HashMap<_, _>>();
        memory_heights.extend(preprocessed_heights.clone());

        let precompile_only_shapes = self.partial_precompile_shapes.iter().flat_map(
            move |(&air, (mem_events_per_row, allowed_log_heights))| {
                allowed_log_heights.iter().flat_map(move |allowed_log_height| {
                    self.get_precompile_shapes(air, *mem_events_per_row, *allowed_log_height)
                })
            },
        );

        let precompile_shapes =
            Self::generate_all_shapes_from_allowed_log_heights(preprocessed_heights.clone())
                .flat_map(move |preprocessed_shape| {
                    precompile_only_shapes.clone().map(move |precompile_shape| {
                        preprocessed_shape
                            .clone()
                            .into_iter()
                            .chain(precompile_shape)
                            .collect::<OrderedShape>()
                    })
                });

        self.partial_core_shapes
            .values()
            .flatten()
            .chain(self.partial_small_shapes.iter())
            .flat_map(move |allowed_log_heights| {
                Self::generate_all_shapes_from_allowed_log_heights({
                    let mut log_heights = allowed_log_heights
                        .iter()
                        .map(|(air, heights)| (air.to_string(), heights.clone()))
                        .collect::<HashMap<_, _>>();
                    log_heights.extend(preprocessed_heights.clone());
                    log_heights
                })
            })
            .chain(Self::generate_all_shapes_from_allowed_log_heights(memory_heights))
            .chain(precompile_shapes)
            .filter(Self::ordered_shape_within_active_capacity)
    }

    pub fn maximal_core_shapes(&self, max_log_shard_size: usize) -> Vec<Shape<RiscvAirId>> {
        let min_key = *self.partial_core_shapes.keys().min().unwrap();
        let max_key = *self.partial_core_shapes.keys().max().unwrap();
        let log_shard_size = max_log_shard_size.max(min_key).min(max_key);
        let max_preprocessed = self
            .partial_preprocessed_shapes
            .iter()
            .map(|(air, allowed_heights)| {
                (air.to_string(), allowed_heights.last().unwrap().unwrap())
            })
            .collect::<HashMap<_, _>>();

        let max_core_shapes =
            self.partial_core_shapes[&log_shard_size].iter().map(|allowed_log_heights| {
                max_preprocessed
                    .clone()
                    .into_iter()
                    .chain(allowed_log_heights.iter().flat_map(|(air, allowed_heights)| {
                        allowed_heights
                            .last()
                            .unwrap()
                            .map(|log_height| (air.to_string(), log_height))
                    }))
                    .map(|(air, log_height)| (RiscvAirId::from_str(&air).unwrap(), log_height))
                    .collect::<Shape<RiscvAirId>>()
            });

        max_core_shapes.filter(Self::shape_within_active_capacity).collect()
    }

    pub fn maximal_core_plus_precompile_shapes(
        &self,
        max_log_shard_size: usize,
    ) -> Vec<Shape<RiscvAirId>> {
        let max_preprocessed = self
            .partial_preprocessed_shapes
            .iter()
            .map(|(air, allowed_heights)| {
                (air.to_string(), allowed_heights.last().unwrap().unwrap())
            })
            .collect::<HashMap<_, _>>();

        let precompile_only_shapes = self.partial_precompile_shapes.iter().flat_map(
            move |(&air, (mem_events_per_row, allowed_log_heights))| {
                self.get_precompile_shapes(
                    air,
                    *mem_events_per_row,
                    *allowed_log_heights.last().unwrap(),
                )
            },
        );

        let precompile_shapes: Vec<Shape<RiscvAirId>> = precompile_only_shapes
            .map(|x| {
                max_preprocessed
                    .clone()
                    .into_iter()
                    .chain(x)
                    .map(|(air, log_height)| (RiscvAirId::from_str(&air).unwrap(), log_height))
                    .collect::<Shape<RiscvAirId>>()
            })
            .filter(|shape| shape.log2_height(&RiscvAirId::Global).unwrap() < 21)
            .collect();

        self.maximal_core_shapes(max_log_shard_size)
            .into_iter()
            .chain(precompile_shapes)
            .filter(Self::shape_within_active_capacity)
            .collect()
    }

    pub fn estimate_lde_size(&self, shape: &Shape<RiscvAirId>) -> usize {
        shape.iter().map(|(air, height)| self.costs[air] * (1 << height)).sum()
    }

    // TODO: cleanup..
    pub fn small_program_shapes(&self) -> Vec<OrderedShape> {
        self.partial_small_shapes
            .iter()
            .map(|log_heights| {
                OrderedShape::from_log2_heights(
                    &log_heights
                        .iter()
                        .filter(|(_, v)| v[0].is_some())
                        .map(|(k, v)| (k.to_string(), v.last().unwrap().unwrap()))
                        .chain(vec![
                            (MachineAir::<BabyBear>::name(&ProgramChip), 19),
                            (MachineAir::<BabyBear>::name(&ByteChip::default()), 16),
                        ])
                        .collect::<Vec<_>>(),
                )
            })
            .filter(Self::ordered_shape_within_active_capacity)
            .collect()
    }
}

impl<F: PrimeField32> Default for CoreShapeConfig<F> {
    fn default() -> Self {
        // Load the maximal shapes (JSON may still have old keys Cpu/AddSub/MemoryInstrs/Jump).
        let raw_maximal: BTreeMap<usize, Vec<RawMaximalShape>> =
            serde_json::from_slice(MAXIMAL_SHAPES).unwrap();
        let maximal_shapes: BTreeMap<usize, Vec<Shape<RiscvAirId>>> = raw_maximal
            .into_iter()
            .map(|(k, v)| (k, v.into_iter().map(migrate_maximal_shape).collect()))
            .collect();
        let raw_small: Vec<RawMaximalShape> = serde_json::from_slice(SMALL_SHAPES).unwrap();
        let small_shapes: Vec<Shape<RiscvAirId>> =
            raw_small.into_iter().map(migrate_maximal_shape).collect();

        // Set the allowed preprocessed log2 heights.
        let allowed_preprocessed_log2_heights = HashMap::from([
            (RiscvAirId::Program, vec![Some(21), Some(22)]),
            (RiscvAirId::Byte, vec![Some(16)]),
        ]);

        // Generate the clusters from the maximal shapes and register them indexed by max chip
        // log height. In v6_final there is no CPU chip; the single key is 22
        // (SHARD_HEIGHT_THRESHOLD = 1 << 22).
        let mut core_allowed_log2_heights = BTreeMap::new();
        for (log2_shard_size, maximal_shapes) in maximal_shapes {
            let mut clusters = vec![];

            for maximal_shape in maximal_shapes.iter() {
                let cluster = derive_cluster_from_maximal_shape(maximal_shape);
                clusters.push(cluster);
            }

            core_allowed_log2_heights.insert(log2_shard_size, clusters);
        }

        // Set the memory init and finalize heights.
        // Heights extended to cover RSP production data: Global up to 24,
        // MemInit/Finalize up to 23.
        let memory_allowed_log2_heights = HashMap::from(
            [
                (
                    RiscvAirId::MemoryGlobalInit,
                    vec![
                        None,
                        Some(10),
                        Some(16),
                        Some(18),
                        Some(19),
                        Some(20),
                        Some(21),
                        Some(22),
                        Some(23),
                    ],
                ),
                (
                    RiscvAirId::MemoryGlobalFinalize,
                    vec![
                        None,
                        Some(10),
                        Some(16),
                        Some(18),
                        Some(19),
                        Some(20),
                        Some(21),
                        Some(22),
                        Some(23),
                    ],
                ),
                (
                    RiscvAirId::Global,
                    vec![
                        None,
                        Some(11),
                        Some(17),
                        Some(19),
                        Some(21),
                        Some(22),
                        Some(23),
                        Some(24),
                    ],
                ),
                (
                    RiscvAirId::GlobalTileReducer,
                    vec![
                        None,
                        Some(4),
                        Some(5),
                        Some(6),
                        Some(7),
                        Some(8),
                        Some(9),
                        Some(10),
                        Some(11),
                        Some(12),
                        Some(13),
                    ],
                ),
            ]
            .map(|(air, log_heights)| (air, log_heights)),
        );

        let mut precompile_allowed_log2_heights = HashMap::new();
        let precompile_heights = (3..21).collect::<Vec<_>>();
        for (air, memory_events_per_row) in
            RiscvAir::<F>::precompile_airs_with_memory_events_per_row()
        {
            precompile_allowed_log2_heights
                .insert(air, (memory_events_per_row, precompile_heights.clone()));
        }

        Self {
            partial_preprocessed_shapes: ShapeCluster::new(allowed_preprocessed_log2_heights),
            partial_core_shapes: core_allowed_log2_heights,
            partial_memory_shapes: ShapeCluster::new(memory_allowed_log2_heights),
            partial_precompile_shapes: precompile_allowed_log2_heights,
            partial_small_shapes: small_shapes
                .into_iter()
                .map(|x| {
                    ShapeCluster::new(x.into_iter().map(|(k, v)| (k, vec![Some(v)])).collect())
                })
                .collect(),
            costs: serde_json::from_str(include_str!("rv32im_costs.json"))
                .expect("Failed to load rv32im_costs.json file. Verify that `git config core.symlinks` is not set to false."),
            _data: PhantomData,
        }
    }
}

fn derive_cluster_from_maximal_shape(shape: &Shape<RiscvAirId>) -> ShapeCluster<RiscvAirId> {
    let range_down = |h: usize, levels: usize| -> Vec<Option<usize>> {
        let min_h = h.saturating_sub(levels);
        (min_h..=h).map(Some).collect()
    };

    let optional_with_range =
        |h: Option<usize>, levels: usize, buffer: usize| -> Vec<Option<usize>> {
            match h {
                Some(h) => {
                    let mut opts = vec![None];
                    let min_h = h.saturating_sub(levels);
                    opts.extend((min_h..=h).map(Some));
                    opts
                }
                None => vec![None, Some(buffer)],
            }
        };

    let optional_fixed = |h: Option<usize>, buffer: usize| -> Vec<Option<usize>> {
        match h {
            Some(h) => vec![None, Some(h)],
            None => vec![None, Some(buffer)],
        }
    };

    let fixed = |h: usize| -> Vec<Option<usize>> { vec![Some(h)] };

    let mut m = HashMap::new();

    // Global228 and its fixed TileReducer83 keep the narrow high-cost shape neighborhoods.
    m.insert(
        RiscvAirId::Global,
        optional_with_range(shape.log2_height(&RiscvAirId::Global), 1, 10),
    );
    m.insert(
        RiscvAirId::GlobalTileReducer,
        optional_with_range(shape.log2_height(&RiscvAirId::GlobalTileReducer), 4, 18),
    );

    // High-frequency / high-cost chips: 2 options {h-1, h}
    for &chip in
        &[RiscvAirId::Add, RiscvAirId::Addi, RiscvAirId::Sub, RiscvAirId::Lt, RiscvAirId::LoadWord]
    {
        let h = shape.log2_height(&chip).unwrap_or(4);
        m.insert(chip, range_down(h, 1));
    }

    // Medium chips: 1 option (fixed at maximal height)
    for &chip in &[
        RiscvAirId::Branch,
        RiscvAirId::Bitwise,
        RiscvAirId::StoreWord,
        RiscvAirId::MemoryLocal,
        RiscvAirId::Auipc,
        RiscvAirId::Jal,
        RiscvAirId::Jalr,
        RiscvAirId::LoadByte,
        RiscvAirId::LoadHalf,
        RiscvAirId::StoreByte,
        RiscvAirId::StoreHalf,
        RiscvAirId::ShiftRight,
        RiscvAirId::ShiftLeft,
    ] {
        let h = shape.log2_height(&chip).unwrap_or(4);
        m.insert(chip, fixed(h));
    }

    // Often-absent chips: 2 options {None, h}
    for &chip in
        &[RiscvAirId::DivRem, RiscvAirId::Mul, RiscvAirId::SyscallCore, RiscvAirId::SyscallInstrs]
    {
        m.insert(chip, optional_fixed(shape.log2_height(&chip), 10));
    }

    // Only the 23 chips returned by core_heights() belong in core clusters.
    // Precompile chips (KeccakPermute, Secp256k1*, etc.) and memory global
    // chips (MemoryGlobalInit/Finalize) in maximal_shapes are artifacts of
    // data collection and handled by their own dedicated shape configs.

    ShapeCluster::new(m)
}

#[derive(Debug, Error)]
pub enum CoreShapeError {
    #[error("no preprocessed shape found")]
    PreprocessedShapeError,
    #[error("Preprocessed shape already fixed")]
    PreprocessedShapeAlreadyFixed,
    #[error("no shape found {0:?}")]
    ShapeError(HashMap<String, usize>),
    #[error("Preprocessed shape missing")]
    PreprocessedShapeMissing,
    #[error("Shape already fixed")]
    ShapeAlreadyFixed,
    #[error("Precompile not included in allowed shapes {0:?}")]
    PrecompileNotIncluded(HashMap<String, usize>),
    #[error(
        "shape exceeds active PCS capacity: {air} log_height {log_height} > max {max_log_height}"
    )]
    ShapeCapacityExceeded { air: String, log_height: usize, max_log_height: usize },
}

pub fn create_dummy_program(shape: &Shape<RiscvAirId>) -> Program {
    let mut program =
        Program::new(vec![Instruction::new(Opcode::ADD, 30, 0, 0, false, false)], 1 << 5, 1 << 5);
    program.preprocessed_shape = Some(shape.clone());
    program
}

pub fn create_dummy_record(shape: &Shape<RiscvAirId>) -> ExecutionRecord {
    let program = std::sync::Arc::new(create_dummy_program(shape));
    let mut record = ExecutionRecord::new(program);
    record.shape = Some(shape.clone());
    record
}

#[cfg(test)]
pub mod tests {
    #![allow(clippy::print_stdout)]

    use hashbrown::HashSet;

    use super::*;

    #[test]
    #[ignore]
    fn test_making_shapes() {
        use p3_koala_bear::KoalaBear;
        let shape_config = CoreShapeConfig::<KoalaBear>::default();
        let num_shapes = shape_config.all_shapes().collect::<HashSet<_>>().len();
        println!("There are {} core shapes (all_shapes)", num_shapes);
        assert!(num_shapes < 1 << 24);
    }

    #[cfg(feature = "koalabear")]
    #[test]
    fn test_quintic_capacity_rejects_global_log_height_24() {
        use p3_koala_bear::KoalaBear;

        let shape_config = CoreShapeConfig::<KoalaBear>::default();
        let mut shape = Shape::default();
        shape.insert(RiscvAirId::Global, 24);

        let err = shape_config
            .ensure_shape_within_active_capacity(&shape)
            .expect_err("quintic path must reject log_height 24 with log_blowup 1");
        assert!(matches!(
            err,
            CoreShapeError::ShapeCapacityExceeded {
                air,
                log_height: 24,
                max_log_height: 23,
            } if air == "Global"
        ));
    }

    #[cfg(feature = "koalabear")]
    #[test]
    fn test_quintic_ordered_shape_capacity_filter() {
        use p3_koala_bear::KoalaBear;

        let ok = OrderedShape::from_log2_heights(&[("Global".to_string(), 23)]);
        let over = OrderedShape::from_log2_heights(&[("Global".to_string(), 24)]);

        assert!(CoreShapeConfig::<KoalaBear>::ordered_shape_within_active_capacity(&ok));
        assert!(!CoreShapeConfig::<KoalaBear>::ordered_shape_within_active_capacity(&over));
    }
}
