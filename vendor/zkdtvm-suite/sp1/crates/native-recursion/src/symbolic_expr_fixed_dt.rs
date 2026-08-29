use std::collections::HashSet;

use dt_stark::air::{InteractionScope, MachineAir, PairCol};
use polyair::{
    symbolic::{SymbolicAirBuilder, SymbolicExpression, SymbolicVar},
    Chip,
};
use serde::{Deserialize, Serialize};

use crate::{
    child_views::NativeChildRole,
    config::{DIGEST_SIZE, D_EF, F},
    symbolic_expr_adapter_dt::{
        encode_f, fuse_mul_add_nodes, put_u32, put_u8, put_usize, RecursionAdaptedRootStreams,
        RecursionAdapterError, RecursionSymbolicExprAdapter, RecursionSymbolicExpression,
    },
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecursionChildRole {
    Core,
    Compress,
    Shrink,
}

impl From<NativeChildRole> for RecursionChildRole {
    fn from(value: NativeChildRole) -> Self {
        match value {
            NativeChildRole::Core => Self::Core,
            NativeChildRole::Compress => Self::Compress,
            NativeChildRole::Shrink => Self::Shrink,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RecursionFixedSymbolicProgram {
    pub version: u32,
    pub role: RecursionChildRole,
    pub chips: Vec<RecursionFixedSymbolicChip>,
    pub max_required_beta_power: usize,
    pub canonical_bytes_len: usize,
    pub artifact_digest: [F; DIGEST_SIZE],
}

#[derive(Debug, Clone)]
pub struct RecursionFixedSymbolicChip {
    pub static_chip_id: usize,
    pub chip_name: String,
    pub main_width: usize,
    pub preprocessed_width: usize,
    pub public_width: usize,
    pub commit_scope: InteractionScope,
    pub reserved_poly: Vec<PairCol>,
    pub logup_batch_size: usize,
    pub num_gate_roots: usize,
    pub num_constraints_from_builder: usize,
    pub builder_snapshot: RecursionSymbolicBuilderSnapshot,
    pub adapted_roots: RecursionAdaptedRootStreams,
}

#[derive(Debug, Clone)]
pub struct RecursionSymbolicBuilderSnapshot {
    pub precompute_root_count: usize,
    pub lookup_is_send: Vec<bool>,
    pub gate_root_count: usize,
    pub required_max_beta_power: usize,
    pub main_width: usize,
    pub preprocessed_width: usize,
    pub public_width: usize,
    pub commit_scope: InteractionScope,
    pub logup_batch_size: usize,
    pub builder_version: u32,
    pub adapter_version: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecursionFixedSymbolicProgramError {
    Adapter(RecursionAdapterError),
}

impl From<RecursionAdapterError> for RecursionFixedSymbolicProgramError {
    fn from(value: RecursionAdapterError) -> Self {
        Self::Adapter(value)
    }
}

impl RecursionFixedSymbolicProgram {
    pub fn new(
        version: u32,
        role: RecursionChildRole,
        chips: Vec<RecursionFixedSymbolicChip>,
        artifact_digest: [F; DIGEST_SIZE],
    ) -> Result<Self, RecursionFixedSymbolicProgramError> {
        let max_required_beta_power = chips
            .iter()
            .map(|chip| chip.builder_snapshot.required_max_beta_power)
            .max()
            .unwrap_or(0);
        let mut program = Self {
            version,
            role,
            chips,
            max_required_beta_power,
            canonical_bytes_len: 0,
            artifact_digest,
        };
        program.canonical_bytes_len = program.canonical_payload_bytes()?.len();
        Ok(program)
    }

    pub fn canonical_payload_bytes(&self) -> Result<Vec<u8>, RecursionFixedSymbolicProgramError> {
        let adapter = RecursionSymbolicExprAdapter::default();
        let mut out = Vec::new();
        put_u32(&mut out, self.version);
        put_u8(&mut out, role_tag(self.role));
        put_usize(&mut out, self.chips.len());
        for chip in &self.chips {
            chip.encode_canonical_payload(&adapter, &mut out)?;
        }
        put_usize(&mut out, self.max_required_beta_power);
        Ok(out)
    }

    pub fn canonical_bytes_with_digest(
        &self,
    ) -> Result<Vec<u8>, RecursionFixedSymbolicProgramError> {
        let mut out = self.canonical_payload_bytes()?;
        for limb in self.artifact_digest {
            encode_f(&mut out, &limb);
        }
        Ok(out)
    }
}

impl RecursionFixedSymbolicChip {
    pub fn from_polyair_chip<A>(
        static_chip_id: usize,
        chip: &Chip<A, F, D_EF>,
    ) -> Result<Self, RecursionFixedSymbolicProgramError>
    where
        A: MachineAir<F>,
    {
        Self::from_symbolic_builder(
            static_chip_id,
            chip.air.name(),
            chip.air.commit_scope(),
            chip.logup_batch_size(),
            chip.num_alpha,
            &chip.symbolic_builder,
        )
    }

    pub fn from_symbolic_builder(
        static_chip_id: usize,
        chip_name: String,
        commit_scope: InteractionScope,
        logup_batch_size: usize,
        num_constraints_from_builder: usize,
        builder: &SymbolicAirBuilder<F, D_EF>,
    ) -> Result<Self, RecursionFixedSymbolicProgramError> {
        let public_width = max_public_width(builder);
        let builder_snapshot = RecursionSymbolicBuilderSnapshot::from_symbolic_builder(
            builder,
            commit_scope,
            logup_batch_size,
            public_width,
        );
        let adapted_roots = RecursionSymbolicBuilderSnapshot::adapt_roots(builder)?;
        let pre_fusion_node_count = adapted_roots.node_table.len();
        let pre_fusion_mul_count = adapted_roots.op_mix.muls;
        let adapted_roots = fuse_mul_add_nodes(adapted_roots)?;
        let fused_count = pre_fusion_node_count.saturating_sub(adapted_roots.node_table.len());
        if fused_count != 0 && crate::debug_prints_enabled() {
            eprintln!(
                "[native-recursion][v8] chip={} fused={}/{} nodes ({} bps), fused/mul={} bps",
                chip_name,
                fused_count,
                pre_fusion_node_count,
                ratio_bps(fused_count, pre_fusion_node_count),
                ratio_bps(fused_count, pre_fusion_mul_count)
            );
        }
        Ok(Self {
            static_chip_id,
            chip_name,
            main_width: builder.main.len(),
            preprocessed_width: builder.preprocessed.len(),
            public_width,
            commit_scope,
            reserved_poly: builder.reserved_poly_output.clone(),
            logup_batch_size,
            num_gate_roots: builder.gate.len(),
            num_constraints_from_builder,
            builder_snapshot,
            adapted_roots,
        })
    }

    fn encode_canonical_payload(
        &self,
        adapter: &RecursionSymbolicExprAdapter,
        out: &mut Vec<u8>,
    ) -> Result<(), RecursionFixedSymbolicProgramError> {
        put_usize(out, self.static_chip_id);
        encode_string(out, &self.chip_name);
        put_usize(out, self.main_width);
        put_usize(out, self.preprocessed_width);
        put_usize(out, self.public_width);
        put_u8(out, commit_scope_tag(self.commit_scope));
        encode_pair_cols(out, &self.reserved_poly);
        put_usize(out, self.logup_batch_size);
        put_usize(out, self.num_gate_roots);
        put_usize(out, self.num_constraints_from_builder);
        self.builder_snapshot.encode_canonical_metadata(out);
        let root_bytes = adapter.canonical_bytes_for_root_streams(&self.adapted_roots);
        put_usize(out, root_bytes.len());
        out.extend_from_slice(&root_bytes);
        Ok(())
    }
}

impl RecursionSymbolicBuilderSnapshot {
    pub fn from_symbolic_builder(
        builder: &SymbolicAirBuilder<F, D_EF>,
        commit_scope: InteractionScope,
        logup_batch_size: usize,
        public_width: usize,
    ) -> Self {
        let lookup_is_send = builder.lookup_infos.iter().map(|lookup| lookup.is_send).collect();
        Self {
            precompute_root_count: builder.precomputed_lc_output.len(),
            lookup_is_send,
            gate_root_count: builder.gate.len(),
            required_max_beta_power: builder.beta_powers.len().saturating_sub(1),
            main_width: builder.main.len(),
            preprocessed_width: builder.preprocessed.len(),
            public_width,
            commit_scope,
            logup_batch_size,
            builder_version: 1,
            adapter_version: RecursionSymbolicExprAdapter::default().version,
        }
    }

    pub fn adapt_roots(
        builder: &SymbolicAirBuilder<F, D_EF>,
    ) -> Result<RecursionAdaptedRootStreams, RecursionFixedSymbolicProgramError> {
        let adapter = RecursionSymbolicExprAdapter::default();
        let lookup_multiplicity_roots = builder
            .lookup_infos
            .iter()
            .map(|lookup| lookup.multiplicity.clone())
            .collect::<Vec<_>>();
        Ok(adapter.adapt_root_streams(
            &builder.precomputed_lc_output,
            &lookup_multiplicity_roots,
            &builder.gate,
        )?)
    }

    fn encode_canonical_metadata(&self, out: &mut Vec<u8>) {
        put_usize(out, self.precompute_root_count);
        put_usize(out, self.lookup_is_send.len());
        for is_send in &self.lookup_is_send {
            put_u8(out, u8::from(*is_send));
        }
        put_usize(out, self.gate_root_count);
        put_usize(out, self.required_max_beta_power);
        put_usize(out, self.main_width);
        put_usize(out, self.preprocessed_width);
        put_usize(out, self.public_width);
        put_u8(out, commit_scope_tag(self.commit_scope));
        put_usize(out, self.logup_batch_size);
        put_u32(out, self.builder_version);
        put_u32(out, self.adapter_version);
    }
}

fn max_public_width(builder: &SymbolicAirBuilder<F, D_EF>) -> usize {
    let mut max_index = None;
    observe_public_width(&mut max_index, &builder.precomputed_lc_output);
    observe_public_width(&mut max_index, &builder.gate);
    for lookup in &builder.lookup_infos {
        observe_public_width(&mut max_index, std::slice::from_ref(&lookup.multiplicity));
    }
    max_index.map_or(0, |index| index + 1)
}

fn ratio_bps(numerator: usize, denominator: usize) -> usize {
    if denominator == 0 {
        0
    } else {
        numerator.saturating_mul(10_000) / denominator
    }
}

fn observe_public_width(
    max_index: &mut Option<usize>,
    expressions: &[RecursionSymbolicExpression],
) {
    let mut visited = HashSet::new();
    for expression in expressions {
        observe_public_width_expr(max_index, &mut visited, expression);
    }
}

fn observe_public_width_expr(
    max_index: &mut Option<usize>,
    visited: &mut HashSet<usize>,
    expression: &RecursionSymbolicExpression,
) {
    let ptr = expression as *const RecursionSymbolicExpression as usize;
    if !visited.insert(ptr) {
        return;
    }

    match expression {
        SymbolicExpression::VARiable(SymbolicVar::Public(index)) => {
            *max_index = Some(max_index.map_or(*index, |current| current.max(*index)));
        }
        SymbolicExpression::VARiable(_) |
        SymbolicExpression::Constant(_) |
        SymbolicExpression::ConstantExt(_) => {}
        SymbolicExpression::Add { x, y, .. } |
        SymbolicExpression::Sub { x, y, .. } |
        SymbolicExpression::Mul { x, y, .. } => {
            observe_public_width_expr(max_index, visited, x.as_ref());
            observe_public_width_expr(max_index, visited, y.as_ref());
        }
        SymbolicExpression::Neg { x, .. } => {
            observe_public_width_expr(max_index, visited, x.as_ref());
        }
    }
}

fn encode_pair_cols(out: &mut Vec<u8>, pair_cols: &[PairCol]) {
    put_usize(out, pair_cols.len());
    for pair_col in pair_cols {
        match pair_col {
            PairCol::Prep(col) => {
                put_u8(out, 0);
                put_usize(out, *col);
            }
            PairCol::Main(col) => {
                put_u8(out, 1);
                put_usize(out, *col);
            }
        }
    }
}

fn encode_string(out: &mut Vec<u8>, value: &str) {
    put_usize(out, value.len());
    out.extend_from_slice(value.as_bytes());
}

fn role_tag(role: RecursionChildRole) -> u8 {
    match role {
        RecursionChildRole::Core => 0,
        RecursionChildRole::Compress => 1,
        RecursionChildRole::Shrink => 2,
    }
}

fn commit_scope_tag(scope: InteractionScope) -> u8 {
    match scope {
        InteractionScope::Local => 0,
        InteractionScope::Global => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_field::AbstractField;
    use polyair::symbolic::SymbolicExpression;

    #[test]
    fn canonical_payload_changes_when_gate_changes() {
        let digest = [F::zero(); DIGEST_SIZE];
        let mut builder = SymbolicAirBuilder::<F, D_EF>::new_empty();
        builder.with_main_width(1);
        builder.gate.push(SymbolicExpression::VARiable(SymbolicVar::Public(0)));

        let chip_a = RecursionFixedSymbolicChip::from_symbolic_builder(
            0,
            "chip".to_string(),
            InteractionScope::Local,
            1,
            1,
            &builder,
        )
        .unwrap();
        let program_a = RecursionFixedSymbolicProgram::new(
            crate::symbolic_ir_dt::CONSTRAINT_PROGRAM_SCHEMA_VERSION,
            RecursionChildRole::Core,
            vec![chip_a],
            digest,
        )
        .unwrap();

        builder.gate.push(SymbolicExpression::from(F::one()));
        let chip_b = RecursionFixedSymbolicChip::from_symbolic_builder(
            0,
            "chip".to_string(),
            InteractionScope::Local,
            1,
            2,
            &builder,
        )
        .unwrap();
        let program_b = RecursionFixedSymbolicProgram::new(
            crate::symbolic_ir_dt::CONSTRAINT_PROGRAM_SCHEMA_VERSION,
            RecursionChildRole::Core,
            vec![chip_b],
            digest,
        )
        .unwrap();

        assert_ne!(
            program_a.canonical_payload_bytes().unwrap(),
            program_b.canonical_payload_bytes().unwrap()
        );
    }

    #[test]
    fn public_width_is_max_referenced_index_plus_one() {
        let mut builder = SymbolicAirBuilder::<F, D_EF>::new_empty();
        builder.gate.push(SymbolicExpression::VARiable(SymbolicVar::Public(9)));
        let chip = RecursionFixedSymbolicChip::from_symbolic_builder(
            0,
            "chip".to_string(),
            InteractionScope::Local,
            1,
            1,
            &builder,
        )
        .unwrap();
        assert_eq!(chip.public_width, 10);
    }
}
