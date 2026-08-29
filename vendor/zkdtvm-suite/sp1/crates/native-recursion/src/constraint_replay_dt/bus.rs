use dt_stark::air::FullAirBuilder;

use crate::{
    config::D_EF,
    interaction_full_air_dt::RecursionFullAirBus,
    interaction_registry_dt::{
        BETA_LADDER_CHAIN_SCHEMA, CONSTRAINT_CHALLENGE_SCHEMA, CONSTRAINT_EQ_CHAIN_SCHEMA,
        CONSTRAINT_FOLD_CHAIN_SCHEMA, CONSTRAINT_FOLD_PLAN_CHAIN_SCHEMA,
        CONSTRAINT_HEIGHT_INVERSE_SCHEMA, CONSTRAINT_NODE_VALUE_SCHEMA, CONSTRAINT_PROGRAM_SCHEMA,
        CONSTRAINT_ROOT_TABLE_SCHEMA,
    },
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConstraintProgramBus {
    bus: RecursionFullAirBus,
}

impl ConstraintProgramBus {
    pub const fn new() -> Self {
        Self { bus: RecursionFullAirBus::new(CONSTRAINT_PROGRAM_SCHEMA) }
    }

    pub const fn required_max_beta_power_floor(&self) -> usize {
        self.bus.required_max_beta_power_floor()
    }

    pub fn denominator<AB>(
        &self,
        builder: &AB,
        static_chip_id: AB::VarMaybeExt,
        node_idx: AB::VarMaybeExt,
        op_code: AB::VarMaybeExt,
        lhs_idx: AB::VarMaybeExt,
        rhs_idx: AB::VarMaybeExt,
        third_idx: AB::VarMaybeExt,
        aux: AB::VarMaybeExt,
        leaf_kind: AB::VarMaybeExt,
        fanout: AB::VarMaybeExt,
    ) -> AB::VarExt
    where
        AB: FullAirBuilder,
    {
        self.bus.denominator(
            builder,
            [
                static_chip_id,
                node_idx,
                op_code,
                lhs_idx,
                rhs_idx,
                third_idx,
                aux,
                leaf_kind,
                fanout,
            ],
        )
    }
}

impl Default for ConstraintProgramBus {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConstraintRootTableBus {
    bus: RecursionFullAirBus,
}

impl ConstraintRootTableBus {
    pub const fn new() -> Self {
        Self { bus: RecursionFullAirBus::new(CONSTRAINT_ROOT_TABLE_SCHEMA) }
    }

    pub const fn required_max_beta_power_floor(&self) -> usize {
        self.bus.required_max_beta_power_floor()
    }

    pub fn denominator<AB>(
        &self,
        builder: &AB,
        static_chip_id: AB::VarMaybeExt,
        root_kind: AB::VarMaybeExt,
        root_ord: AB::VarMaybeExt,
        node_idx: AB::VarMaybeExt,
        sign: AB::VarMaybeExt,
    ) -> AB::VarExt
    where
        AB: FullAirBuilder,
    {
        self.bus.denominator(builder, [static_chip_id, root_kind, root_ord, node_idx, sign])
    }
}

impl Default for ConstraintRootTableBus {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConstraintHeightInverseBus {
    bus: RecursionFullAirBus,
}

impl ConstraintHeightInverseBus {
    pub const fn new() -> Self {
        Self { bus: RecursionFullAirBus::new(CONSTRAINT_HEIGHT_INVERSE_SCHEMA) }
    }

    pub const fn required_max_beta_power_floor(&self) -> usize {
        self.bus.required_max_beta_power_floor()
    }

    pub fn denominator<AB>(
        &self,
        builder: &AB,
        log_height: AB::VarMaybeExt,
        height_inverse: AB::VarMaybeExt,
    ) -> AB::VarExt
    where
        AB: FullAirBuilder,
    {
        self.bus.denominator(builder, [log_height, height_inverse])
    }
}

impl Default for ConstraintHeightInverseBus {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConstraintNodeValueBus {
    bus: RecursionFullAirBus,
}

impl ConstraintNodeValueBus {
    pub const fn new() -> Self {
        Self { bus: RecursionFullAirBus::new(CONSTRAINT_NODE_VALUE_SCHEMA) }
    }

    pub const fn required_max_beta_power_floor(&self) -> usize {
        self.bus.required_max_beta_power_floor()
    }

    pub fn denominator<AB>(
        &self,
        builder: &AB,
        proof_idx: AB::VarMaybeExt,
        chip_idx: AB::VarMaybeExt,
        static_chip_id: AB::VarMaybeExt,
        node_idx: AB::VarMaybeExt,
        value: [AB::VarMaybeExt; D_EF],
    ) -> AB::VarExt
    where
        AB: FullAirBuilder,
    {
        let mut values = Vec::with_capacity(3 + D_EF);
        values.push(chip_idx);
        values.push(static_chip_id);
        values.push(node_idx);
        values.extend(value);
        self.bus.denominator_for_proof(builder, proof_idx, values)
    }
}

impl Default for ConstraintNodeValueBus {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConstraintChallengeBus {
    bus: RecursionFullAirBus,
}

impl ConstraintChallengeBus {
    pub const fn new() -> Self {
        Self { bus: RecursionFullAirBus::new(CONSTRAINT_CHALLENGE_SCHEMA) }
    }

    pub const fn required_max_beta_power_floor(&self) -> usize {
        self.bus.required_max_beta_power_floor()
    }

    pub fn denominator<AB>(
        &self,
        builder: &AB,
        proof_idx: AB::VarMaybeExt,
        kind: AB::VarMaybeExt,
        key0: AB::VarMaybeExt,
        key1: AB::VarMaybeExt,
        value: [AB::VarMaybeExt; D_EF],
    ) -> AB::VarExt
    where
        AB: FullAirBuilder,
    {
        let mut values = Vec::with_capacity(3 + D_EF);
        values.push(kind);
        values.push(key0);
        values.push(key1);
        values.extend(value);
        self.bus.denominator_for_proof(builder, proof_idx, values)
    }
}

impl Default for ConstraintChallengeBus {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BetaLadderChainBus {
    bus: RecursionFullAirBus,
}

impl BetaLadderChainBus {
    pub const fn new() -> Self {
        Self { bus: RecursionFullAirBus::new(BETA_LADDER_CHAIN_SCHEMA) }
    }

    pub const fn required_max_beta_power_floor(&self) -> usize {
        self.bus.required_max_beta_power_floor()
    }

    pub fn denominator<AB>(
        &self,
        builder: &AB,
        proof_idx: AB::VarMaybeExt,
        power_idx: AB::VarMaybeExt,
        power: [AB::VarMaybeExt; D_EF],
        beta: [AB::VarMaybeExt; D_EF],
    ) -> AB::VarExt
    where
        AB: FullAirBuilder,
    {
        let mut values = Vec::with_capacity(1 + 2 * D_EF);
        values.push(power_idx);
        values.extend(power);
        values.extend(beta);
        self.bus.denominator_for_proof(builder, proof_idx, values)
    }
}

impl Default for BetaLadderChainBus {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConstraintFoldChainBus {
    bus: RecursionFullAirBus,
}

impl ConstraintFoldChainBus {
    pub const fn new() -> Self {
        Self { bus: RecursionFullAirBus::new(CONSTRAINT_FOLD_CHAIN_SCHEMA) }
    }

    pub const fn required_max_beta_power_floor(&self) -> usize {
        self.bus.required_max_beta_power_floor()
    }

    /// Ordered payload:
    /// `cursor, alpha[5], acc[5], pacc[5], perm_sum[5]`.
    pub fn denominator<AB>(
        &self,
        builder: &AB,
        proof_idx: AB::VarMaybeExt,
        cursor: AB::VarMaybeExt,
        alpha: [AB::VarMaybeExt; D_EF],
        acc: [AB::VarMaybeExt; D_EF],
        pacc: [AB::VarMaybeExt; D_EF],
        perm_sum: [AB::VarMaybeExt; D_EF],
    ) -> AB::VarExt
    where
        AB: FullAirBuilder,
    {
        let mut values = Vec::with_capacity(1 + 4 * D_EF);
        values.push(cursor);
        values.extend(alpha);
        values.extend(acc);
        values.extend(pacc);
        values.extend(perm_sum);
        self.bus.denominator_for_proof(builder, proof_idx, values)
    }
}

impl Default for ConstraintFoldChainBus {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConstraintFoldPlanChainBus {
    bus: RecursionFullAirBus,
}

impl ConstraintFoldPlanChainBus {
    pub const fn new() -> Self {
        Self { bus: RecursionFullAirBus::new(CONSTRAINT_FOLD_PLAN_CHAIN_SCHEMA) }
    }

    pub const fn required_max_beta_power_floor(&self) -> usize {
        self.bus.required_max_beta_power_floor()
    }

    /// Ordered payload: `cursor, remaining_chips, local_ord`.
    pub fn denominator<AB>(
        &self,
        builder: &AB,
        proof_idx: AB::VarMaybeExt,
        cursor: AB::VarMaybeExt,
        remaining_chips: AB::VarMaybeExt,
        local_ord: AB::VarMaybeExt,
    ) -> AB::VarExt
    where
        AB: FullAirBuilder,
    {
        self.bus.denominator_for_proof(builder, proof_idx, [cursor, remaining_chips, local_ord])
    }
}

impl Default for ConstraintFoldPlanChainBus {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConstraintEqChainBus {
    bus: RecursionFullAirBus,
}

impl ConstraintEqChainBus {
    pub const fn new() -> Self {
        Self { bus: RecursionFullAirBus::new(CONSTRAINT_EQ_CHAIN_SCHEMA) }
    }

    pub const fn required_max_beta_power_floor(&self) -> usize {
        self.bus.required_max_beta_power_floor()
    }

    pub fn denominator<AB>(
        &self,
        builder: &AB,
        proof_idx: AB::VarMaybeExt,
        round_idx: AB::VarMaybeExt,
        eq_acc: [AB::VarMaybeExt; D_EF],
        first_prefix: [AB::VarMaybeExt; D_EF],
        last_prefix: [AB::VarMaybeExt; D_EF],
    ) -> AB::VarExt
    where
        AB: FullAirBuilder,
    {
        let mut values = Vec::with_capacity(1 + 3 * D_EF);
        values.push(round_idx);
        values.extend(eq_acc);
        values.extend(first_prefix);
        values.extend(last_prefix);
        self.bus.denominator_for_proof(builder, proof_idx, values)
    }
}

impl Default for ConstraintEqChainBus {
    fn default() -> Self {
        Self::new()
    }
}
