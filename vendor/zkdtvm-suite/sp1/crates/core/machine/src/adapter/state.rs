use dt_core_executor::{events::ByteRecord, ByteOpcode};
use dt_derive::AlignedBorrow;
use dt_stark::air::{DTAirBuilder, FullAirBuilder};
use p3_air::AirBuilder;
use p3_field::{AbstractField, Field};
use serde::{Deserialize, Serialize};

// use crate::operations::BabyBearWordRangeChecker;

/// A set of columns to describe the state of the CPU.
/// The state is composed of the shard, clock, and the program counter.
/// The clock is split into 24 bits, 8 bits, 16 bits limbs.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy, Serialize, Deserialize)]
#[repr(C)]
pub struct CPUState<T> {
    pub shard: T,
    pub clk_16_28: T,
    pub clk_0_16: T,
    pub pc: T,
    // pub pc_range_checker: BabyBearWordRangeChecker<T>,
}

impl<T: Copy> CPUState<T> {
    pub fn clk<AB>(&self) -> AB::Expr
    where
        AB: DTAirBuilder<Var = T>,
        T: Into<AB::Expr>,
    {
        self.clk_0_16.into() + self.clk_16_28.into() * AB::Expr::from_canonical_u32(1 << 16)
    }
}

impl<F: Field> CPUState<F> {
    #[allow(clippy::too_many_arguments)]
    pub fn populate(&mut self, blu_events: &mut impl ByteRecord, clk: u32, pc: u32, shard: u32) {
        // let clk_high = (clk >> 24) as u32;
        let clk_16_28 = ((clk >> 16) & 0x0FFF) as u16;
        let clk_0_16 = (clk & 0xFFFF) as u16;

        self.clk_16_28 = F::from_canonical_u16(clk_16_28);
        self.clk_0_16 = F::from_canonical_u16(clk_0_16);
        self.pc = F::from_canonical_u32(pc);
        // self.pc_range_checker.populate(self.pc, blu_events);
        self.shard = F::from_canonical_u32(shard);

        blu_events.add_u16_range_check(clk_0_16);
        blu_events.add_bit_range_check(clk_16_28, 12);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn eval<AB: DTAirBuilder>(
        builder: &mut AB,
        cols: CPUState<AB::Var>,
        next_pc: AB::Expr,
        clk_increment: AB::Expr,
        is_real: AB::Expr,
        expected_shard: AB::Expr,
    ) {
        let clk = cols.clk::<AB>();
        builder.assert_bool(is_real.clone());
        // Constrain that the trace shard matches the expected shard (execution_shard).
        builder.when(is_real.clone()).assert_eq(cols.shard, expected_shard);
        builder.receive_state(cols.shard, clk.clone(), cols.pc, is_real.clone());
        builder.send_state(cols.shard, clk.clone() + clk_increment, next_pc, is_real.clone());
        // BabyBearWordRangeChecker::<AB::F>::range_check(
        //     builder,
        //     cols.pc,
        //     cols.pc_range_checker,
        //     is_real.clone(),
        // );
        // Range check clk_0_16 as a u16 (must match populate's add_u16_range_check).
        builder.send_byte(
            AB::Expr::from_canonical_u32(ByteOpcode::U16Range as u32),
            cols.clk_0_16,
            AB::Expr::zero(),
            AB::Expr::zero(),
            is_real.clone(),
        );
        // Range check clk_16_28 fits in 12 bits (must match populate's add_bit_range_check).
        builder.send_byte(
            AB::Expr::from_canonical_u32(ByteOpcode::BitRange as u32),
            cols.clk_16_28,
            AB::Expr::from_canonical_u32(12),
            AB::Expr::zero(),
            is_real.clone(),
        );
    }
}

pub(crate) const CPU_STATE_NUM_INTERACTIONS: usize = 4;

pub fn cpu_state_gate_constraints<AB: dt_stark::air::FullAirBuilder>(
    builder: &mut AB,
    shard: AB::VarMaybeExt,
    execution_shard: AB::VarMaybeExt,
    is_real: AB::VarMaybeExt,
) {
    builder.when(is_real).assert_zero(shard - execution_shard);
}

pub fn cpu_state_precompute_lc<AB: dt_stark::air::FullAirBuilder>(
    builder: &mut AB,
    shard: AB::VarMaybeExt,
    clk: AB::VarMaybeExt,
    clk_0_16: AB::VarMaybeExt,
    clk_16_28: AB::VarMaybeExt,
    pc: AB::VarMaybeExt,
    next_pc: AB::VarMaybeExt,
) {
    use dt_core_executor::{ByteOpcode, DEFAULT_PC_INC};
    use dt_stark::InteractionKind;
    use p3_field::AbstractField;

    let zero = AB::zero_maybe();
    let state_kind =
        AB::VarMaybeExt::from(AB::F::from_canonical_usize(InteractionKind::State as usize));
    let byte_kind =
        AB::VarMaybeExt::from(AB::F::from_canonical_usize(InteractionKind::Byte as usize));
    let u16_opcode = AB::VarMaybeExt::from(AB::F::from_canonical_u8(ByteOpcode::U16Range as u8));
    let bit_opcode = AB::VarMaybeExt::from(AB::F::from_canonical_u8(ByteOpcode::BitRange as u8));
    let twelve = AB::VarMaybeExt::from(AB::F::from_canonical_u32(12));

    builder.retain_precomputed(
        builder.lookup_denominator(state_kind.clone(), vec![shard.clone(), clk.clone(), pc]),
    );
    builder.retain_precomputed(builder.lookup_denominator(
        state_kind,
        vec![
            shard,
            clk + AB::VarMaybeExt::from(AB::F::from_canonical_u32(DEFAULT_PC_INC)),
            next_pc,
        ],
    ));
    builder.retain_precomputed(builder.lookup_denominator(
        byte_kind.clone(),
        vec![u16_opcode, clk_0_16, zero.clone(), zero.clone(), zero.clone()],
    ));
    builder.retain_precomputed(
        builder
            .lookup_denominator(byte_kind, vec![bit_opcode, clk_16_28, zero.clone(), twelve, zero]),
    );
}

pub fn cpu_state_lookup<AB: dt_stark::air::FullAirBuilder>(
    builder: &mut AB,
    is_real: AB::VarMaybeExt,
) {
    builder.recv(is_real.clone());
    builder.send(is_real.clone());
    builder.send(is_real.clone());
    builder.send(is_real);
}
