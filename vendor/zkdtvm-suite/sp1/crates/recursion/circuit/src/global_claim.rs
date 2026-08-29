use std::borrow::Borrow;

use dt_recursion_compiler::ir::{Builder, Config, Ext, Felt};
use dt_stark::{
    air::{active_shape_transcript_words_v2, ActiveShapeEntryV1, PublicValues},
    global_d11::PROJECTIVE_CHAIN_BLOCK_WIDTH,
    InteractionKind, Word,
};
use p3_field::{AbstractExtensionField, AbstractField, PrimeField32};

use crate::{challenger::CanObserveVariable, CircuitConfig};

pub(crate) fn observe_active_shape<C, Challenger>(
    builder: &mut Builder<C>,
    challenger: &mut Challenger,
    entries: &[ActiveShapeEntryV1],
) where
    C: CircuitConfig,
    C::F: PrimeField32,
    Challenger: CanObserveVariable<C, Felt<C::F>>,
{
    for word in active_shape_transcript_words_v2(entries) {
        let value = builder.eval(C::F::from_canonical_u32(word));
        challenger.observe(builder, value);
    }
}

fn nonzero_flag<C: CircuitConfig>(builder: &mut Builder<C>, value: Felt<C::F>) -> Felt<C::F>
where
    C::F: PrimeField32,
{
    let mut exponent = u64::from(C::F::ORDER_U32) - 1;
    let mut base = value;
    let mut result = builder.eval(C::F::one());
    while exponent != 0 {
        if exponent & 1 == 1 {
            result = builder.eval(result * base);
        }
        exponent >>= 1;
        if exponent != 0 {
            base = builder.eval(base * base);
        }
    }
    result
}

pub(crate) type D11PointVar<C> = [[Felt<<C as Config>::F>; 11]; 3];

pub(crate) fn global_identity<C: CircuitConfig>(builder: &mut Builder<C>) -> D11PointVar<C> {
    let zero = builder.eval(C::F::zero());
    let one = builder.eval(C::F::one());
    let mut point = [[zero; 11]; 3];
    point[1][0] = one;
    point
}

fn assert_canonical_point<C: CircuitConfig>(
    builder: &mut Builder<C>,
    point: D11PointVar<C>,
) where
    C::F: PrimeField32,
{
    let zero: Felt<C::F> = builder.eval(C::F::zero());
    let one: Felt<C::F> = builder.eval(C::F::one());
    let finite = point[2][0];
    builder.assert_felt_eq(finite * (finite - one), zero);
    for limb in 1..11 {
        builder.assert_felt_eq(point[2][limb], zero);
    }
    for limb in 0..11 {
        builder.assert_felt_eq((one - finite) * point[0][limb], zero);
        let expected_y = if limb == 0 { one } else { zero };
        builder.assert_felt_eq((one - finite) * (point[1][limb] - expected_y), zero);
    }
}

pub(crate) fn assert_canonical_shard_interval<C: CircuitConfig>(
    builder: &mut Builder<C>,
    has: Felt<C::F>,
    count: Felt<C::F>,
    start: D11PointVar<C>,
    end: D11PointVar<C>,
) where
    C::F: PrimeField32,
{
    let zero: Felt<C::F> = builder.eval(C::F::zero());
    let one: Felt<C::F> = builder.eval(C::F::one());
    builder.assert_felt_eq(has * (has - one), zero);
    builder.assert_felt_eq((one - has) * count, zero);
    builder.assert_felt_ne(has * count + (one - has), zero);
    for coordinate in 0..3 {
        for limb in 0..11 {
            builder.assert_felt_eq(
                (one - has) * (end[coordinate][limb] - start[coordinate][limb]),
                zero,
            );
        }
    }
    assert_canonical_point(builder, start);
    assert_canonical_point(builder, end);
}

fn fingerprint<C: CircuitConfig>(
    builder: &mut Builder<C>,
    alpha: Ext<C::F, C::EF>,
    beta: Ext<C::F, C::EF>,
    payload: &[Felt<C::F>],
) -> Ext<C::F, C::EF>
where
    C::F: PrimeField32,
{
    assert_eq!(C::EF::D, PROJECTIVE_CHAIN_BLOCK_WIDTH);
    let kind: Ext<C::F, C::EF> = builder
        .constant(C::EF::from_canonical_usize(InteractionKind::GlobalProjectiveChainV2 as usize));
    let zero = builder.eval(C::F::zero());
    let mut result = builder.eval(alpha + kind);
    let mut beta_power = beta;
    for chunk in payload.chunks(PROJECTIVE_CHAIN_BLOCK_WIDTH) {
        let mut limbs = [zero; PROJECTIVE_CHAIN_BLOCK_WIDTH];
        limbs[..chunk.len()].copy_from_slice(chunk);
        let block = builder.ext_from_base_slice(&limbs);
        result = builder.eval(result + beta_power * block);
        beta_power = builder.eval(beta_power * beta);
    }
    result
}

/// Compute the authenticated state/address/Global-interval lookup imbalance.
pub(crate) fn expected_local_imbalance<C: CircuitConfig>(
    builder: &mut Builder<C>,
    public_values: &[Felt<C::F>],
    alpha: Ext<C::F, C::EF>,
    beta: Ext<C::F, C::EF>,
) -> Ext<C::F, C::EF>
where
    C::F: PrimeField32,
{
    let pv: &PublicValues<Word<Felt<C::F>>, Felt<C::F>> = public_values.borrow();
    let one_ext: Ext<C::F, C::EF> = builder.constant(C::EF::one());
    let beta2: Ext<C::F, C::EF> = builder.eval(beta * beta);
    let beta3: Ext<C::F, C::EF> = builder.eval(beta2 * beta);
    let mut result: Ext<C::F, C::EF> = builder.constant(C::EF::zero());

    let state_kind: Ext<C::F, C::EF> =
        builder.constant(C::EF::from_canonical_usize(InteractionKind::State as usize));
    let shard_term: Ext<C::F, C::EF> = builder.eval(beta * pv.execution_shard);
    let recv_state: Ext<C::F, C::EF> =
        builder.eval(alpha + state_kind + shard_term + beta2 * pv.start_clk + beta3 * pv.start_pc);
    let send_state: Ext<C::F, C::EF> =
        builder.eval(alpha + state_kind + shard_term + beta2 * pv.exit_clk + beta3 * pv.next_pc);
    let clock_diff = builder.eval(pv.start_clk - pv.exit_clk);
    let clock_flag = nonzero_flag(builder, clock_diff);
    let state_contribution: Ext<C::F, C::EF> =
        builder.eval(one_ext / send_state - one_ext / recv_state);
    result = builder.eval(result + state_contribution * clock_flag);

    let address_kind: Ext<C::F, C::EF> = builder.constant(C::EF::from_canonical_usize(
        InteractionKind::MemoryGlobalAddr as usize,
    ));
    let init_base: Ext<C::F, C::EF> = builder.eval(alpha + address_kind);
    let recv_init: Ext<C::F, C::EF> = builder.eval(init_base + beta2 * pv.previous_init_addr);
    let send_init: Ext<C::F, C::EF> = builder.eval(init_base + beta2 * pv.last_init_addr);
    let init_diff = builder.eval(pv.previous_init_addr - pv.last_init_addr);
    let init_flag = nonzero_flag(builder, init_diff);
    result = builder.eval(result + (one_ext / send_init - one_ext / recv_init) * init_flag);

    let finalize_base: Ext<C::F, C::EF> = builder.eval(alpha + address_kind + beta);
    let recv_finalize: Ext<C::F, C::EF> =
        builder.eval(finalize_base + beta2 * pv.previous_finalize_addr);
    let send_finalize: Ext<C::F, C::EF> =
        builder.eval(finalize_base + beta2 * pv.last_finalize_addr);
    let finalize_diff = builder.eval(pv.previous_finalize_addr - pv.last_finalize_addr);
    let finalize_flag = nonzero_flag(builder, finalize_diff);
    result =
        builder.eval(result + (one_ext / send_finalize - one_ext / recv_finalize) * finalize_flag);

    let zero = builder.eval(C::F::zero());
    let mut source_payload = Vec::with_capacity(34);
    source_payload.push(zero);
    source_payload.extend(pv.global.interval.start.x);
    source_payload.extend(pv.global.interval.start.y);
    source_payload.extend(pv.global.interval.start.z);
    let mut sink_payload = Vec::with_capacity(34);
    sink_payload.push(pv.global.count);
    sink_payload.extend(pv.global.interval.end.x);
    sink_payload.extend(pv.global.interval.end.y);
    sink_payload.extend(pv.global.interval.end.z);
    let source = fingerprint(builder, alpha, beta, &source_payload);
    let sink = fingerprint(builder, alpha, beta, &sink_payload);
    result =
        builder.eval(result + (one_ext / sink - one_ext / source) * pv.global.has_global_opening);
    result
}
