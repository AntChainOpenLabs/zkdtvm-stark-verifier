use core::ops::Deref;
use std::{
    any::{Any, TypeId},
    borrow::Borrow,
};

use dt_core_executor::{ExecutionRecord, Program};
use dt_stark::{
    air::{
        AirInteraction, DTAirBuilder, FullAir, FullAirBuilder, InteractionScope, MachineAir,
        PairCol,
    },
    global_d11::{
        PROJECTIVE_CHAIN_BASE_VALUES, PROJECTIVE_CHAIN_BLOCKS, PROJECTIVE_CHAIN_BLOCK_WIDTH,
    },
    sumcheck::trace::CompressedMatrix,
    InteractionKind,
};
use p3_air::{Air, BaseAir, PairBuilder};
use p3_field::{AbstractField, Field, PrimeField32};
use p3_koala_bear::KoalaBear;
use p3_matrix::Matrix;

use super::{
    columns::{
        GlobalTileReducerCols, NUM_GLOBAL_TILE_REDUCER_COLS, REDUCER_LEAF_END_VALUE,
        REDUCER_LEAF_GAP_BITS, REDUCER_LEAF_GAP_BITS_START, REDUCER_LEAF_K_VALUE,
        REDUCER_LEAF_P_BITS, REDUCER_LEAF_P_BITS_START, REDUCER_PRODUCT_INFINITY_VALUE,
        REDUCER_PRODUCT_CONTINUE_VALUE, REDUCER_PRODUCT_REBASE_VALUE,
        REDUCER_PRODUCT_TO_MIDDLE_VALUE, REDUCER_PRODUCT_TO_NODE_VALUE,
        REDUCER_PRODUCT_TO_ROOT_VALUE, REDUCER_PRODUCT_WAVE_VALUE,
    },
    writer::global_tile_reducer_padded_rows,
};

pub(crate) const GLOBAL_TILE_REDUCER_LOOKUP_SLOTS: usize = 38;
pub(crate) const GLOBAL_TILE_REDUCER_LOOKUP_BATCH_SIZE: usize = 2;
pub(crate) const GLOBAL_TILE_REDUCER_PERMUTATION_WIDTH: usize = 19;
pub(crate) const GLOBAL_TILE_REDUCER_MAX_BETA_POWER: usize = 11;
pub(crate) const GLOBAL_TILE_REDUCER_VALUE_PROJECTIONS: usize = 4;

const RECV_CLASSES: usize = 14;
const SEND_CLASSES: usize = 22;
const SCHEDULE_DOMAIN: u32 = 1 << 23;
const NODE_TAG: u32 = 4;
const REBASE_TAG: u32 = 8;
const ROOT_IN_TAG: u32 = 12;
const ROOT_OUT_TAG: u32 = 16;

#[derive(Clone, Copy, Debug, Default)]
pub struct GlobalTileReducerChip;

#[derive(Clone)]
struct ReducerMessages<T: Clone> {
    recv: [[T; PROJECTIVE_CHAIN_BASE_VALUES]; RECV_CLASSES],
    send: [[T; PROJECTIVE_CHAIN_BASE_VALUES]; SEND_CLASSES],
    recv_mult: [T; RECV_CLASSES],
    send_mult: [T; SEND_CLASSES],
    control_recv: [T; PROJECTIVE_CHAIN_BASE_VALUES],
    control_send: [T; PROJECTIVE_CHAIN_BASE_VALUES],
    control_recv_mult: T,
    control_send_mult: T,
}

fn d11<T: Clone>(values: &[T; 66], group: usize) -> [T; 11] {
    core::array::from_fn(|limb| values[group * 11 + limb].clone())
}

fn add<T: AbstractField + Clone>(left: &[T; 11], right: &[T; 11]) -> [T; 11] {
    core::array::from_fn(|limb| left[limb].clone() + right[limb].clone())
}

fn sub<T: AbstractField + Clone>(left: &[T; 11], right: &[T; 11]) -> [T; 11] {
    core::array::from_fn(|limb| left[limb].clone() - right[limb].clone())
}

fn scale<T: AbstractField + Clone>(value: &[T; 11], constant: u32) -> [T; 11] {
    let constant = T::from_canonical_u32(constant);
    core::array::from_fn(|limb| value[limb].clone() * constant.clone())
}

fn mul_by_b<T: AbstractField + Clone>(value: &[T; 11]) -> [T; 11] {
    let mut result = core::array::from_fn(|limb| value[limb].clone() * T::from_canonical_u32(36));
    result[0] = result[0].clone() + value[10].clone().double();
    for limb in 1..11 {
        result[limb] = result[limb].clone() + value[limb - 1].clone();
    }
    result[3] = result[3].clone() + value[10].clone();
    result
}

fn identity<T: AbstractField + Clone>() -> [[T; 11]; 3] {
    let zero = core::array::from_fn(|_| T::zero());
    let mut y = core::array::from_fn(|_| T::zero());
    y[0] = T::one();
    [zero.clone(), y, zero]
}

fn point_message<T: AbstractField + Clone>(key: T, point: &[[T; 11]; 3]) -> [T; 34] {
    core::array::from_fn(|offset| match offset {
        0 => key.clone(),
        1..=11 => point[0][offset - 1].clone(),
        12..=22 => point[1][offset - 12].clone(),
        _ => point[2][offset - 23].clone(),
    })
}

fn operand_message<T: AbstractField + Clone>(
    key: T,
    left: &[T; 11],
    right: &[T; 11],
    n: T,
    p: T,
    node: T,
    stage: T,
    product: T,
    flow: T,
) -> [T; 34] {
    core::array::from_fn(|offset| match offset {
        0 => key.clone(),
        1..=11 => left[offset - 1].clone(),
        12..=22 => right[offset - 12].clone(),
        23 => n.clone(),
        24 => p.clone(),
        25 => node.clone(),
        26 => stage.clone(),
        27 => product.clone(),
        28 => flow.clone(),
        _ => T::zero(),
    })
}

fn reduced_message<T: AbstractField + Clone>(
    key: T,
    reduced: &[T; 11],
    n: T,
    p: T,
    node: T,
    stage: T,
    product: T,
    flow: T,
) -> [T; 34] {
    core::array::from_fn(|offset| match offset {
        0 => key.clone(),
        1..=11 => reduced[offset - 1].clone(),
        23 => n.clone(),
        24 => p.clone(),
        25 => node.clone(),
        26 => stage.clone(),
        27 => product.clone(),
        28 => flow.clone(),
        _ => T::zero(),
    })
}

fn control_message<T: AbstractField + Clone>(n: T, p: T, rank: T, tag: T) -> [T; 34] {
    core::array::from_fn(|offset| match offset {
        0 => T::from_canonical_u32(SCHEDULE_DOMAIN) + rank.clone(),
        1 => n.clone(),
        2 => p.clone(),
        3 => rank.clone(),
        4 => tag.clone(),
        _ => T::zero(),
    })
}

fn point_key<T: AbstractField + Clone>(n: T, heap_id: T) -> T {
    n + T::one() + heap_id
}

fn product_key<T: AbstractField + Clone>(
    n: T,
    p: T,
    stage: T,
    node: T,
    product: T,
) -> T {
    n +
        T::one() +
        p.clone() * T::from_canonical_u32(2) +
        (node - T::one()) * T::from_canonical_u32(24) +
        stage * T::from_canonical_u32(12) +
        product * T::from_canonical_u32(2)
}

fn first_operands<T: AbstractField + Clone>(
    left: &[[T; 11]; 3],
    right: &[[T; 11]; 3],
) -> [([T; 11], [T; 11]); 6] {
    [
        (left[0].clone(), right[0].clone()),
        (left[1].clone(), right[1].clone()),
        (left[2].clone(), right[2].clone()),
        (add(&left[0], &left[1]), add(&right[0], &right[1])),
        (add(&left[1], &left[2]), add(&right[1], &right[2])),
        (add(&left[0], &left[2]), add(&right[0], &right[2])),
    ]
}

fn second_operands<T: AbstractField + Clone>(
    first: &[[T; 11]; 6],
) -> [([T; 11], [T; 11]); 6] {
    let xx = first[0].clone();
    let yy = first[1].clone();
    let zz = first[2].clone();
    let xy = sub(&first[3], &add(&xx, &yy));
    let yz = sub(&first[4], &add(&yy, &zz));
    let xz = sub(&first[5], &add(&xx, &zz));
    let bzz3 = scale(&sub(&xz, &mul_by_b(&zz)), 3);
    let yy_minus = sub(&yy, &bzz3);
    let yy_plus = add(&yy, &bzz3);
    let zz3 = scale(&zz, 3);
    let bxz3 = scale(&sub(&mul_by_b(&xz), &add(&zz3, &xx)), 3);
    let xx3_minus_zz3 = sub(&scale(&xx, 3), &zz3);
    [
        (yy_plus.clone(), xy.clone()),
        (yz.clone(), bxz3.clone()),
        (yy_plus, yy_minus.clone()),
        (xx3_minus_zz3.clone(), bxz3),
        (yy_minus, yz),
        (xy, xx3_minus_zz3),
    ]
}

fn reducer_messages<T: AbstractField + Clone>(
    cols: &GlobalTileReducerCols<T>,
) -> ReducerMessages<T> {
    let zero_message = core::array::from_fn(|_| T::zero());
    let mut messages = ReducerMessages {
        recv: core::array::from_fn(|_| zero_message.clone()),
        send: core::array::from_fn(|_| zero_message.clone()),
        recv_mult: core::array::from_fn(|_| T::zero()),
        send_mult: core::array::from_fn(|_| T::zero()),
        control_recv: zero_message.clone(),
        control_send: zero_message,
        control_recv_mult: T::zero(),
        control_send_mult: T::zero(),
    };
    let c = &cols.payload.control;
    let v = &cols.payload.values;
    let n = c[0].clone();
    let p = c[1].clone();
    let id = c[2].clone();
    let aux = c[3].clone();
    let leaf_point = [d11(v, 0), d11(v, 1), d11(v, 2)];
    let left = [d11(v, 0), d11(v, 1), d11(v, 2)];
    let right = [d11(v, 3), d11(v, 4), d11(v, 5)];
    let first = [d11(v, 0), d11(v, 1), d11(v, 2), d11(v, 3), d11(v, 4), d11(v, 5)];
    let second = first.clone();
    let raw_output = [
        sub(&second[0], &second[1]),
        add(&second[2], &second[3]),
        add(&second[4], &second[5]),
    ];
    let first_pairs = first_operands(&left, &right);
    let second_pairs = second_operands(&first);
    let product_left = d11(v, 0);
    let product_right = d11(v, 1);
    let product_reduced = d11(v, 2);
    let root_raw = [d11(v, 0), d11(v, 1), d11(v, 2)];
    let root_lambda = d11(v, 3);
    let root_inverse = d11(v, 4);
    let root_reduced = [d11(v, 0), d11(v, 1), d11(v, 2), d11(v, 3)];
    let root_point = [root_reduced[1].clone(), root_reduced[2].clone(), root_reduced[3].clone()];
    let leaf = cols.mode_leaf.clone();
    let node_input = cols.mode_node_input.clone();
    let product = cols.mode_product.clone();
    let middle = cols.mode_node_middle.clone();
    let node_output = cols.mode_node_output.clone();
    let root_input = cols.mode_root_input.clone();
    let root_output = cols.mode_root_output.clone();
    let rebase_input = cols.selector_spare.clone();
    let mut leaf_identity = identity();
    leaf_identity[1][0] = leaf.clone();
    let mut root_identity = identity();
    root_identity[1][0] = root_output.clone();

    messages.recv[0] = point_message(aux.clone(), &leaf_point);
    messages.recv_mult[0] = leaf.clone();
    messages.recv[1] = point_message(
        point_key(n.clone(), id.clone() * T::from_canonical_u32(2)),
        &left,
    );
    messages.recv_mult[1] = node_input.clone();
    messages.recv[2] = point_message(
        point_key(n.clone(), id.clone() * T::from_canonical_u32(2) + T::one()),
        &right,
    );
    messages.recv_mult[2] = node_input.clone();

    let product_stage = v[REDUCER_PRODUCT_WAVE_VALUE].clone() +
        c[3].clone() * T::from_canonical_u32(2);
    let product_id = c[4].clone();
    let product_flow = v[REDUCER_PRODUCT_REBASE_VALUE].clone() +
        c[3].clone() * T::from_canonical_u32(4) +
        v[REDUCER_PRODUCT_INFINITY_VALUE].clone();
    let product_message_key = product_key(
        n.clone(),
        p.clone(),
        product_stage.clone(),
        id.clone(),
        product_id.clone(),
    );
    messages.recv[3] = operand_message(
        product_message_key.clone(),
        &product_left,
        &product_right,
        n.clone(),
        p.clone(),
        id.clone(),
        product_stage.clone(),
        product_id.clone(),
        product_flow.clone(),
    );
    messages.recv_mult[3] = product.clone();

    let merged_stage = node_output.clone() + root_output.clone() * T::from_canonical_u32(2);
    let merged_flow = c[3].clone() + root_output.clone() * T::from_canonical_u32(4);
    for product_id in 0..6 {
        let product_id_value = T::from_canonical_usize(product_id);
        messages.recv[4 + product_id] = reduced_message(
            product_key(
                n.clone(),
                p.clone(),
                merged_stage.clone(),
                id.clone(),
                product_id_value.clone(),
            ) + T::one(),
            &first[product_id],
            n.clone(),
            p.clone(),
            id.clone(),
            merged_stage.clone(),
            product_id_value,
            merged_flow.clone(),
        );
        messages.recv_mult[4 + product_id] = middle.clone() + node_output.clone() +
            if product_id < 4 { root_output.clone() } else { T::zero() };
    }
    messages.recv[10] = point_message(T::zero(), &left);
    messages.recv_mult[10] = rebase_input.clone();
    messages.recv[11] = point_message(point_key(n.clone(), T::one()), &right);
    messages.recv_mult[11] = rebase_input.clone();
    messages.recv[12] = point_message(point_key(n.clone(), T::zero()), &root_raw);
    messages.recv_mult[12] = root_input.clone();
    messages.recv[13] = point_message(n.clone(), &root_identity);
    messages.recv_mult[13] = root_output.clone();

    messages.send[0] = point_message(aux, &leaf_identity);
    messages.send_mult[0] = leaf.clone();
    messages.send[1] =
        point_message(point_key(n.clone(), p.clone() + id.clone()), &leaf_point);
    messages.send_mult[1] = leaf.clone();
    for product_id in 0..6 {
        let product_id_value = T::from_canonical_usize(product_id);
        messages.send[2 + product_id] = operand_message(
            product_key(
                n.clone(),
                p.clone(),
                T::zero(),
                id.clone(),
                product_id_value.clone(),
            ),
            &first_pairs[product_id].0,
            &first_pairs[product_id].1,
            n.clone(),
            p.clone(),
            id.clone(),
            T::zero(),
            product_id_value,
            rebase_input.clone(),
        );
        messages.send_mult[2 + product_id] = node_input.clone() + rebase_input.clone();
    }
    messages.send[8] = reduced_message(
        product_message_key + T::one(),
        &product_reduced,
        n.clone(),
        p.clone(),
        id.clone(),
        product_stage,
        product_id,
        product_flow,
    );
    messages.send_mult[8] = product.clone();
    for product_id in 0..6 {
        let product_id_value = T::from_canonical_usize(product_id);
        messages.send[9 + product_id] = operand_message(
            product_key(
                n.clone(),
                p.clone(),
                T::one(),
                id.clone(),
                product_id_value.clone(),
            ),
            &second_pairs[product_id].0,
            &second_pairs[product_id].1,
            n.clone(),
            p.clone(),
            id.clone(),
            T::one(),
            product_id_value,
            c[3].clone(),
        );
        messages.send_mult[9 + product_id] = middle.clone();
    }
    let inv2 = T::from_canonical_u32((KoalaBear::ORDER_U32 + 1) / 2);
    let heap_id = p.clone() * T::from_canonical_u32(2) -
        cols.control_next_rank.clone() * inv2;
    messages.send[15] = point_message(point_key(n.clone(), heap_id), &raw_output);
    messages.send_mult[15] = node_output.clone();
    for product_id in 0..4 {
        let lhs = if product_id == 0 {
            root_lambda.clone()
        } else {
            root_raw[product_id - 1].clone()
        };
        let product_id_value = T::from_canonical_usize(product_id);
        messages.send[16 + product_id] = operand_message(
            product_key(
                n.clone(),
                p.clone(),
                T::from_canonical_u32(2),
                p.clone(),
                product_id_value.clone(),
            ),
            &lhs,
            &root_inverse,
            n.clone(),
            p.clone(),
            p.clone(),
            T::from_canonical_u32(2),
            product_id_value,
            T::from_canonical_u32(4) + c[3].clone(),
        );
        messages.send_mult[16 + product_id] = root_input.clone();
    }
    messages.send[20] = point_message(n.clone(), &root_point);
    messages.send_mult[20] = root_output.clone();
    messages.send[21] = point_message(T::zero(), &root_identity);
    messages.send_mult[21] = root_output.clone();

    let current_tag = node_input.clone() * T::from_canonical_u32(NODE_TAG) +
        rebase_input.clone() * T::from_canonical_u32(REBASE_TAG) +
        root_input.clone() * T::from_canonical_u32(ROOT_IN_TAG) +
        root_output.clone() * T::from_canonical_u32(ROOT_OUT_TAG);
    messages.control_recv =
        control_message(n.clone(), p.clone(), cols.control_rank.clone(), current_tag);
    messages.control_recv_mult =
        leaf.clone() + node_input + rebase_input + root_input.clone() + root_output.clone();
    messages.control_send = control_message(
        n,
        p,
        cols.control_next_rank.clone(),
        cols.control_next_tag.clone(),
    );
    messages.control_send_mult = leaf + node_output + root_input + root_output;
    messages
}

fn reducer_denominator<AB: FullAirBuilder>(
    builder: &AB,
    payload: &[AB::VarMaybeExt; PROJECTIVE_CHAIN_BASE_VALUES],
) -> AB::VarExt {
    let blocks: [AB::VarExt; PROJECTIVE_CHAIN_BLOCKS] = core::array::from_fn(|block| {
        let start = block * 5;
        let limbs: [AB::VarMaybeExt; PROJECTIVE_CHAIN_BLOCK_WIDTH] =
            core::array::from_fn(|limb| {
                payload.get(start + limb).cloned().unwrap_or_else(AB::zero_maybe)
            });
        AB::pack_ext_limbs(&limbs)
    });
    builder.lookup_denominator_ext_blocks(
        AB::VarMaybeExt::from(AB::F::from_canonical_usize(
            InteractionKind::GlobalProjectiveChainV2 as usize,
        )),
        blocks,
    )
}

fn reducer_active<T: AbstractField + Clone>(cols: &GlobalTileReducerCols<T>) -> T {
    cols.mode_leaf.clone() +
        cols.mode_node_input.clone() +
        cols.mode_product.clone() +
        cols.mode_node_middle.clone() +
        cols.mode_node_output.clone() +
        cols.mode_root_input.clone() +
        cols.mode_root_output.clone() +
        cols.selector_spare.clone()
}

fn constrain_local<T: AbstractField + Clone>(
    cols: &GlobalTileReducerCols<T>,
    mut emit: impl FnMut(T),
) {
    let one = T::one();
    let selectors = [
        cols.mode_leaf.clone(),
        cols.mode_node_input.clone(),
        cols.mode_product.clone(),
        cols.mode_node_middle.clone(),
        cols.mode_node_output.clone(),
        cols.mode_root_input.clone(),
        cols.mode_root_output.clone(),
        cols.selector_spare.clone(),
    ];
    for selector in selectors {
        emit(selector.clone() * (one.clone() - selector));
    }
    let active = reducer_active(cols);
    emit(active.clone() * (one.clone() - active));

    let c = &cols.payload.control;
    let v = &cols.payload.values;
    let leaf = cols.mode_leaf.clone();
    let leaf_real = c[4].clone();
    let leaf_last = c[5].clone();
    let leaf_end = v[REDUCER_LEAF_END_VALUE].clone();
    let k = v[REDUCER_LEAF_K_VALUE].clone();
    emit(leaf.clone() * leaf_real.clone() * (one.clone() - leaf_real.clone()));
    emit(leaf.clone() * leaf_last.clone() * (one.clone() - leaf_last.clone()));
    emit(leaf.clone() * leaf_last.clone() * (one.clone() - leaf_real.clone()));
    emit(leaf.clone() * leaf_end.clone() * (one.clone() - leaf_end.clone()));
    emit(leaf.clone() * leaf_end.clone() * (c[2].clone() + one.clone() - c[1].clone()));
    emit(leaf.clone() * leaf_end.clone() * (leaf_real.clone() - leaf_last.clone()));
    emit(leaf.clone() * leaf_last.clone() * (c[2].clone() + one.clone() - k.clone()));
    emit(
        leaf.clone() *
            (leaf_real.clone() - leaf_last.clone()) *
            (c[3].clone() -
                (c[2].clone() + one.clone()) * T::from_canonical_u32(512)),
    );
    emit(leaf.clone() * leaf_last.clone() * (c[3].clone() - c[0].clone()));
    emit(
        leaf.clone() *
            (one.clone() - leaf_real.clone()) *
            (c[3].clone() - c[0].clone()),
    );

    let dummy = leaf.clone() * (one.clone() - leaf_real.clone());
    let identity = identity::<T>();
    for coordinate in 0..3 {
        for limb in 0..11 {
            emit(
                dummy.clone() *
                    (v[coordinate * 11 + limb].clone() - identity[coordinate][limb].clone()),
            );
        }
    }

    let mut p_bit_sum = T::zero();
    let mut p_value = T::zero();
    let mut p_half = T::zero();
    for bit in 0..REDUCER_LEAF_P_BITS {
        let value = v[REDUCER_LEAF_P_BITS_START + bit].clone();
        emit(leaf.clone() * value.clone() * (one.clone() - value.clone()));
        emit(leaf.clone() * (one.clone() - leaf_end.clone()) * value.clone());
        p_bit_sum = p_bit_sum + value.clone();
        p_value = p_value + value.clone() * T::from_canonical_usize(1usize << bit);
        if bit > 0 {
            p_half = p_half + value * T::from_canonical_usize(1usize << (bit - 1));
        }
    }
    emit(leaf.clone() * (p_bit_sum - leaf_end.clone()));
    emit(leaf.clone() * leaf_end.clone() * (c[1].clone() - p_value));

    let mut gap = T::zero();
    for bit in 0..REDUCER_LEAF_GAP_BITS {
        let value = v[REDUCER_LEAF_GAP_BITS_START + bit].clone();
        emit(leaf.clone() * value.clone() * (one.clone() - value.clone()));
        emit(leaf.clone() * (one.clone() - leaf_end.clone()) * value.clone());
        gap = gap + value * T::from_canonical_usize(1usize << bit);
    }
    emit(leaf.clone() * leaf_end.clone() * (k - p_half - one.clone() - gap));
    for value in &v[REDUCER_LEAF_GAP_BITS_START + REDUCER_LEAF_GAP_BITS..] {
        emit(leaf.clone() * value.clone());
    }

    let node_input = cols.mode_node_input.clone();
    for control in &c[3..] {
        emit(node_input.clone() * control.clone());
    }
    let rebase_input = cols.selector_spare.clone();
    emit(rebase_input.clone() * (c[2].clone() - c[1].clone()));
    for control in &c[3..] {
        emit(rebase_input.clone() * control.clone());
    }
    let middle = cols.mode_node_middle.clone();
    let middle_rebase = c[3].clone();
    emit(middle.clone() * middle_rebase.clone() * (one.clone() - middle_rebase));
    for control in &c[4..] {
        emit(middle.clone() * control.clone());
    }

    let node_output = cols.mode_node_output.clone();
    let rebase = c[3].clone();
    let last_node = c[4].clone();
    emit(node_output.clone() * rebase.clone() * (one.clone() - rebase.clone()));
    emit(node_output.clone() * last_node.clone() * (one.clone() - last_node.clone()));
    emit(node_output.clone() * rebase.clone() * last_node.clone());
    emit(node_output.clone() * rebase.clone() * (c[2].clone() - c[1].clone()));
    emit(node_output.clone() * last_node.clone() * (c[2].clone() - one.clone()));
    emit(node_output.clone() * c[5].clone());

    let product = cols.mode_product.clone();
    let product_root = c[3].clone();
    let product_last = c[5].clone();
    let product_wave = v[REDUCER_PRODUCT_WAVE_VALUE].clone();
    let product_infinity = v[REDUCER_PRODUCT_INFINITY_VALUE].clone();
    let product_rebase = v[REDUCER_PRODUCT_REBASE_VALUE].clone();
    let product_continue = v[REDUCER_PRODUCT_CONTINUE_VALUE].clone();
    let product_to_middle = v[REDUCER_PRODUCT_TO_MIDDLE_VALUE].clone();
    let product_to_node = v[REDUCER_PRODUCT_TO_NODE_VALUE].clone();
    let product_to_root = v[REDUCER_PRODUCT_TO_ROOT_VALUE].clone();
    for value in [
        product_root.clone(),
        product_last.clone(),
        product_wave.clone(),
        product_infinity.clone(),
        product_rebase.clone(),
        product_continue.clone(),
        product_to_middle.clone(),
        product_to_node.clone(),
        product_to_root.clone(),
    ] {
        emit(product.clone() * value.clone() * (one.clone() - value));
    }
    emit(product.clone() * product_root.clone() * product_wave.clone());
    emit(product.clone() * (one.clone() - product_root.clone()) * product_infinity);
    emit(product.clone() * product_root.clone() * product_rebase.clone());
    emit(
        product.clone() *
            (product_continue.clone() - (one.clone() - product_last.clone())),
    );
    emit(
        product.clone() *
            (product_to_node.clone() - product_last.clone() * product_wave.clone()),
    );
    emit(
        product.clone() *
            (product_to_root.clone() - product_last.clone() * product_root.clone()),
    );
    emit(
        product.clone() *
            (product_to_middle - one.clone() + product_continue + product_to_node +
                product_to_root),
    );
    emit(
        product.clone() *
            product_last *
            (c[4].clone() + product_root * T::from_canonical_u32(2) -
                T::from_canonical_u32(5)),
    );
    for value in &v[46..62] {
        emit(product.clone() * value.clone());
    }

    let root_input = cols.mode_root_input.clone();
    let root_infinity = c[3].clone();
    emit(root_input.clone() * (c[2].clone() - c[1].clone()));
    emit(root_input.clone() * root_infinity.clone() * (one.clone() - root_infinity.clone()));
    emit(root_input.clone() * c[4].clone());
    emit(root_input.clone() * c[5].clone());
    let finite = one.clone() - root_infinity.clone();
    for limb in 0..11 {
        let expected = finite.clone() * v[22 + limb].clone() +
            root_infinity.clone() * v[11 + limb].clone();
        emit(root_input.clone() * (v[33 + limb].clone() - expected));
    }
    for value in &v[55..] {
        emit(root_input.clone() * value.clone());
    }

    let root_output = cols.mode_root_output.clone();
    emit(root_output.clone() * (c[2].clone() - c[1].clone()));
    emit(root_output.clone() * root_infinity.clone() * (one.clone() - root_infinity.clone()));
    emit(root_output.clone() * c[4].clone());
    emit(root_output.clone() * c[5].clone());
    for limb in 0..11 {
        emit(
            root_output.clone() *
                (v[limb].clone() - if limb == 0 { one.clone() } else { T::zero() }),
        );
        emit(
            root_output.clone() *
                (v[33 + limb].clone() - if limb == 0 { finite.clone() } else { T::zero() }),
        );
        emit(root_output.clone() * root_infinity.clone() * v[11 + limb].clone());
        emit(
            root_output.clone() *
                root_infinity.clone() *
                (v[22 + limb].clone() - if limb == 0 { one.clone() } else { T::zero() }),
        );
    }
    for value in &v[44..] {
        emit(root_output.clone() * value.clone());
    }

    let two = T::from_canonical_u32(2);
    let four = T::from_canonical_u32(4);
    emit(
        cols.control_rank.clone() -
            (leaf.clone() * (c[2].clone() * two.clone() + leaf_real.clone()) +
                node_input.clone() *
                    ((c[1].clone() * two.clone() - one.clone() - c[2].clone()) * two.clone()) +
                rebase_input.clone() * (c[1].clone() * four.clone() - two.clone()) +
                root_input.clone() * c[1].clone() * four.clone() +
                root_output.clone() *
                    (c[1].clone() * four.clone() + two.clone())),
    );
    emit(
        cols.control_next_rank.clone() -
            (leaf.clone() *
                    ((c[2].clone() + one.clone()) * two.clone() + leaf_real - leaf_last) +
                node_output.clone() *
                    (c[1].clone() * two.clone() -
                        c[2].clone() * (one.clone() - rebase.clone())) *
                    two.clone() +
                root_input.clone() *
                    (c[1].clone() * four.clone() + two.clone()) +
                root_output.clone()),
    );
    emit(
        cols.control_next_tag.clone() -
            (leaf *
                    four.clone() *
                    leaf_end *
                    (one.clone() + v[REDUCER_LEAF_P_BITS_START].clone()) +
                node_output *
                    (four.clone() +
                        last_node * four.clone() +
                        rebase * T::from_canonical_u32(8)) +
                root_input * T::from_canonical_u32(ROOT_OUT_TAG)),
    );
}

#[cfg(test)]
#[allow(dead_code)]
fn constrain_schedule<T: AbstractField + Clone>(
    local: &GlobalTileReducerCols<T>,
    next: &GlobalTileReducerCols<T>,
    is_first: T,
    is_last: T,
    is_transition: T,
    mut emit: impl FnMut(T),
) {
    let one = T::one();
    let active = reducer_active(local);
    let next_active = reducer_active(next);
    let c = &local.payload.control;
    let next_c = &next.payload.control;
    let v = &local.payload.values;
    let next_v = &next.payload.values;

    emit(is_first.clone() * (local.mode_leaf.clone() - one.clone()));
    emit(is_first.clone() * c[2].clone());
    emit(is_first * (c[4].clone() - one.clone()));
    emit(is_last * active.clone());

    emit(is_transition.clone() * (one.clone() - active.clone()) * next_active.clone());
    emit(is_transition.clone() * local.mode_root_output.clone() * next_active.clone());
    let continues = active - local.mode_root_output.clone();
    emit(is_transition.clone() * continues.clone() * (one.clone() - next_active));
    emit(is_transition.clone() * continues.clone() * (next_c[0].clone() - c[0].clone()));
    emit(is_transition.clone() * continues * (next_c[1].clone() - c[1].clone()));

    let leaf = local.mode_leaf.clone();
    let leaf_real = c[4].clone();
    let leaf_last = c[5].clone();
    let leaf_end = v[REDUCER_LEAF_END_VALUE].clone();
    emit(
        is_transition.clone() *
            leaf.clone() *
            (next.mode_leaf.clone() +
                next.mode_node_input.clone() +
                next.selector_spare.clone() -
                one.clone()),
    );
    emit(
        is_transition.clone() *
            leaf.clone() *
            (leaf_end.clone() - next.mode_node_input.clone() - next.selector_spare.clone()),
    );
    emit(
        is_transition.clone() *
            next.mode_leaf.clone() *
            (next_c[2].clone() - c[2].clone() - one.clone()),
    );
    emit(
        is_transition.clone() *
            next.mode_leaf.clone() *
            (next_v[REDUCER_LEAF_K_VALUE].clone() - v[REDUCER_LEAF_K_VALUE].clone()),
    );
    emit(
        is_transition.clone() *
            next.mode_leaf.clone() *
            (next_c[4].clone() - leaf_real + leaf_last),
    );
    let p_is_one = v[REDUCER_LEAF_P_BITS_START].clone();
    let leaf_exit =
        leaf.clone() * (next.mode_node_input.clone() + next.selector_spare.clone());
    emit(
        is_transition.clone() *
            leaf_exit.clone() *
            (next.selector_spare.clone() - p_is_one.clone()),
    );
    emit(
        is_transition.clone() *
            leaf_exit.clone() *
            (next.mode_node_input.clone() - (one.clone() - p_is_one.clone())),
    );
    emit(
        is_transition.clone() *
            leaf_exit *
            (next_c[2].clone() - (c[1].clone() - one.clone() + p_is_one)),
    );

    let add_input = local.mode_node_input.clone() + local.selector_spare.clone();
    emit(is_transition.clone() * add_input.clone() * (next.mode_product.clone() - one.clone()));
    emit(is_transition.clone() * add_input.clone() * (next_c[2].clone() - c[2].clone()));
    emit(is_transition.clone() * add_input.clone() * next_c[3].clone());
    emit(is_transition.clone() * add_input.clone() * next_c[4].clone());
    emit(is_transition.clone() * add_input.clone() * next_v[REDUCER_PRODUCT_WAVE_VALUE].clone());
    emit(is_transition.clone() * add_input.clone() * next_v[REDUCER_PRODUCT_INFINITY_VALUE].clone());
    emit(
        is_transition.clone() *
            add_input *
            (next_v[REDUCER_PRODUCT_REBASE_VALUE].clone() - local.selector_spare.clone()),
    );

    let middle = local.mode_node_middle.clone();
    emit(is_transition.clone() * middle.clone() * (next.mode_product.clone() - one.clone()));
    emit(is_transition.clone() * middle.clone() * (next_c[2].clone() - c[2].clone()));
    emit(is_transition.clone() * middle.clone() * next_c[3].clone());
    emit(is_transition.clone() * middle.clone() * next_c[4].clone());
    emit(
        is_transition.clone() *
            middle.clone() *
            (next_v[REDUCER_PRODUCT_WAVE_VALUE].clone() - one.clone()),
    );
    emit(is_transition.clone() * middle.clone() * next_v[REDUCER_PRODUCT_INFINITY_VALUE].clone());
    emit(
        is_transition.clone() *
            middle *
            (next_v[REDUCER_PRODUCT_REBASE_VALUE].clone() - c[3].clone()),
    );

    let root_input = local.mode_root_input.clone();
    emit(is_transition.clone() * root_input.clone() * (next.mode_product.clone() - one.clone()));
    emit(is_transition.clone() * root_input.clone() * (next_c[2].clone() - c[1].clone()));
    emit(is_transition.clone() * root_input.clone() * (next_c[3].clone() - one.clone()));
    emit(is_transition.clone() * root_input.clone() * next_c[4].clone());
    emit(is_transition.clone() * root_input.clone() * next_v[REDUCER_PRODUCT_WAVE_VALUE].clone());
    emit(
        is_transition.clone() *
            root_input.clone() *
            (next_v[REDUCER_PRODUCT_INFINITY_VALUE].clone() - c[3].clone()),
    );
    emit(is_transition.clone() * root_input * next_v[REDUCER_PRODUCT_REBASE_VALUE].clone());

    let product_continue =
        local.mode_product.clone() * v[REDUCER_PRODUCT_CONTINUE_VALUE].clone();
    let product_to_middle =
        local.mode_product.clone() * v[REDUCER_PRODUCT_TO_MIDDLE_VALUE].clone();
    let product_to_node =
        local.mode_product.clone() * v[REDUCER_PRODUCT_TO_NODE_VALUE].clone();
    let product_to_root =
        local.mode_product.clone() * v[REDUCER_PRODUCT_TO_ROOT_VALUE].clone();
    emit(
        is_transition.clone() *
            (next.mode_product.clone() -
                local.mode_node_input.clone() -
                local.selector_spare.clone() -
                local.mode_node_middle.clone() -
                local.mode_root_input.clone() -
                product_continue.clone()),
    );
    emit(
        is_transition.clone() *
            (next.mode_node_middle.clone() - product_to_middle.clone()),
    );
    emit(
        is_transition.clone() *
            (next.mode_node_output.clone() - product_to_node.clone()),
    );
    emit(
        is_transition.clone() *
            (next.mode_root_output.clone() - product_to_root.clone()),
    );
    emit(
        is_transition.clone() *
            product_continue.clone() *
            (next_c[2].clone() - c[2].clone()),
    );
    emit(
        is_transition.clone() *
            product_continue.clone() *
            (next_c[3].clone() - c[3].clone()),
    );
    emit(
        is_transition.clone() *
            product_continue.clone() *
            (next_c[4].clone() - c[4].clone() - one.clone()),
    );
    emit(
        is_transition.clone() *
            product_continue.clone() *
            (next_v[REDUCER_PRODUCT_WAVE_VALUE].clone() -
                v[REDUCER_PRODUCT_WAVE_VALUE].clone()),
    );
    emit(
        is_transition.clone() *
            product_continue.clone() *
            (next_v[REDUCER_PRODUCT_INFINITY_VALUE].clone() -
                v[REDUCER_PRODUCT_INFINITY_VALUE].clone()),
    );
    emit(
        is_transition.clone() *
            product_continue *
            (next_v[REDUCER_PRODUCT_REBASE_VALUE].clone() -
                v[REDUCER_PRODUCT_REBASE_VALUE].clone()),
    );
    emit(
        is_transition.clone() *
            next.mode_node_middle.clone() *
            (next_c[2].clone() - c[2].clone()),
    );
    emit(
        is_transition.clone() *
            next.mode_node_middle.clone() *
            (next_c[3].clone() - v[REDUCER_PRODUCT_REBASE_VALUE].clone()),
    );
    emit(
        is_transition.clone() *
            next.mode_node_output.clone() *
            (next_c[2].clone() - c[2].clone()),
    );
    emit(
        is_transition.clone() *
            next.mode_node_output.clone() *
            (next_c[3].clone() - v[REDUCER_PRODUCT_REBASE_VALUE].clone()),
    );
    emit(
        is_transition.clone() *
            next.mode_root_output.clone() *
            (next_c[2].clone() - c[2].clone()),
    );
    emit(
        is_transition.clone() *
            next.mode_root_output.clone() *
            (next_c[3].clone() - v[REDUCER_PRODUCT_INFINITY_VALUE].clone()),
    );

    let node_output = local.mode_node_output.clone();
    let rebase = c[3].clone();
    let last_node = c[4].clone();
    emit(
        is_transition.clone() *
            node_output.clone() *
            (next.mode_root_input.clone() - rebase.clone()),
    );
    emit(
        is_transition.clone() *
            node_output.clone() *
            (next.selector_spare.clone() - last_node.clone()),
    );
    emit(
        is_transition.clone() *
            node_output.clone() *
            (next.mode_node_input.clone() - (one.clone() - rebase.clone() - last_node.clone())),
    );
    emit(
        is_transition.clone() *
            next.mode_root_input.clone() *
            (next_c[2].clone() - c[1].clone()),
    );
    emit(
        is_transition.clone() *
            next.selector_spare.clone() *
            (next_c[2].clone() - c[1].clone()),
    );
    emit(
        is_transition *
            next.mode_node_input.clone() *
            (next_c[2].clone() - c[2].clone() + node_output),
    );
}

impl<F: Field> BaseAir<F> for GlobalTileReducerChip {
    fn width(&self) -> usize {
        NUM_GLOBAL_TILE_REDUCER_COLS
    }
}

impl<F: Field> MachineAir<F> for GlobalTileReducerChip {
    type Record = ExecutionRecord;
    type Program = Program;

    fn name(&self) -> String {
        "GlobalTileReducerV3".to_string()
    }

    fn generate_trace(&self, input: &Self::Record, _: &mut Self::Record) -> CompressedMatrix<F> {
        assert_eq!(TypeId::of::<F>(), TypeId::of::<KoalaBear>());
        let trace = input
            .take_global_tile_reducer_trace_artifact()
            .expect("canonical GlobalTileReducerV3 trace artifact missing");
        let erased: Box<dyn Any> = Box::new(trace);
        *erased.downcast::<CompressedMatrix<F>>().expect("Global reducer field identity checked")
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        if !<Self as MachineAir<F>>::included(self, input) || input.has_global_trace_artifact() {
            return;
        }
        let prepared =
            super::writer::prepare_global_trace(input).expect("Global trace preparation failed");
        let retained = prepared.consume_byte_delta(output);
        output.install_global_trace_artifact(retained.trace, retained.reducer_trace);
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        if TypeId::of::<F>() != TypeId::of::<KoalaBear>() {
            return None;
        }
        Some(
            global_tile_reducer_padded_rows(crate::global::sources::global_endpoint_count(input))
                .expect("GlobalTileReducerV3 height exceeds h22"),
        )
    }

    fn included(&self, input: &Self::Record) -> bool {
        TypeId::of::<F>() == TypeId::of::<KoalaBear>() &&
            crate::global::sources::global_endpoint_count(input) > 0
    }

    fn commit_scope(&self) -> InteractionScope {
        InteractionScope::Global
    }

    fn local_only(&self) -> bool {
        true
    }

    fn global_boundary_owner(&self) -> Option<dt_stark::global_d11::StableChipId> {
        Some(dt_stark::global_d11::CORE_GLOBAL_OWNER)
    }

    fn extract_global_claim(
        &self,
        trace: &CompressedMatrix<F>,
    ) -> Result<Option<dt_stark::air::GlobalClaim<F>>, String> {
        super::boundary::claim_from_compressed_tile_reducer_trace(trace)
            .map_err(|error| format!("{error:?}"))
    }
}

impl<AB> Air<AB> for GlobalTileReducerChip
where
    AB: DTAirBuilder + PairBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local_row = main.row_slice(0);
        let local_values: [AB::Expr; NUM_GLOBAL_TILE_REDUCER_COLS] =
            core::array::from_fn(|index| local_row[index].into());
        let local: &GlobalTileReducerCols<AB::Expr> = local_values.as_slice().borrow();
        constrain_local(local, |residual| builder.assert_zero(residual));
        let messages = reducer_messages(local);
        let kind = InteractionKind::GlobalProjectiveChainV2;
        for slot in 0..RECV_CLASSES {
            builder.receive(
                AirInteraction::new_extension_blocks(
                    messages.recv[slot].to_vec(),
                    PROJECTIVE_CHAIN_BLOCK_WIDTH,
                    messages.recv_mult[slot].clone(),
                    kind,
                ),
                InteractionScope::Local,
            );
        }
        for slot in 0..SEND_CLASSES {
            builder.send(
                AirInteraction::new_extension_blocks(
                    messages.send[slot].to_vec(),
                    PROJECTIVE_CHAIN_BLOCK_WIDTH,
                    messages.send_mult[slot].clone(),
                    kind,
                ),
                InteractionScope::Local,
            );
        }
        builder.receive(
            AirInteraction::new_extension_blocks(
                messages.control_recv.to_vec(),
                PROJECTIVE_CHAIN_BLOCK_WIDTH,
                messages.control_recv_mult,
                kind,
            ),
            InteractionScope::Local,
        );
        builder.send(
            AirInteraction::new_extension_blocks(
                messages.control_send.to_vec(),
                PROJECTIVE_CHAIN_BLOCK_WIDTH,
                messages.control_send_mult,
                kind,
            ),
            InteractionScope::Local,
        );
    }
}

impl<AB: FullAirBuilder> FullAir<AB> for GlobalTileReducerChip
where
    AB::VarMaybeExt: Clone,
{
    fn width(&self) -> usize {
        NUM_GLOBAL_TILE_REDUCER_COLS
    }

    fn required_max_beta_power(&self) -> usize {
        GLOBAL_TILE_REDUCER_MAX_BETA_POWER
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        (0..NUM_GLOBAL_TILE_REDUCER_COLS).map(PairCol::Main).collect()
    }

    fn precompute_lc(&self, builder: &mut AB) {
        let main: [AB::VarMaybeExt; NUM_GLOBAL_TILE_REDUCER_COLS] = {
            let main = builder.main();
            core::array::from_fn(|index| main[index].clone())
        };
        let local: &GlobalTileReducerCols<AB::VarMaybeExt> = main.as_slice().borrow();
        let messages = reducer_messages(local);
        for payload in &messages.recv {
            builder.retain_precomputed(reducer_denominator(builder, payload));
        }
        for payload in &messages.send {
            builder.retain_precomputed(reducer_denominator(builder, payload));
        }
        builder.retain_precomputed(reducer_denominator(builder, &messages.control_recv));
        builder.retain_precomputed(reducer_denominator(builder, &messages.control_send));
        let beta = builder.beta_powers().to_vec();
        for values in [
            &local.payload.values[0..11],
            &local.payload.values[11..22],
            &local.payload.values[22..33],
            &local.payload.values[33..43],
        ] {
            let mut projection = beta[0].clone() * values[0].clone();
            for (power, value) in values.iter().enumerate().skip(1) {
                projection = projection + beta[power].clone() * value.clone();
            }
            builder.retain_precomputed(projection);
        }
    }

    fn eval(&self, builder: &mut AB) {
        let reserved = builder.reserved_poly();
        let local_row = reserved.row_slice(0);
        let local: &GlobalTileReducerCols<AB::VarMaybeExt> = local_row.deref().borrow();
        constrain_local(local, |residual| builder.assert_zero(residual));
        let precomputed = builder.precomputed();
        let precomputed = precomputed.row_slice(0);
        let two = AB::pack_ext_limbs(&[AB::VarMaybeExt::from(
            AB::F::from_canonical_u32(2),
        )]);
        let f_beta = builder.beta_powers()[11].clone() -
            builder.beta_powers()[3].clone() -
            two;
        let relation = precomputed[38].clone() * precomputed[39].clone() -
            precomputed[40].clone() -
            precomputed[41].clone() * f_beta;
        builder.assert_zero_ext(relation * local.mode_product.clone());
    }

    fn lookup(&self, builder: &mut AB) {
        let reserved = builder.reserved_poly();
        let local_row = reserved.row_slice(0);
        let local: &GlobalTileReducerCols<AB::VarMaybeExt> = local_row.deref().borrow();
        let messages = reducer_messages(local);
        for multiplicity in messages.recv_mult {
            builder.recv(multiplicity);
        }
        for multiplicity in messages.send_mult {
            builder.send(multiplicity);
        }
        builder.recv(messages.control_recv_mult);
        builder.send(messages.control_send_mult);
    }
}

const _: () = {
    assert!(NUM_GLOBAL_TILE_REDUCER_COLS == 83);
    assert!(GLOBAL_TILE_REDUCER_LOOKUP_SLOTS == RECV_CLASSES + SEND_CLASSES + 2);
    assert!(GLOBAL_TILE_REDUCER_LOOKUP_BATCH_SIZE == 2);
    assert!(GLOBAL_TILE_REDUCER_PERMUTATION_WIDTH == 19);
};

#[cfg(feature = "test-utils")]
pub(crate) mod p8_kats {
    use super::*;

    use dt_stark::global_d11::{
        canonicalize_projective_v2, construct_map, D11ProjectivePointV1, GlobalPackInputV1, D11,
    };
    fn evaluate<const N: usize>(coefficients: &[KoalaBear; N], beta: KoalaBear) -> KoalaBear {
        coefficients.iter().rev().fold(KoalaBear::zero(), |value, coefficient| {
            value * beta + *coefficient
        })
    }

    pub(crate) fn tile_reducer_fixed_tree_and_product_beta() {
        assert_eq!(NUM_GLOBAL_TILE_REDUCER_COLS, 83);
        assert_eq!(GLOBAL_TILE_REDUCER_LOOKUP_SLOTS, 38);
        assert_eq!(GLOBAL_TILE_REDUCER_LOOKUP_BATCH_SIZE, 2);
        assert_eq!(GLOBAL_TILE_REDUCER_PERMUTATION_WIDTH, 19);
        for (k, p, rows) in [(1usize, 1usize, 22usize), (2, 2, 38), (3, 4, 70), (5, 8, 134)] {
            assert_eq!(p, k.next_power_of_two());
            assert_eq!(rows, p + 15 * (p - 1) + 15 + 6);
            assert_eq!(rows, 16 * p + 6);
        }

        let mapped = construct_map::<KoalaBear>(
            GlobalPackInputV1 { message: [3, 5, 8, 13, 21, 34, 55], kind: 9 },
            false,
        )
        .unwrap();
        let point = D11ProjectivePointV1 {
            x: mapped.packed_x,
            y: mapped.signed_y,
            z: D11::one(),
        };
        let identity = D11ProjectivePointV1::identity();
        let (finite, _, _) = identity.add_complete_with_products(&point);
        assert_eq!(canonicalize_projective_v2(finite).unwrap(), point);
        let (doubled, _, _) = point.add_complete_with_products(&point);
        assert!(!doubled.z.is_zero());
        let negated = D11ProjectivePointV1 { x: point.x, y: -point.y, z: point.z };
        let (cancelled, _, _) = point.add_complete_with_products(&negated);
        assert_eq!(canonicalize_projective_v2(cancelled).unwrap(), identity);
        let (infinity, _, _) = identity.add_complete_with_products(&identity);
        assert_eq!(canonicalize_projective_v2(infinity).unwrap(), identity);

        let lhs = core::array::from_fn(|i| KoalaBear::from_canonical_usize(i + 1));
        let rhs = core::array::from_fn(|i| KoalaBear::from_canonical_usize(2 * i + 3));
        let witnessed = super::super::constraints::mul_with_quotient(&lhs, &rhs);
        let beta = KoalaBear::from_canonical_u32(17);
        let f_beta = beta.exp_u64(11) - beta.exp_u64(3) - KoalaBear::from_canonical_u32(2);
        assert_eq!(
            evaluate(&lhs, beta) * evaluate(&rhs, beta),
            evaluate(&witnessed.reduced, beta) + evaluate(&witnessed.quotient, beta) * f_beta,
        );

        let n = 1usize << 22;
        let p = 8192usize;
        let public_max = n;
        let internal_min = n + 1;
        let internal_max = n + 32 * p + 8;
        assert!(public_max < internal_min);
        assert!(internal_max < KoalaBear::ORDER_U32 as usize);
        assert!(SCHEDULE_DOMAIN as usize > internal_max);
        assert!(SCHEDULE_DOMAIN as usize + 4 * p + 2 < KoalaBear::ORDER_U32 as usize);
    }

    fn reducer_constraints_accept(trace: &CompressedMatrix<KoalaBear>) -> bool {
        crate::check_constraints::check_trace(
            &GlobalTileReducerChip,
            &trace.decompress(),
            &[],
            0,
        )
        .is_empty()
    }

    fn control_residuals(
        trace: &CompressedMatrix<KoalaBear>,
    ) -> std::collections::BTreeMap<[u32; PROJECTIVE_CHAIN_BASE_VALUES], i64> {
        let mut residuals = std::collections::BTreeMap::new();
        for row in trace.main.values.chunks_exact(NUM_GLOBAL_TILE_REDUCER_COLS) {
            let cols: &GlobalTileReducerCols<KoalaBear> = row.borrow();
            let messages = reducer_messages(cols);
            for (payload, multiplicity, sign) in [
                (&messages.control_recv, messages.control_recv_mult, -1i64),
                (&messages.control_send, messages.control_send_mult, 1i64),
            ] {
                let multiplicity = i64::from(multiplicity.as_canonical_u32());
                if multiplicity == 0 {
                    continue;
                }
                let key = payload.map(|value| value.as_canonical_u32());
                *residuals.entry(key).or_default() += sign * multiplicity;
            }
        }
        residuals.retain(|_, multiplicity| *multiplicity != 0);
        residuals
    }

    pub(crate) fn tile_reducer_rejects_malicious_schedule() {
        let mapped = construct_map::<KoalaBear>(
            GlobalPackInputV1 { message: [1, 2, 3, 4, 5, 6, 7], kind: 8 },
            false,
        )
        .unwrap();
        let point = D11ProjectivePointV1 {
            x: mapped.packed_x,
            y: mapped.signed_y,
            z: D11::one(),
        };
        let identity = D11ProjectivePointV1::identity();
        let (mut trace, _) = super::super::writer::build_tile_reducer_trace(
            1025,
            &[point, point, point],
            identity,
        )
        .unwrap();
        assert!(reducer_constraints_accept(&trace));
        assert!(control_residuals(&trace).is_empty());
        let first_node_row = 4;
        let node_ordinal = super::super::columns::GLOBAL_TILE_REDUCER_COL_MAP.payload.control[2];
        trace.main.values[first_node_row * NUM_GLOBAL_TILE_REDUCER_COLS + node_ordinal] =
            KoalaBear::from_canonical_u32(2);
        let control_rank = super::super::columns::GLOBAL_TILE_REDUCER_COL_MAP.control_rank;
        trace.main.values[first_node_row * NUM_GLOBAL_TILE_REDUCER_COLS + control_rank] =
            KoalaBear::from_canonical_u32(10);
        assert!(reducer_constraints_accept(&trace));
        assert!(!control_residuals(&trace).is_empty());
    }
}
