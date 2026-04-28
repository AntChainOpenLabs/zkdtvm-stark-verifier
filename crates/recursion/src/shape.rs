#![allow(clippy::never_loop)]

pub const NUM_SKIP_ROUNDS: usize = 1;
pub const CHIP_LOG_HEIGHT_THRESHOLD: usize = 0;
pub const CHIP_HEIGHT_THRESHOLD: usize = 1 << CHIP_LOG_HEIGHT_THRESHOLD;

use std::marker::PhantomData;

use hashbrown::HashMap;

use dt_stark::{air::MachineAir, shape::OrderedShape};
use itertools::Itertools;
use p3_field::{extension::BinomiallyExtendable, PrimeField32};
use serde::{Deserialize, Serialize};

use crate::{
    chips::{
        alu_base::BaseAluChip,
        alu_ext::ExtAluChip,
        ext_exp_reverse_bits::ExtExpReverseBitsChip,
        mem::{MemoryConstChip, MemoryVarChip},
        prefix_sum_checks::PrefixSumChecksChip,
        public_values::{PublicValuesChip, PUB_VALUES_LOG_HEIGHT},
        select::SelectChip,
    },
    machine::RecursionAir,
    RecursionProgram, D,
};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RecursionShape {
    pub(crate) inner: HashMap<String, usize>,
}

impl RecursionShape {
    pub fn clone_into_hash_map(&self) -> HashMap<String, usize> {
        self.inner.clone()
    }
}

impl From<HashMap<String, usize>> for RecursionShape {
    fn from(value: HashMap<String, usize>) -> Self {
        Self { inner: value }
    }
}

pub struct RecursionShapeConfig<F, A> {
    allowed_shapes: Vec<HashMap<String, usize>>,
    _marker: PhantomData<(F, A)>,
}

impl<F: PrimeField32 + BinomiallyExtendable<D>, const DEGREE: usize>
    RecursionShapeConfig<F, RecursionAir<F, DEGREE>>
{
    pub fn fix_shape(&self, program: &mut RecursionProgram<F>) {
        let heights = RecursionAir::<F, DEGREE>::heights(program);

        let mut closest_shape = None;

        for shape in self.allowed_shapes.iter() {
            let mut valid = true;
            for (name, height) in heights.iter() {
                if *height > (1 << shape.get(name).unwrap()) {
                    valid = false;
                }
            }

            if !valid {
                continue;
            }

            closest_shape = Some(shape.clone());
            break;
        }

        if let Some(shape) = closest_shape {
            let shape = RecursionShape { inner: shape };
            *program.shape_mut() = Some(shape);
        } else {
            panic!("no shape found for heights: {heights:?}");
        }
    }

    pub fn get_all_shape_combinations(
        &self,
        batch_size: usize,
    ) -> impl Iterator<Item = Vec<OrderedShape>> + '_ {
        (0..batch_size)
            .map(|_| {
                self.allowed_shapes
                    .iter()
                    .cloned()
                    .map(|map| map.into_iter().collect::<OrderedShape>())
            })
            .multi_cartesian_product()
    }

    pub fn union_config_with_extra_room(&self) -> Self {
        let mut map = HashMap::new();
        for shape in self.allowed_shapes.clone() {
            for key in shape.keys() {
                let current = map.get(key).unwrap_or(&0);
                map.insert(key.clone(), *current.max(shape.get(key).unwrap()));
            }
        }
        map.values_mut().for_each(|x| *x += 2);
        map.insert("PublicValues".to_string(), 4);
        Self { allowed_shapes: vec![map], _marker: PhantomData }
    }

    pub fn from_hash_map(hash_map: &HashMap<String, usize>) -> Self {
        Self { allowed_shapes: vec![hash_map.clone()], _marker: PhantomData }
    }

    pub fn first(&self) -> Option<&HashMap<String, usize>> {
        self.allowed_shapes.first()
    }
}

impl<F: PrimeField32 + BinomiallyExtendable<D>, const DEGREE: usize> Default
    for RecursionShapeConfig<F, RecursionAir<F, DEGREE>>
{
    fn default() -> Self {
        let mem_const = RecursionAir::<F, DEGREE>::MemoryConst(MemoryConstChip::default()).name();
        let mem_var = RecursionAir::<F, DEGREE>::MemoryVar(MemoryVarChip::default()).name();
        let base_alu = RecursionAir::<F, DEGREE>::BaseAlu(BaseAluChip).name();
        let ext_alu = RecursionAir::<F, DEGREE>::ExtAlu(ExtAluChip).name();
        let poseidon2_wide = RecursionAir::<F, DEGREE>::poseidon2_wide_chip().name();
        let select = RecursionAir::<F, DEGREE>::Select(SelectChip).name();
        let public_values = RecursionAir::<F, DEGREE>::PublicValues(PublicValuesChip).name();
        // let sumcheck_round = RecursionAir::<F, DEGREE>::SumcheckRound(SumcheckRoundChip).name();
        let prefix_sum_checks =
            RecursionAir::<F, DEGREE>::PrefixSumChecks(PrefixSumChecksChip).name();
        let ext_exp_reverse_bits =
            RecursionAir::<F, DEGREE>::ExtExpReverseBits(ExtExpReverseBitsChip::<DEGREE>).name();

        // 2+1 tier config tuned from dt-shape-bench production data (18 circuits).
        // Data shows 2 natural groups: Small (BA<=16,EA=17) 50% | Large (BA<=17,EA=18) 50%.
        // Tier 3 = tight catch-all (EA=19). Dynamic fallback handles anything beyond.
        let allowed_shapes = vec![
            // Tier 1 (~56%): mainstream. Covers EA<=19, P2W<=18, MV<=20.
            HashMap::from([
                (mem_const.clone(), 18),
                (mem_var.clone(), 20),
                (base_alu.clone(), 17),
                (ext_alu.clone(), 19),
                (poseidon2_wide.clone(), 18),
                (select.clone(), 20),
                (public_values.clone(), PUB_VALUES_LOG_HEIGHT),
                (prefix_sum_checks.clone(), 10),
                (ext_exp_reverse_bits.clone(), 17),
            ]),
            // Tier 2 (safety net): mid-range. Covers EA<=20, P2W<=19.
            // Narrow chips (BA/EERB/PS) match Tier 1 - no runtime benefit from widening them.
            HashMap::from([
                (mem_const.clone(), 18),
                (mem_var.clone(), 20),
                (base_alu.clone(), 17),
                (ext_alu.clone(), 20),
                (poseidon2_wide.clone(), 19),
                (select.clone(), 20),
                (public_values.clone(), PUB_VALUES_LOG_HEIGHT),
                (prefix_sum_checks.clone(), 10),
                (ext_exp_reverse_bits.clone(), 17),
            ]),
            // Tier 3 (~44%): large shards. Covers EA<=21, MV<=21, PS<=11.
            HashMap::from([
                (mem_const.clone(), 19),
                (mem_var.clone(), 21),
                (base_alu.clone(), 18),
                (ext_alu.clone(), 21),
                (poseidon2_wide.clone(), 19),
                (select.clone(), 20),
                (public_values.clone(), PUB_VALUES_LOG_HEIGHT),
                (prefix_sum_checks.clone(), 11),
                (ext_exp_reverse_bits.clone(), 18),
            ]), 
        ];
        Self { allowed_shapes, _marker: PhantomData }
    }
}
