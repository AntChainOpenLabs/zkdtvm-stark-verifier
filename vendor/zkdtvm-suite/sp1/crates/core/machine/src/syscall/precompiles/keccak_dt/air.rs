use core::borrow::Borrow;
use std::iter::once;

use dt_stark::{
    air::{AirInteraction, DTAirBuilder, InteractionScope},
    InteractionKind,
};
use p3_air::{Air, BaseAir};
use p3_field::AbstractField;
use p3_matrix::Matrix;

use crate::{
    operations_dt::CompactWord,
    syscall::precompiles::keccak_dt::{
        columns::{KeccakPermuteCols, NUM_KECCAK_PERMUTE_COLS},
        keccak_cols::KeccakCols,
        KeccakPermuteChip,
    },
};

impl<F> BaseAir<F> for KeccakPermuteChip {
    fn width(&self) -> usize {
        NUM_KECCAK_PERMUTE_COLS
    }
}

impl<AB> Air<AB> for KeccakPermuteChip
where
    AB: DTAirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &KeccakPermuteCols<AB::Var> = (*local).borrow();

        let a_prime_prime_prime_0_0 =
            KeccakCols::<AB::F>::eval(&local.keccak, builder, local.is_real);
        let a_prime_prime_prime_0_0: [CompactWord<_>; 2] =
            a_prime_prime_prime_0_0.map(|a| a.into());

        let receive_values = once(local.shard.into())
            .chain(once(local.clk.into()))
            .chain(once(local.keccak.step.into()))
            .chain(
                local
                    .keccak
                    .a
                    .as_flattened()
                    .as_flattened()
                    .iter()
                    .flat_map(|a| a.0.into_iter().map(|a| a.into())),
            )
            .collect::<Vec<_>>();
        builder.receive(
            AirInteraction::new(receive_values, local.is_real.into(), InteractionKind::Keccak),
            InteractionScope::Local,
        );

        let send_values = once(local.shard.into())
            .chain(once(local.clk.into()))
            .chain(once(local.keccak.step + AB::F::one()))
            .chain(a_prime_prime_prime_0_0.into_iter().flat_map(|a| a.0.into_iter()))
            .chain(
                local.keccak.a_prime_prime.as_flattened()[1..]
                    .as_flattened()
                    .iter()
                    .flat_map(|a| a.0.into_iter().map(|a| a.into())),
            )
            .collect::<Vec<_>>();
        builder.send(
            AirInteraction::new(send_values, local.is_real.into(), InteractionKind::Keccak),
            InteractionScope::Local,
        );
    }
}
