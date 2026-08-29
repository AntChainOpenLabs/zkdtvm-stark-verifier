mod builder;
mod compiler;
mod config;

pub use builder::*;
pub use compiler::*;
pub use config::*;

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use p3_baby_bear::DiffusionMatrixBabyBear;
    use p3_field::AbstractField;

    use crate::{
        circuit::{AsmBuilder, AsmCompiler, CircuitV2Builder},
        ir::*,
    };
    use dt_core_machine::utils::{run_test_machine, sc_run_test_machine};
    use dt_recursion_core::{machine::RecursionAir, Runtime, RuntimeError};
    use dt_stark::{
        baby_bear_poseidon2::{BabyBearPoseidon2, SCBabyBearPoseidon2},
        BabyBearPoseidon2Inner, InnerChallenge, InnerVal, StarkGenericConfig,
    };

    const DEGREE: usize = 3;

    type SC = BabyBearPoseidon2Inner;
    type F = InnerVal;
    type EF = InnerChallenge;
    type A = RecursionAir<F, DEGREE>;

    #[test]
    fn test_empty_witness_stream() {
        let mut builder = AsmBuilder::<F, EF>::default();

        let felts = builder.hint_felts_v2(3);
        assert_eq!(felts.len(), 3);
        let sum: Felt<_> = builder.eval(felts[0] + felts[1]);
        builder.assert_felt_eq(sum, felts[2]);

        let exts = builder.hint_exts_v2(3);
        assert_eq!(exts.len(), 3);
        let sum: Ext<_, _> = builder.eval(exts[0] + exts[1]);
        builder.assert_ext_ne(sum, exts[2]);

        let block = builder.into_root_block();
        let mut compiler = AsmCompiler::default();
        let program = Arc::new(compiler.compile_inner(block).validate().unwrap());
        let mut runtime = Runtime::<
            F,
            EF,
            DiffusionMatrixBabyBear,
            { dt_recursion_core::runtime::POSEIDON2_SBOX_DEGREE },
        >::new(program.clone(), SC::new().perm);
        runtime.witness_stream =
            [vec![F::one().into(), F::one().into(), F::two().into()]].concat().into();

        match runtime.run() {
            Err(RuntimeError::EmptyWitnessStream) => (),
            Ok(_) => panic!("should not succeed"),
            Err(x) => panic!("should not yield error variant: {}", x),
        }
    }

    #[test]
    fn test_ext_exp_reverse_bits() {
        let mut builder = AsmBuilder::<F, EF>::default();

        let exp = builder.hint_felts_v2(3);
        assert_eq!(exp.len(), 3);
        let base = builder.hint_ext_v2();
        let out = builder.hint_ext_v2();
        let run_out = builder.ext_exp_reverse_bits(base, exp);
        builder.assert_ext_eq(out, run_out);

        let block = builder.into_root_block();
        let mut compiler = AsmCompiler::default();
        let program = Arc::new(compiler.compile_inner(block).validate().unwrap());
        let mut runtime = Runtime::<
            F,
            EF,
            DiffusionMatrixBabyBear,
            { dt_recursion_core::runtime::POSEIDON2_SBOX_DEGREE },
        >::new(program.clone(), SC::new().perm);
        runtime.witness_stream = [
            vec![F::one().into(), F::one().into(), F::zero().into()],
            vec![F::two().into()],
            vec![F::from_canonical_u64(64).into()],
        ]
        .concat()
        .into();
        runtime.run().unwrap();

        let machine = A::sc_compress_machine(SCBabyBearPoseidon2::compressed());

        let (pk, vk) = machine.setup(&program);
        let result = sc_run_test_machine(vec![runtime.record], machine, pk, vk.clone())
            .expect("should verify");
    }

    #[test]
    fn test_ext_dot_prod() {
        let mut builder = AsmBuilder::<F, EF>::default();

        let x_vec = builder.hint_exts_v2(10000);
        let y_vec = builder.hint_exts_v2(10000);
        assert_eq!(x_vec.len(), 10000);
        assert_eq!(y_vec.len(), 10000);
        let out = builder.hint_ext_v2();
        let run_out = builder.ext_dot_prod(x_vec.clone(), y_vec.clone());
        builder.assert_ext_eq(out, run_out);

        let block = builder.into_root_block();
        let mut compiler = AsmCompiler::default();
        let program = Arc::new(compiler.compile_inner(block).validate().unwrap());
        let mut runtime = Runtime::<
            F,
            EF,
            DiffusionMatrixBabyBear,
            { dt_recursion_core::runtime::POSEIDON2_SBOX_DEGREE },
        >::new(program.clone(), SC::new().perm);

        // 生成随机向量值
        let mut x_values = Vec::with_capacity(10000);
        let mut y_values = Vec::with_capacity(10000);
        let mut dot_product = F::zero();

        for i in 0..10000 {
            let x_val = F::from_canonical_u32((i as u32 + 1) * 7 % 97);
            let y_val = F::from_canonical_u32((i as u32 + 1) * 11 % 89);
            x_values.push(x_val);
            y_values.push(y_val);
            dot_product += x_val * y_val;
        }

        runtime.witness_stream = [
            x_values.into_iter().map(Into::into).collect::<Vec<_>>(),
            y_values.into_iter().map(Into::into).collect::<Vec<_>>(),
            vec![dot_product.into()],
        ]
        .concat()
        .into();
        runtime.run().unwrap();

        let machine = A::sc_compress_machine(SCBabyBearPoseidon2::compressed());

        let (pk, vk) = machine.setup(&program);
        let result = sc_run_test_machine(vec![runtime.record], machine, pk, vk.clone())
            .expect("should verify");

        println!("num shard proofs: {}", result.shard_proofs.len());
    }

    #[test]
    fn test_ext_dot_prod_without_instr() {
        let mut builder = AsmBuilder::<F, EF>::default();

        let x_vec = builder.hint_exts_v2(10000);
        let y_vec = builder.hint_exts_v2(10000);
        assert_eq!(x_vec.len(), 10000);
        assert_eq!(y_vec.len(), 10000);
        let out = builder.hint_ext_v2();
        let mut run_out: Ext<F, EF> = builder.constant(EF::zero());
        for i in 0..10000 {
            run_out = builder.eval(run_out + x_vec[i] * y_vec[i]);
        }
        builder.assert_ext_eq(out, run_out);

        let block = builder.into_root_block();
        let mut compiler = AsmCompiler::default();
        let program = Arc::new(compiler.compile_inner(block).validate().unwrap());
        let mut runtime = Runtime::<
            F,
            EF,
            DiffusionMatrixBabyBear,
            { dt_recursion_core::runtime::POSEIDON2_SBOX_DEGREE },
        >::new(program.clone(), SC::new().perm);

        // 生成随机向量值
        let mut x_values = Vec::with_capacity(10000);
        let mut y_values = Vec::with_capacity(10000);
        let mut dot_product = F::zero();

        for i in 0..10000 {
            let x_val = F::from_canonical_u32((i as u32 + 1) * 7 % 97);
            let y_val = F::from_canonical_u32((i as u32 + 1) * 11 % 89);
            x_values.push(x_val);
            y_values.push(y_val);
            dot_product += x_val * y_val;
        }

        runtime.witness_stream = [
            x_values.into_iter().map(Into::into).collect::<Vec<_>>(),
            y_values.into_iter().map(Into::into).collect::<Vec<_>>(),
            vec![dot_product.into()],
        ]
        .concat()
        .into();
        runtime.run().unwrap();

        let machine = A::sc_compress_machine(SCBabyBearPoseidon2::compressed());

        let (pk, vk) = machine.setup(&program);
        let result = sc_run_test_machine(vec![runtime.record], machine, pk, vk.clone())
            .expect("should verify");

        println!("num shard proofs: {}", result.shard_proofs.len());
    }

    #[test]
    fn test_eq_poly() {
        let mut builder = AsmBuilder::<F, EF>::default();

        let x_vec = builder.hint_exts_v2(2);
        let y_vec = builder.hint_exts_v2(2);
        assert_eq!(x_vec.len(), 2);
        assert_eq!(y_vec.len(), 2);
        let out = builder.hint_ext_v2();
        let run_out = builder.eq_poly(x_vec.clone(), y_vec.clone());
        builder.assert_ext_eq(out, run_out);

        let block = builder.into_root_block();
        let mut compiler = AsmCompiler::default();
        let program = Arc::new(compiler.compile_inner(block).validate().unwrap());
        let mut runtime = Runtime::<
            F,
            EF,
            DiffusionMatrixBabyBear,
            { dt_recursion_core::runtime::POSEIDON2_SBOX_DEGREE },
        >::new(program.clone(), SC::new().perm);
        runtime.witness_stream = [
            vec![F::zero().into(), F::two().into()],
            vec![F::from_canonical_u64(3).into(), F::two().into()],
            vec![(F::neg_one() * F::from_canonical_u64(10)).into()],
        ]
        .concat()
        .into();
        runtime.run().unwrap();

        let machine = A::sc_compress_machine(SCBabyBearPoseidon2::compressed());

        let (pk, vk) = machine.setup(&program);
        let result = sc_run_test_machine(vec![runtime.record], machine, pk, vk.clone())
            .expect("should verify");

        println!("num shard proofs: {}", result.shard_proofs.len());
    }
}
