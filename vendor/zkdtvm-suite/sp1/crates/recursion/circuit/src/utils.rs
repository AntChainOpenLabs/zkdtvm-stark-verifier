use p3_baby_bear::BabyBear;
use p3_bn254_fr::Bn254Fr;
use p3_field::{AbstractField, PrimeField32};

use dt_recursion_compiler::ir::{Builder, Config, Felt, Var};
use dt_recursion_core::DIGEST_SIZE;

use dt_stark::Word;

/// Convert 8 BabyBear words into a Bn254Fr field element by shifting by 31 bits each time. The last
/// word becomes the least significant bits.
#[allow(dead_code)]
pub fn babybears_to_bn254(digest: &[BabyBear; 8]) -> Bn254Fr {
    let mut result = Bn254Fr::zero();
    for word in digest.iter() {
        // Since BabyBear prime is less than 2^31, we can shift by 31 bits each time and still be
        // within the Bn254Fr field, so we don't have to truncate the top 3 bits.
        result *= Bn254Fr::from_canonical_u64(1 << 31);
        result += Bn254Fr::from_canonical_u32(word.as_canonical_u32());
    }
    result
}

/// Convert 32 BabyBear bytes into a Bn254Fr field element. The first byte's most significant 3 bits
/// (which would become the 3 most significant bits) are truncated.
#[allow(dead_code)]
pub fn babybear_bytes_to_bn254(bytes: &[BabyBear; 32]) -> Bn254Fr {
    let mut result = Bn254Fr::zero();
    for (i, byte) in bytes.iter().enumerate() {
        debug_assert!(byte < &BabyBear::from_canonical_u32(256));
        if i == 0 {
            // 32 bytes is more than Bn254 prime, so we need to truncate the top 3 bits.
            result = Bn254Fr::from_canonical_u32(byte.as_canonical_u32() & 0x1f);
        } else {
            result *= Bn254Fr::from_canonical_u32(256);
            result += Bn254Fr::from_canonical_u32(byte.as_canonical_u32());
        }
    }
    result
}

#[allow(dead_code)]
pub fn felts_to_bn254_var<C: Config>(
    builder: &mut Builder<C>,
    digest: &[Felt<C::F>; DIGEST_SIZE],
) -> Var<C::N> {
    let var_2_31: Var<_> = builder.constant(C::N::from_canonical_u32(1 << 31));
    let result = builder.constant(C::N::zero());
    for (i, word) in digest.iter().enumerate() {
        let word_var = builder.felt2var_circuit(*word);
        if i == 0 {
            builder.assign(result, word_var);
        } else {
            builder.assign(result, result * var_2_31 + word_var);
        }
    }
    result
}

#[allow(dead_code)]
pub fn felt_bytes_to_bn254_var<C: Config>(
    builder: &mut Builder<C>,
    bytes: &[Felt<C::F>; 32],
) -> Var<C::N> {
    let var_256: Var<_> = builder.constant(C::N::from_canonical_u32(256));
    let zero_var: Var<_> = builder.constant(C::N::zero());
    let result = builder.constant(C::N::zero());
    for (i, byte) in bytes.iter().enumerate() {
        let byte_bits = builder.num2bits_f_circuit(*byte);
        if i == 0 {
            // Since 32 bytes doesn't fit into Bn254, we need to truncate the top 3 bits.
            // For first byte, zero out 3 most significant bits.
            for i in 0..3 {
                builder.assign(byte_bits[8 - i - 1], zero_var);
            }
            let byte_var = builder.bits2num_v_circuit(&byte_bits);
            builder.assign(result, byte_var);
        } else {
            let byte_var = builder.bits2num_v_circuit(&byte_bits);
            builder.assign(result, result * var_256 + byte_var);
        }
    }
    result
}

/// Converts a slice of words to a flat vector of bytes/elements.
/// Pre-allocates capacity based on the known word size (DIGEST_SIZE).
#[allow(dead_code)]
pub fn words_to_bytes<T: Copy>(words: &[Word<T>]) -> Vec<T> {
    let mut result = Vec::with_capacity(words.len() * DIGEST_SIZE);
    result.extend(words.iter().flat_map(|w| w.0));
    result
}

#[cfg(test)]
pub(crate) mod sc_tests {
    use std::sync::Arc;

    use dt_core_machine::utils::{sc_run_test_machine_with_prover, setup_logger};
    use dt_recursion_compiler::circuit::{AsmCompiler, AsmConfig};

    use crate::witness::WitnessBlock;
    use dt_recursion_compiler::ir::DslIrBlock;
    use dt_recursion_core::{machine::RecursionAir, Runtime};
    use dt_stark::sumcheck::prover::{SCMachineProver, SumcheckProver};
    use log::debug;

    #[cfg(not(feature = "koalabear"))]
    use dt_stark::baby_bear_poseidon2::SCBabyBearPoseidon2;
    #[cfg(not(feature = "koalabear"))]
    use dt_stark::{InnerChallenge, InnerVal};

    #[cfg(feature = "koalabear")]
    use dt_stark::koalabear_poseidon2::koala_bear_poseidon2::SCKoalaBearPoseidon2 as SCBabyBearPoseidon2;
    #[cfg(feature = "koalabear")]
    use dt_stark::koalabear_poseidon2::{InnerChallenge, InnerVal};

    type SC = SCBabyBearPoseidon2;
    type F = InnerVal;
    type EF = InnerChallenge;

    /// A simplified version of some code from `recursion/core/src/stark/mod.rs`.
    /// Takes in a program and runs it with the given witness and generates a proof with a variety
    /// of machines depending on the provided test_config.
    pub(crate) fn run_test_recursion_with_prover<
        P: SCMachineProver<SC, RecursionAir<F, 3>, RecursionAir<EF, 3>>,
    >(
        block: DslIrBlock<AsmConfig<F, EF>>,
        witness_stream: impl IntoIterator<Item = WitnessBlock<AsmConfig<F, EF>>>,
        num_skip_rounds: usize,
        chip_log_height_threshold: usize,
    ) {
        // setup_logger();
        debug!("num of ir: {}", block.ops.len());
        let compile_span = tracing::debug_span!("compile").entered();
        let mut compiler = AsmCompiler::<AsmConfig<F, EF>>::default();
        let program = Arc::new(compiler.compile_inner(block).validate().unwrap());
        compile_span.exit();
        debug!("compile done");

        let config = SC::default();

        let run_span = tracing::debug_span!("run the recursive program").entered();
        #[cfg(not(feature = "koalabear"))]
        let mut runtime = Runtime::<F, EF, _>::new(program.clone(), config.perm.clone());
        #[cfg(feature = "koalabear")]
        let mut runtime = Runtime::<F, EF, _, 3>::new(program.clone(), config.perm.clone());
        runtime.witness_stream.extend(witness_stream);
        tracing::debug!("start running");
        tracing::debug_span!("run").in_scope(|| runtime.run().unwrap());
        assert!(runtime.witness_stream.is_empty());
        run_span.exit();
        debug!("run done");

        let records = vec![runtime.record];

        // Run with the poseidon2 wide chip.
        let proof_wide_span = tracing::debug_span!("Run test with wide machine").entered();
        let wide_machine =
            RecursionAir::<_, 3>::sc_compress_machine(SCBabyBearPoseidon2::compressed());
        let (pk, vk) = wide_machine.setup(&program);
        let prover = P::new(wide_machine);
        let pk = prover.pk_to_device(&pk);
        let result = sc_run_test_machine_with_prover::<_, _, _, P>(
            &prover,
            records.clone(),
            pk,
            vk,
            num_skip_rounds,
            chip_log_height_threshold,
        );
        debug!("proof wide done");
        proof_wide_span.exit();

        if let Err(e) = result {
            panic!("Verification failed: {:?}", e);
        }
    }

    #[allow(dead_code)]
    pub(crate) fn run_test_recursion(
        block: DslIrBlock<AsmConfig<F, EF>>,
        witness_stream: impl IntoIterator<Item = WitnessBlock<AsmConfig<F, EF>>>,
        num_skip_rounds: usize,
        chip_log_height_threshold: usize,
    ) {
        run_test_recursion_with_prover::<SumcheckProver<_, _, _>>(
            block,
            witness_stream,
            num_skip_rounds,
            chip_log_height_threshold,
        )
    }
}
