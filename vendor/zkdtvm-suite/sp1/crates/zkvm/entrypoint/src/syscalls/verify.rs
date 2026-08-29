#[cfg(target_os = "zkvm")]
use core::arch::asm;

cfg_if::cfg_if! {
    if #[cfg(target_os = "zkvm")] {
        use crate::syscalls::VERIFY_DT_PROOF;
        use crate::zkvm::DEFERRED_PROOFS_DIGEST;
        use p3_field::AbstractField;

        #[cfg(feature = "babybear")]
        use p3_baby_bear::BabyBear;
        #[cfg(feature = "babybear")]
        use dt_primitives::hash_deferred_proof;

        #[cfg(feature = "koalabear")]
        use p3_koala_bear::KoalaBear;
        #[cfg(feature = "koalabear")]
        use dt_primitives::sc_hash_deferred_proof;
    }
}

#[no_mangle]
#[allow(unused_variables)]
pub fn syscall_verify_dt_proof(vk_digest: &[u32; 8], pv_digest: &[u8; 32]) {
    #[cfg(target_os = "zkvm")]
    {
        // SAFETY: zkvm is single-threaded; we hold the only mutable access to
        // DEFERRED_PROOFS_DIGEST.
        unsafe {
            asm!(
                "ecall",
                in("t0") VERIFY_DT_PROOF,
                in("a0") vk_digest.as_ptr(),
                in("a1") pv_digest.as_ptr(),
            );

            let deferred_proofs_digest = DEFERRED_PROOFS_DIGEST.as_mut().unwrap();

            // Deferred-proof digest must use the same field type and hasher as the prover:
            // KoalaBear path uses `SCField` and `sc_hash_deferred_proof` (see `dt-primitives`);
            // BabyBear path uses `hash_deferred_proof`. Mixing these breaks verification.
            #[cfg(feature = "babybear")]
            {
                let vk_digest_field =
                    core::array::from_fn(|i| BabyBear::from_canonical_u32(vk_digest[i]));
                let pv_digest_field =
                    core::array::from_fn(|i| BabyBear::from_canonical_u8(pv_digest[i]));

                *deferred_proofs_digest =
                    hash_deferred_proof(deferred_proofs_digest, &vk_digest_field, &pv_digest_field);
            }
            #[cfg(feature = "koalabear")]
            {
                let vk_digest_field =
                    core::array::from_fn(|i| KoalaBear::from_canonical_u32(vk_digest[i]));
                let pv_digest_field =
                    core::array::from_fn(|i| KoalaBear::from_canonical_u8(pv_digest[i]));

                *deferred_proofs_digest = sc_hash_deferred_proof(
                    deferred_proofs_digest,
                    &vk_digest_field,
                    &pv_digest_field,
                );
            }
        }
    }

    #[cfg(not(target_os = "zkvm"))]
    unreachable!()
}
