use p3_sha256::Sha256;
use p3_symmetric::CryptographicHasher;

use super::constants::{OVERFLOW_CERTIFICATE_JSON, PARAMETER_MANIFEST_CANONICAL_JSON};

#[must_use]
pub fn parameter_manifest_digest() -> [u8; 32] {
    Sha256.hash_iter_slices([PARAMETER_MANIFEST_CANONICAL_JSON.as_bytes()])
}

#[must_use]
pub fn overflow_certificate_digest() -> [u8; 32] {
    Sha256.hash_iter_slices([OVERFLOW_CERTIFICATE_JSON.as_bytes()])
}
