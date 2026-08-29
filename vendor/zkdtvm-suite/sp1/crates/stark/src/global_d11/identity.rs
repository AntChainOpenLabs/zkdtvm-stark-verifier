//! Product-local composite identity for the frozen Global146 protocol.

use p3_field::PrimeField32;
use p3_sha256::Sha256;
use p3_symmetric::CryptographicHasher;

use super::{
    CORE_GLOBAL_OWNER_REGISTRY_DIGEST, D11_FIELD_ID, D11_FULL_ADD_FORMULA_ID,
    D11_MIXED_ADD_FORMULA_ID, D11_PACK_ID, D11_PROJECTIVE_228_QDELTA_WIRE_ID,
    D11_PUBLIC_VALUES_DIGEST_ID, D11_SCHEME_ID, GLOBAL_CLAIM_SCHEMA_TAG_V3,
    OVERFLOW_CERTIFICATE_SHA256, PARAMETER_MANIFEST_SHA256,
};

const IDENTITY_DOMAIN: &[u8] = b"dt-product-global228-qdelta-composite-identity-v2\0";
const IDENTITY_HASH_ID: &str = "sha256-v1";

/// Product-local schema for the composite identity carried by keys and artifacts.
pub const GLOBAL146_PRODUCT_SCHEMA_VERSION: u32 = 8;
/// Current native constraint-program schema bound by this identity.
pub const GLOBAL146_CONSTRAINT_PROGRAM_SCHEMA_VERSION: u32 = 11;
/// Current native AIR registry bound by this identity.
pub const GLOBAL146_NATIVE_AIR_REGISTRY_VERSION: u32 = 13;
/// Current native ladder-cache schema bound by this identity.
pub const GLOBAL146_NATIVE_LADDER_CACHE_SCHEMA_VERSION: u32 = 25;
/// Current compact reduce-proof wire bound by this identity.
pub const GLOBAL146_DT_REDUCE_PROOF_WIRE_VERSION: u32 = 11;
/// Current circuit/key artifact version bound by this identity.
pub const GLOBAL146_CIRCUIT_VERSION: &str = "v8.0.5";

/// System-wide shared-beta union bound for the admitted interval-V6 product.  The quintic challenge
/// space has more than 154 bits.  Two classes (relation roots and lookup collisions), at most
/// 256 beta-dependent families, 2^16 shards, 2^22 rows, and degree below 64 contribute fewer
/// than 2^53 roots, leaving strictly more than 101 bits of statistical soundness.
pub const GLOBAL_SHARED_BETA_CHALLENGE_BITS_FLOOR: u32 = 154;
pub const GLOBAL_SHARED_BETA_UNION_LOG2_CEIL: u32 = 53;
pub const GLOBAL_SHARED_BETA_SOUNDNESS_BITS_FLOOR: u32 =
    GLOBAL_SHARED_BETA_CHALLENGE_BITS_FLOOR - GLOBAL_SHARED_BETA_UNION_LOG2_CEIL;
pub const GLOBAL_SHARED_BETA_TARGET_BITS: u32 = 100;

pub const GLOBAL146_LAYOUT_DESCRIPTOR: &str = "Projective228QIntervalV6|CorePV120=has,count,start33,end33|CoreVKMeta32=commit8,pc1,boundary-kind1,x11,y11|Main228=Map24,index1,input33,products55,cumulative33,qmap10,qu0:6,qu1:10,qu3:10,qu4:6,qu5:10,qoutx:10,qouty:10,qoutz:10|TileReducer83=selectors8,control6,typed-values66,rank1,next-rank1,next-tag1|StatementBoundary=wide257,narrow241,rows10n+4|ConstraintTerminal=main94,reserved6,precomputed25|ConstraintBoundary=main167,reserved80,precomputed41,lookups24|ProofShapeGlobalPacked=rows48..112,Ext5x2";
pub const GLOBAL146_CONSTRAINT_DESCRIPTOR: &str = "Projective228QIntervalV6|GlobalClaim=has,count,start33,end33|Main:raw=reduced+Q*f,MapQ,u0Q,u1Q,u3Q,u4Q,u5Q,outXQ=q0-q1,outYQ=q2+q3,outZQ=q4+q5|Reducer:selector-onehot,active-row,N/P-fixed,P-onehot,K-gap,real-prefix,last-real,dummy-I,rank-next-tag-local,rebase15,normalize6,ordinal-product-stage,product-tail,root-finite-infinity,product-beta|Recursion:terminal-summary-split,canonical-interval-equality,canonical-program-boundary,statement-kind-child-interval-only,absent-preserves-state";
pub const GLOBAL146_INTERACTION_DESCRIPTOR: &str = "Projective228QIntervalV6|Main:tile-I,row-recv(index,input),row-send(index+1,cumulative),endpoint+Byte|Reducer:public(-start@0,+end@N),internal34base(point,operand,reduced,control),36semantic+2control-affine,batch2-perm19,schedule-domain2^23,tuple=N/P/node/stage/product/flow|owner=GlobalTileReducerV3:43|Recursion:terminal-state-lcs-summary,proof-shape-global-packed-1050,adjacent-end-start,root-seed-to-I,boundary-kind-child-interval-only";

/// Frozen digest of the canonical Main228 plus TileReducer83 layout.
pub const GLOBAL146_LAYOUT_DIGEST: [u8; 32] = [
    0x9a, 0xdb, 0xaf, 0x0a, 0xb6, 0x68, 0x52, 0xd5, 0x50, 0x0f, 0x86, 0xd7, 0x2d, 0x38, 0xd8, 0x2a,
    0x69, 0x14, 0x69, 0xca, 0xcd, 0x7f, 0x5e, 0x78, 0xfc, 0xc5, 0x00, 0xbd, 0x96, 0x3b, 0xeb, 0x01,
];
/// Frozen digest of the canonical Global interval-V6 constraint roots.
pub const GLOBAL146_CONSTRAINT_ROOTS_DIGEST: [u8; 32] = [
    0x90, 0xd0, 0xb1, 0x1a, 0x4a, 0xe7, 0x5d, 0xe3, 0x07, 0xb7, 0x14, 0xca, 0x8d, 0xef, 0x52, 0x31,
    0xe3, 0xdc, 0xc8, 0xed, 0x58, 0xec, 0xa8, 0x67, 0x8b, 0x92, 0x80, 0x57, 0xe0, 0xdb, 0x7b, 0x99,
];
/// Frozen digest of the canonical Global interval-V6 interactions and their order.
pub const GLOBAL146_INTERACTION_SCHEDULE_DIGEST: [u8; 32] = [
    0xa9, 0x56, 0xf6, 0x0b, 0xb2, 0x4d, 0x28, 0x88, 0x37, 0x42, 0xca, 0x4f, 0x09, 0xba, 0x8f, 0x45,
    0x7c, 0x6d, 0xf4, 0x2d, 0x99, 0x73, 0xcf, 0xfa, 0xcf, 0xd5, 0x16, 0x73, 0x35, 0xb9, 0x2a, 0x70,
];
/// Registry digest for proof roles which canonically have no Global owner.
pub const EMPTY_GLOBAL_OWNER_REGISTRY_DIGEST: [u8; 32] = [
    0x90, 0x17, 0x13, 0x7a, 0x2f, 0x66, 0xbf, 0x41, 0x01, 0xc2, 0x24, 0x8f, 0xf4, 0x78, 0xf1, 0x7a,
    0x0c, 0x57, 0xc3, 0xd7, 0xeb, 0xe5, 0x85, 0xab, 0x4a, 0x5c, 0xfa, 0xa6, 0x60, 0xc6, 0xf7, 0xac,
];

/// Canonical product-local composite identity. It is data, never a protocol selector.
pub const GLOBAL146_COMPOSITE_IDENTITY: [u8; 32] = [
    0x84, 0xe7, 0x69, 0xdf, 0x0b, 0x8d, 0xa2, 0x96, 0xd6, 0x19, 0xb6, 0xda, 0x61, 0x39, 0xa9, 0x03,
    0x87, 0xd5, 0x65, 0x03, 0x0d, 0x99, 0x76, 0x53, 0x03, 0xd0, 0x0e, 0x0d, 0xbe, 0x22, 0x26, 0x2a,
];

fn append_bytes(out: &mut Vec<u8>, label: &str, value: &[u8]) {
    for bytes in [label.as_bytes(), value] {
        let len = u32::try_from(bytes.len()).expect("Global146 identity component fits u32");
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(bytes);
    }
}

fn append_str(out: &mut Vec<u8>, label: &str, value: &str) {
    append_bytes(out, label, value.as_bytes());
}

fn append_u32(out: &mut Vec<u8>, label: &str, value: u32) {
    append_bytes(out, label, &value.to_le_bytes());
}

/// Recompute the canonical identity from its frozen components.
#[must_use]
pub fn compute_global146_composite_identity() -> [u8; 32] {
    let mut bytes = Vec::with_capacity(1024);
    bytes.extend_from_slice(IDENTITY_DOMAIN);
    append_str(&mut bytes, "identity-hash", IDENTITY_HASH_ID);
    append_str(&mut bytes, "scheme", D11_SCHEME_ID);
    append_str(&mut bytes, "field", D11_FIELD_ID);
    append_str(&mut bytes, "pack", D11_PACK_ID);
    append_str(&mut bytes, "mixed-add-formula", D11_MIXED_ADD_FORMULA_ID);
    append_str(&mut bytes, "full-add-formula", D11_FULL_ADD_FORMULA_ID);
    append_str(&mut bytes, "public-values-digest", D11_PUBLIC_VALUES_DIGEST_ID);
    append_u32(&mut bytes, "global-claim-schema", GLOBAL_CLAIM_SCHEMA_TAG_V3);
    append_bytes(&mut bytes, "parameter-manifest", &PARAMETER_MANIFEST_SHA256);
    append_bytes(&mut bytes, "overflow-certificate", &OVERFLOW_CERTIFICATE_SHA256);
    append_bytes(&mut bytes, "global-layout", &GLOBAL146_LAYOUT_DIGEST);
    append_bytes(&mut bytes, "global-constraint-roots", &GLOBAL146_CONSTRAINT_ROOTS_DIGEST);
    append_bytes(&mut bytes, "global-interaction-schedule", &GLOBAL146_INTERACTION_SCHEDULE_DIGEST);
    append_bytes(&mut bytes, "core-owner-registry", &CORE_GLOBAL_OWNER_REGISTRY_DIGEST);
    append_bytes(&mut bytes, "empty-owner-registry", &EMPTY_GLOBAL_OWNER_REGISTRY_DIGEST);
    append_u32(&mut bytes, "product-schema", GLOBAL146_PRODUCT_SCHEMA_VERSION);
    append_u32(&mut bytes, "d11-wire", u32::from(D11_PROJECTIVE_228_QDELTA_WIRE_ID));
    append_u32(
        &mut bytes,
        "constraint-program-schema",
        GLOBAL146_CONSTRAINT_PROGRAM_SCHEMA_VERSION,
    );
    append_u32(&mut bytes, "native-air-registry", GLOBAL146_NATIVE_AIR_REGISTRY_VERSION);
    append_u32(
        &mut bytes,
        "native-ladder-cache-schema",
        GLOBAL146_NATIVE_LADDER_CACHE_SCHEMA_VERSION,
    );
    append_u32(&mut bytes, "dt-reduce-proof-wire", GLOBAL146_DT_REDUCE_PROOF_WIRE_VERSION);
    append_str(&mut bytes, "circuit-version", GLOBAL146_CIRCUIT_VERSION);
    Sha256.hash_iter_slices([bytes.as_slice()])
}

/// Lift the identity bytes injectively into any supported proof field.
#[must_use]
pub fn global146_identity_fields<F: PrimeField32>() -> [F; 32] {
    GLOBAL146_COMPOSITE_IDENTITY.map(F::from_canonical_u8)
}

/// Reject a key or artifact carrying anything except the current identity.
pub fn validate_global146_identity(identity: &[u8; 32]) -> Result<(), &'static str> {
    if identity == &GLOBAL146_COMPOSITE_IDENTITY {
        Ok(())
    } else {
        Err("Global146 composite identity mismatch")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::global_d11::BoundaryOwnerRegistryV2;

    #[test]
    fn composite_identity_known_answer_and_owner_digests_are_frozen() {
        assert_eq!(compute_global146_composite_identity(), GLOBAL146_COMPOSITE_IDENTITY);
        assert_eq!(
            Sha256.hash_iter_slices([GLOBAL146_LAYOUT_DESCRIPTOR.as_bytes()]),
            GLOBAL146_LAYOUT_DIGEST
        );
        assert_eq!(
            Sha256.hash_iter_slices([GLOBAL146_CONSTRAINT_DESCRIPTOR.as_bytes()]),
            GLOBAL146_CONSTRAINT_ROOTS_DIGEST
        );
        assert_eq!(
            Sha256.hash_iter_slices([GLOBAL146_INTERACTION_DESCRIPTOR.as_bytes()]),
            GLOBAL146_INTERACTION_SCHEDULE_DIGEST
        );
        assert!(GLOBAL_SHARED_BETA_SOUNDNESS_BITS_FLOOR > GLOBAL_SHARED_BETA_TARGET_BITS);
        assert_eq!(
            BoundaryOwnerRegistryV2::new(Vec::new()).unwrap().digest,
            EMPTY_GLOBAL_OWNER_REGISTRY_DIGEST
        );
        assert!(validate_global146_identity(&GLOBAL146_COMPOSITE_IDENTITY).is_ok());
        let mut wrong = GLOBAL146_COMPOSITE_IDENTITY;
        wrong[0] ^= 1;
        assert!(validate_global146_identity(&wrong).is_err());
    }
}
