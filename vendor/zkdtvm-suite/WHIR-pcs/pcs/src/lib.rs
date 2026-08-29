#![allow(non_snake_case)]
#![allow(clippy::assertions_on_result_states)]
#![allow(clippy::needless_range_loop)]
#![allow(incomplete_features)]
#![allow(long_running_const_eval)]

extern crate alloc;
extern crate core;

pub mod utils;
pub mod whir;

/// Compatibility exports for the existing zkDTVM/sp1 integration.
///
/// The implementation underneath is WHIR-only. This module keeps old import
/// paths such as `pcs::basefold::mlpcs::MlPCS` alive while the upper layers are
/// migrated to WHIR names.
pub mod basefold {
    pub mod mlpcs {
        pub use crate::whir::mlpcs::*;
    }

    pub mod sumcheck {
        pub use crate::whir::sumcheck::*;
    }

    pub mod basefold_pcs {
        pub use crate::whir::whir_pcs::{
            compute_commit_schedule, compute_commit_schedule_with_log_foldings, CommitGroup,
            DimAndNo, PrunedQueryOpenings, WhirError as BaseFoldError,
        };
        pub use crate::whir::{
            WhirConfig as BasefoldConfig, WhirInputProof as BasefoldInputProof,
            WhirPcs as BaseFoldPcs, WhirPcsProverData as BasefoldPcsProverData,
            WhirProof as BasefoldProof,
        };
    }

    pub mod basefold_types {
        pub use crate::whir::whir_types::{
            compute_commit_schedule, compute_commit_schedule_with_log_foldings, CommitGroup,
            PrunedQueryOpenings, WhirConfig as BasefoldConfig, WhirError as BaseFoldError,
            WhirInputProof as BasefoldInputProof, WhirPcs as BaseFoldPcs,
            WhirPcsProverData as BasefoldPcsProverData, WhirProof as BasefoldProof,
        };
    }

    pub use crate::whir::{
        compute_commit_schedule, compute_commit_schedule_with_log_foldings, CommitGroup,
        MlCommitOptions, MlPCS, PrunedQueryOpenings, StackingConfig, StackingReductionProof,
        WhirConfig as BasefoldConfig, WhirError as BaseFoldError,
        WhirInputProof as BasefoldInputProof, WhirPcs as BaseFoldPcs,
        WhirPcsProverData as BasefoldPcsProverData, WhirProof as BasefoldProof,
    };
}
