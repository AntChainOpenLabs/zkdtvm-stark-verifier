//! The pow/range multiplicity counts carried by [`RecursionPowerPool`] requests.
//!
//! There is no PowerChecker AIR consuming these; the pool and its counts stay
//! because they are live record furniture (serde format + append).
//!
//! [`RecursionPowerPool`]: crate::system_dt::RecursionPowerPool

use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PowerCheckerCounts {
    pub pow: u32,
    pub range: u32,
}
