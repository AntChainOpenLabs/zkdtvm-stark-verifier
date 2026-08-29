mod memory;
mod program;
mod word;

pub use memory::*;
pub use program::*;
pub use word::*;

use dt_stark::air::{BaseAirBuilder, DTAirBuilder};

/// A trait which contains methods related to memory interactions in an AIR.
pub trait DTCoreAirBuilder:
    DTAirBuilder + WordAirBuilder + MemoryAirBuilder + ProgramAirBuilder
{
}

impl<AB: BaseAirBuilder> MemoryAirBuilder for AB {}
impl<AB: BaseAirBuilder> ProgramAirBuilder for AB {}
impl<AB: BaseAirBuilder> WordAirBuilder for AB {}
impl<AB: BaseAirBuilder + DTAirBuilder> DTCoreAirBuilder for AB {}
