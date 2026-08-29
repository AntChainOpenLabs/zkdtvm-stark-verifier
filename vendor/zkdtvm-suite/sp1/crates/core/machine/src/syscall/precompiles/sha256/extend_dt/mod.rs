mod air;
pub mod column;
mod controller;
pub mod extend_controller_polyair;
pub mod extend_polyair;
mod trace;

pub use column::*;
pub use controller::*;

#[derive(Default)]
pub struct ShaExtendChip;

impl ShaExtendChip {
    pub const fn new() -> Self {
        Self {}
    }
}
