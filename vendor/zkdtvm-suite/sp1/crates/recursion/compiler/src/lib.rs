//! Recursion compiler: lowers recursion programs (written against the
//! recursion IR) into circuit ASM that the recursion runtime executes. Used to
//! build the compress / shrink / wrap verifier circuits.

#![allow(clippy::type_complexity)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::print_stdout)]

extern crate alloc;

pub mod circuit;
pub mod config;
pub mod constraints;
pub mod ir;

pub mod prelude {
    pub use crate::ir::*;
    pub use dt_recursion_derive::DslVariable;
}
