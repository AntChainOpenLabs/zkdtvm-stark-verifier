use std::iter::once;

use dt_stark::{
    air::{AirInteraction, BaseAirBuilder, InteractionScope},
    InteractionKind,
};
use p3_air::AirBuilder;

use crate::adapter::instruction::InstructionCols;

/// A trait which contains methods related to program interactions in an AIR.
pub trait ProgramAirBuilder: BaseAirBuilder {
    /// Sends an instruction.
    fn send_program(
        &mut self,
        pc: impl Into<Self::Expr>,
        instruction: InstructionCols<impl Into<Self::Expr> + Clone>,
        multiplicity: impl Into<Self::Expr>,
    ) {
        let opcode_expr = instruction.opcode.clone().into();
        let values = once(pc.into())
            .chain(once(opcode_expr))
            .chain(instruction.into_iter().map(|x| x.into()))
            .collect();

        self.send(
            AirInteraction::new(values, multiplicity.into(), InteractionKind::Program),
            InteractionScope::Local,
        );
    }

    /// Receives an instruction.
    fn receive_program(
        &mut self,
        pc: impl Into<Self::Expr>,
        instruction: InstructionCols<impl Into<Self::Expr> + Clone>,
        multiplicity: impl Into<Self::Expr>,
    ) {
        let opcode_expr = instruction.opcode.clone().into();
        let values: Vec<<Self as AirBuilder>::Expr> = once(pc.into())
            .chain(once(opcode_expr))
            .chain(instruction.into_iter().map(|x| x.into()))
            .collect();

        self.receive(
            AirInteraction::new(values, multiplicity.into(), InteractionKind::Program),
            InteractionScope::Local,
        );
    }
}
