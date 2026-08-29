use p3_air::BaseAir;
use p3_field::Field;

use crate::{
    air::GlobalClaim, global_d11::StableChipId, sumcheck::trace::CompressedMatrix, MachineRecord,
};

pub use dt_derive::MachineAir;

use super::InteractionScope;

// TODO: add Id type and also fn id()

#[macro_export]
/// Macro to get the name of a chip.
macro_rules! chip_name {
    ($chip:ident, $field:ty) => {
        <$chip as MachineAir<$field>>::name(&$chip {})
    };
}

// impl<F: Field, EF: ExtensionField<F>, T: MachineAir<F>> MachineAir<EF> for T {
//     type Record = <Self as MachineAir<F>>::Record;
//     type Program = <Self as MachineAir<F>>::Program;
//     fn name(&self) -> String {
//         self.name()
//     }
//     fn preprocessed_width(&self) -> usize {
//         self.preprocessed_width()
//     }
//     fn num_rows(&self, _input: &Self::Record) -> Option<usize> {
//         self.num_rows(_input)
//     }
//     fn commit_scope(&self) -> InteractionScope {
//         self.commit_scope()
//     }
//     fn local_only(&self) -> bool {
//         self.local_only()
//     }
//     fn preprocessed_num_rows(&self, _program: &Self::Program, _instrs_len: usize) ->
// Option<usize> {         self.preprocessed_num_rows(_program, _instrs_len)
//     }

//     fn generate_trace(
//         &self,
//         input: &Self::Record,
//         output: &mut Self::Record,
//     ) -> RowMajorMatrix<EF> {
//         todo!()
//     }

//     fn included(&self, shard: &Self::Record) -> bool {
//         todo!()
//     }
// }
/// An AIR that is part of a multi table AIR arithmetization.
pub trait MachineAir<F: Field>: BaseAir<F> + 'static + Send + Sync {
    /// The execution record containing events for producing the air trace.
    type Record: MachineRecord;

    /// The program that defines the control flow of the machine.
    type Program: MachineProgram<F>;

    /// A unique identifier for this AIR as part of a machine.
    fn name(&self) -> String;

    /// The number of rows in the trace
    fn num_rows(&self, _input: &Self::Record) -> Option<usize> {
        None
    }

    /// Generate the trace for a given execution record (compressed form).
    ///
    /// - `input` is the execution record containing the events to be written to the trace.
    /// - `output` is the execution record containing events that the `MachineAir` can add to the
    ///   record such as byte lookup requests.
    fn generate_trace(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> CompressedMatrix<F>;

    /// Generate the dependencies for a given execution record.
    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        let _ = self.generate_trace(input, output);
    }

    /// Whether this execution record contains events for this air.
    fn included(&self, shard: &Self::Record) -> bool;

    /// The width of the preprocessed trace.
    fn preprocessed_width(&self) -> usize {
        0
    }

    /// The number of rows in the preprocessed trace
    fn preprocessed_num_rows(&self, _program: &Self::Program, _instrs_len: usize) -> Option<usize> {
        None
    }

    /// Generate the preprocessed trace given a specific program (compressed form).
    fn generate_preprocessed_trace(&self, _program: &Self::Program) -> Option<CompressedMatrix<F>> {
        None
    }

    /// Specifies whether it's trace should be part of either the global or local commit.
    fn commit_scope(&self) -> InteractionScope {
        InteractionScope::Local
    }

    /// Specifies whether the air only uses the local row, and not the next row.
    fn local_only(&self) -> bool {
        false
    }

    /// Stable typed-boundary owner for the canonical Global scheme.
    fn global_boundary_owner(&self) -> Option<StableChipId> {
        None
    }

    /// Extract the total Global claim without exposing physical columns to generic transport.
    fn extract_global_claim(
        &self,
        _trace: &CompressedMatrix<F>,
    ) -> Result<Option<GlobalClaim<F>>, String> {
        Ok(None)
    }

    /// Returns a representative padding row for this chip.
    ///
    /// The default implementation returns a row of all zeros with the correct width.
    /// Chips that use non-zero padding values should override this method to ensure
    /// consistency between trace generation and constraint checking.
    fn padding_row(&self) -> Vec<F> {
        vec![F::zero(); self.width()]
    }
}

/// A program that defines the control flow of a machine through a program counter.
pub trait MachineProgram<F>: Send + Sync {
    /// Gets the starting program counter.
    fn pc_start(&self) -> F;

    /// Canonical program-image boundary. Programs without a D11 Global owner leave this absent.
    fn initial_global_boundary(&self) -> Option<crate::global_d11::ProgramImageBoundaryV1<u32>> {
        None
    }
}
