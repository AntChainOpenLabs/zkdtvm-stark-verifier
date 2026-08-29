pub mod instruction;
pub mod register;
pub mod state;

pub use register::{
    addi_type::AddiRegisterOp, alu_type::ALUTypeRegisterOp, b_type::BTypeRegisterOp,
    i_type::ITypeRegisterOp, j_type::JTypeRegisterOp, r_type::RTypeRegisterOp,
};
pub use state::CPUState;
