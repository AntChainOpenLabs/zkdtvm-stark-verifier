pub mod addi_type;
pub mod alu_type;
pub mod i_type;
pub mod j_type;
pub mod r_type;
// branch
pub mod b_type;

//NOTE:
/*
special case:
1, op_a_immutable case: branch,store
2, op_a_value need slice_u8_range_check: jump, hint_len
*/
