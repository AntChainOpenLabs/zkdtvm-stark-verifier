use crate::{
    air::Block,
    chips::{
        alu_base::{BaseAluAccessCols, BaseAluValueCols},
        alu_ext::{ExtAluAccessCols, ExtAluValueCols},
        public_values::{PublicValuesCols, PublicValuesPreprocessedCols},
        select::{SelectCols, SelectPreprocessedCols},
    },
    BaseAluInstr, BaseAluIo, BaseAluOpcode, CommitPublicValuesEvent, CommitPublicValuesInstr,
    ExtAluInstr, ExtAluIo, ExtAluOpcode, Poseidon2Event, Poseidon2Instr, SelectEvent,
    SelectInstr,
};
use p3_baby_bear::BabyBear;

#[cfg(feature = "sys")]
use crate::{
    chips::{
        ext_exp_reverse_bits::{ExtExpReverseBitsCols, ExtExpReverseBitsPreprocessedCols},
        poly::{PolyEvalCols, PolyEvalPreprocessedCols},
        poseidon2_wide::columns::preprocessed::Poseidon2PreprocessedColsWide,
    },
    ExtExpReverseBitsEventFFI, ExtExpReverseBitsInstrFFI, PolyEvalEventFFI, PolyEvalInstrFFI,
};

#[cfg(not(feature = "sys"))]
pub use fallback::*;

#[cfg(feature = "sys")]
#[link(name = "dt-recursion-core-sys", kind = "static")]
extern "C-unwind" {
    // ======================== BabyBear FFI functions ========================
    pub fn alu_base_event_to_row_babybear(
        io: &BaseAluIo<BabyBear>,
        cols: &mut BaseAluValueCols<BabyBear>,
    );
    pub fn alu_base_instr_to_row_babybear(
        instr: &BaseAluInstr<BabyBear>,
        cols: &mut BaseAluAccessCols<BabyBear>,
    );

    pub fn alu_ext_event_to_row_babybear(
        io: &ExtAluIo<Block<BabyBear>>,
        cols: &mut ExtAluValueCols<BabyBear>,
    );
    pub fn alu_ext_instr_to_row_babybear(
        instr: &ExtAluInstr<BabyBear>,
        cols: &mut ExtAluAccessCols<BabyBear>,
    );

    pub fn public_values_event_to_row_babybear(
        io: &CommitPublicValuesEvent<BabyBear>,
        digest_idx: usize,
        cols: &mut PublicValuesCols<BabyBear>,
    );
    pub fn public_values_instr_to_row_babybear(
        instr: &CommitPublicValuesInstr<BabyBear>,
        digest_idx: usize,
        cols: &mut PublicValuesPreprocessedCols<BabyBear>,
    );

    pub fn select_event_to_row_babybear(
        io: &SelectEvent<BabyBear>,
        cols: &mut SelectCols<BabyBear>,
    );
    pub fn select_instr_to_row_babybear(
        instr: &SelectInstr<BabyBear>,
        cols: &mut SelectPreprocessedCols<BabyBear>,
    );

    pub fn poseidon2_skinny_event_to_row_babybear(io: &Poseidon2Event<BabyBear>, cols: *mut u8);

    pub fn poseidon2_wide_event_to_row_babybear(
        input: *const BabyBear,
        input_row: *mut BabyBear,
        sbox_state: bool,
    );
    pub fn poseidon2_wide_instr_to_row_babybear(
        instr: &Poseidon2Instr<BabyBear>,
        cols: &mut Poseidon2PreprocessedColsWide<BabyBear>,
    );

    pub fn poly_eval_event_to_row_babybear(
        io: &PolyEvalEventFFI<BabyBear>,
        i: usize,
        cols: &mut PolyEvalCols<BabyBear>,
    );
    pub fn poly_eval_instr_to_row_babybear(
        instr: &PolyEvalInstrFFI<BabyBear>,
        i: usize,
        len: usize,
        cols: &mut PolyEvalPreprocessedCols<BabyBear>,
    );

    pub fn ext_exp_reverse_bits_event_to_row_babybear(
        io: &ExtExpReverseBitsEventFFI<BabyBear>,
        i: usize,
        cols: &mut ExtExpReverseBitsCols<BabyBear>,
    );
    pub fn ext_exp_reverse_bits_instr_to_row_babybear(
        instr: &ExtExpReverseBitsInstrFFI<BabyBear>,
        i: usize,
        len: usize,
        cols: &mut ExtExpReverseBitsPreprocessedCols<BabyBear>,
    );

    // ======================== KoalaBear FFI functions ========================
    // KoalaBear has the same memory layout as BabyBear (u32 wrapper), so we
    // declare these with BabyBear types. The C++ side uses kb31_t for modular
    // arithmetic while the Rust side transmutes to/from the actual field type.
    pub fn alu_base_event_to_row_koalabear(
        io: &BaseAluIo<BabyBear>,
        cols: &mut BaseAluValueCols<BabyBear>,
    );
    pub fn alu_base_instr_to_row_koalabear(
        instr: &BaseAluInstr<BabyBear>,
        cols: &mut BaseAluAccessCols<BabyBear>,
    );

    pub fn alu_ext_event_to_row_koalabear(
        io: &ExtAluIo<Block<BabyBear>>,
        cols: &mut ExtAluValueCols<BabyBear>,
    );
    pub fn alu_ext_instr_to_row_koalabear(
        instr: &ExtAluInstr<BabyBear>,
        cols: &mut ExtAluAccessCols<BabyBear>,
    );

    pub fn public_values_event_to_row_koalabear(
        io: &CommitPublicValuesEvent<BabyBear>,
        digest_idx: usize,
        cols: &mut PublicValuesCols<BabyBear>,
    );
    pub fn public_values_instr_to_row_koalabear(
        instr: &CommitPublicValuesInstr<BabyBear>,
        digest_idx: usize,
        cols: &mut PublicValuesPreprocessedCols<BabyBear>,
    );

    pub fn select_event_to_row_koalabear(
        io: &SelectEvent<BabyBear>,
        cols: &mut SelectCols<BabyBear>,
    );
    pub fn select_instr_to_row_koalabear(
        instr: &SelectInstr<BabyBear>,
        cols: &mut SelectPreprocessedCols<BabyBear>,
    );

    pub fn poseidon2_skinny_event_to_row_koalabear(io: &Poseidon2Event<BabyBear>, cols: *mut u8);

    pub fn poseidon2_wide_event_to_row_koalabear(
        input: *const BabyBear,
        input_row: *mut BabyBear,
        sbox_state: bool,
    );
    pub fn poseidon2_wide_instr_to_row_koalabear(instr: &Poseidon2Instr<BabyBear>, cols: *mut u8);

    pub fn poly_eval_event_to_row_koalabear(
        io: &PolyEvalEventFFI<BabyBear>,
        i: usize,
        cols: &mut PolyEvalCols<BabyBear>,
    );
    pub fn poly_eval_instr_to_row_koalabear(
        instr: &PolyEvalInstrFFI<BabyBear>,
        i: usize,
        len: usize,
        cols: &mut PolyEvalPreprocessedCols<BabyBear>,
    );

    pub fn ext_exp_reverse_bits_event_to_row_koalabear(
        io: &ExtExpReverseBitsEventFFI<BabyBear>,
        i: usize,
        cols: &mut ExtExpReverseBitsCols<BabyBear>,
    );
    pub fn ext_exp_reverse_bits_instr_to_row_koalabear(
        instr: &ExtExpReverseBitsInstrFFI<BabyBear>,
        i: usize,
        len: usize,
        cols: &mut ExtExpReverseBitsPreprocessedCols<BabyBear>,
    );

}

#[cfg(not(feature = "sys"))]
mod fallback {
    use super::*;
    use crate::chips::{
        mem::MemoryAccessColsChips,
        poseidon2_skinny::{
            columns::Poseidon2 as Poseidon2SkinnyCols,
            external_linear_layer as baby_external_linear_layer,
            internal_linear_layer as baby_internal_linear_layer, NUM_EXTERNAL_ROUNDS as BB_EXT,
            NUM_INTERNAL_ROUNDS as BB_INT, NUM_ROUNDS as BB_ROUNDS, WIDTH as BB_WIDTH,
        },
        poseidon2_skinny_kb::{
            columns::Poseidon2 as Poseidon2SkinnyKbCols,
            external_linear_layer as koala_external_linear_layer,
            internal_linear_layer as koala_internal_linear_layer,
            NUM_INTERNAL_ROUNDS as KB_INT, ROWS_PER_PERMUTE as KB_ROWS, WIDTH as KB_WIDTH,
        },
        poseidon2_wide::columns::preprocessed::Poseidon2PreprocessedColsWide,
        poseidon2_wide_kb::columns::preprocessed::Poseidon2PreprocessedColsWideKb,
    };
    use dt_core_machine::operations::{
        poseidon2::{
            permutation::{NUM_POSEIDON2_DEGREE3_COLS, NUM_POSEIDON2_DEGREE9_COLS},
            trace as baby_poseidon2_trace, WIDTH,
        },
        poseidon2_kb::{
            permutation::NUM_POSEIDON2_DEGREE3_COLS as NUM_KB_POSEIDON2_DEGREE3_COLS,
            trace as koala_poseidon2_trace,
        },
    };
    use dt_primitives::{
        KoalaBear_BEGIN_EXT_CONSTS, KoalaBear_END_EXT_CONSTS, KoalaBear_PARTIAL_CONSTS,
        RC_16_30_U32,
    };
    use p3_field::{AbstractField, PrimeField32};
    use p3_koala_bear::KoalaBear;

    pub unsafe fn alu_base_event_to_row_babybear(
        io: &BaseAluIo<BabyBear>,
        cols: &mut BaseAluValueCols<BabyBear>,
    ) {
        cols.vals = *io;
    }

    pub unsafe fn alu_base_instr_to_row_babybear(
        instr: &BaseAluInstr<BabyBear>,
        cols: &mut BaseAluAccessCols<BabyBear>,
    ) {
        cols.addrs = instr.addrs;
        cols.is_add = BabyBear::from_bool(matches!(instr.opcode, BaseAluOpcode::AddF));
        cols.is_sub = BabyBear::from_bool(matches!(instr.opcode, BaseAluOpcode::SubF));
        cols.is_mul = BabyBear::from_bool(matches!(instr.opcode, BaseAluOpcode::MulF));
        cols.is_div = BabyBear::from_bool(matches!(instr.opcode, BaseAluOpcode::DivF));
        cols.mult = instr.mult;
    }

    pub unsafe fn alu_ext_event_to_row_babybear(
        io: &ExtAluIo<Block<BabyBear>>,
        cols: &mut ExtAluValueCols<BabyBear>,
    ) {
        cols.vals = *io;
    }

    pub unsafe fn alu_ext_instr_to_row_babybear(
        instr: &ExtAluInstr<BabyBear>,
        cols: &mut ExtAluAccessCols<BabyBear>,
    ) {
        cols.addrs = instr.addrs;
        cols.is_add = BabyBear::from_bool(matches!(instr.opcode, ExtAluOpcode::AddE));
        cols.is_sub = BabyBear::from_bool(matches!(instr.opcode, ExtAluOpcode::SubE));
        cols.is_mul = BabyBear::from_bool(matches!(instr.opcode, ExtAluOpcode::MulE));
        cols.is_div = BabyBear::from_bool(matches!(instr.opcode, ExtAluOpcode::DivE));
        cols.mult = instr.mult;
    }

    pub unsafe fn public_values_event_to_row_babybear(
        io: &CommitPublicValuesEvent<BabyBear>,
        digest_idx: usize,
        cols: &mut PublicValuesCols<BabyBear>,
    ) {
        cols.pv_element = io.public_values.digest[digest_idx];
    }

    pub unsafe fn public_values_instr_to_row_babybear(
        instr: &CommitPublicValuesInstr<BabyBear>,
        digest_idx: usize,
        cols: &mut PublicValuesPreprocessedCols<BabyBear>,
    ) {
        cols.pv_idx[digest_idx] = BabyBear::one();
        cols.pv_mem = MemoryAccessColsChips {
            addr: instr.pv_addrs.digest[digest_idx],
            mult: -BabyBear::one(),
        };
    }

    pub unsafe fn select_event_to_row_babybear(
        io: &SelectEvent<BabyBear>,
        cols: &mut SelectCols<BabyBear>,
    ) {
        cols.vals = *io;
    }

    pub unsafe fn select_instr_to_row_babybear(
        instr: &SelectInstr<BabyBear>,
        cols: &mut SelectPreprocessedCols<BabyBear>,
    ) {
        cols.is_real = BabyBear::one();
        cols.addrs = instr.addrs;
        cols.mult1 = instr.mult1;
        cols.mult2 = instr.mult2;
    }

    pub unsafe fn poseidon2_wide_event_to_row_babybear(
        input: *const BabyBear,
        input_row: *mut BabyBear,
        degree_3: bool,
    ) {
        let input = *(input as *const [BabyBear; WIDTH]);
        if degree_3 {
            let row =
                std::slice::from_raw_parts_mut(input_row, NUM_POSEIDON2_DEGREE3_COLS);
            baby_poseidon2_trace::populate_perm::<BabyBear, 3>(input, None, row);
        } else {
            let row =
                std::slice::from_raw_parts_mut(input_row, NUM_POSEIDON2_DEGREE9_COLS);
            baby_poseidon2_trace::populate_perm::<BabyBear, 9>(input, None, row);
        }
    }

    pub unsafe fn poseidon2_wide_instr_to_row_babybear(
        instr: &Poseidon2Instr<BabyBear>,
        cols: &mut Poseidon2PreprocessedColsWide<BabyBear>,
    ) {
        cols.input = instr.addrs.input;
        for i in 0..WIDTH {
            cols.output[i] = MemoryAccessColsChips {
                addr: instr.addrs.output[i],
                mult: instr.mults[i],
            };
        }
        cols.is_real_neg = -BabyBear::one();
    }

    pub unsafe fn alu_base_event_to_row_koalabear(
        io: &BaseAluIo<BabyBear>,
        cols: &mut BaseAluValueCols<BabyBear>,
    ) {
        unsafe { alu_base_event_to_row_babybear(io, cols) };
    }

    pub unsafe fn alu_base_instr_to_row_koalabear(
        instr: &BaseAluInstr<BabyBear>,
        cols: &mut BaseAluAccessCols<BabyBear>,
    ) {
        unsafe { alu_base_instr_to_row_babybear(instr, cols) };
    }

    pub unsafe fn alu_ext_event_to_row_koalabear(
        io: &ExtAluIo<Block<BabyBear>>,
        cols: &mut ExtAluValueCols<BabyBear>,
    ) {
        unsafe { alu_ext_event_to_row_babybear(io, cols) };
    }

    pub unsafe fn alu_ext_instr_to_row_koalabear(
        instr: &ExtAluInstr<BabyBear>,
        cols: &mut ExtAluAccessCols<BabyBear>,
    ) {
        unsafe { alu_ext_instr_to_row_babybear(instr, cols) };
    }

    pub unsafe fn public_values_event_to_row_koalabear(
        io: &CommitPublicValuesEvent<BabyBear>,
        digest_idx: usize,
        cols: &mut PublicValuesCols<BabyBear>,
    ) {
        unsafe { public_values_event_to_row_babybear(io, digest_idx, cols) };
    }

    pub unsafe fn public_values_instr_to_row_koalabear(
        instr: &CommitPublicValuesInstr<BabyBear>,
        digest_idx: usize,
        cols: &mut PublicValuesPreprocessedCols<BabyBear>,
    ) {
        unsafe { public_values_instr_to_row_babybear(instr, digest_idx, cols) };
    }

    pub unsafe fn select_event_to_row_koalabear(
        io: &SelectEvent<BabyBear>,
        cols: &mut SelectCols<BabyBear>,
    ) {
        unsafe { select_event_to_row_babybear(io, cols) };
    }

    pub unsafe fn select_instr_to_row_koalabear(
        instr: &SelectInstr<BabyBear>,
        cols: &mut SelectPreprocessedCols<BabyBear>,
    ) {
        unsafe { select_instr_to_row_babybear(instr, cols) };
    }

    pub unsafe fn poseidon2_wide_event_to_row_koalabear(
        input: *const BabyBear,
        input_row: *mut BabyBear,
        _degree_3: bool,
    ) {
        let input = *(input as *const [KoalaBear; WIDTH]);
        let row = std::slice::from_raw_parts_mut(
            input_row as *mut KoalaBear,
            NUM_KB_POSEIDON2_DEGREE3_COLS,
        );
        koala_poseidon2_trace::populate_perm::<KoalaBear, 3>(input, None, row);
    }

    pub unsafe fn poseidon2_wide_instr_to_row_koalabear(
        instr: &Poseidon2Instr<BabyBear>,
        cols: *mut u8,
    ) {
        let instr = &*(instr as *const Poseidon2Instr<BabyBear> as *const Poseidon2Instr<KoalaBear>);
        let cols = &mut *(cols as *mut Poseidon2PreprocessedColsWideKb<KoalaBear>);
        cols.input = instr.addrs.input;
        for i in 0..WIDTH {
            cols.output[i] = MemoryAccessColsChips {
                addr: instr.addrs.output[i],
                mult: instr.mults[i],
            };
        }
        cols.is_real_neg = -KoalaBear::one();
    }

    pub unsafe fn poseidon2_skinny_event_to_row_babybear(
        io: &Poseidon2Event<BabyBear>,
        cols: *mut u8,
    ) {
        let rows = std::slice::from_raw_parts_mut(
            cols as *mut Poseidon2SkinnyCols<BabyBear>,
            BB_ROUNDS,
        );
        populate_poseidon2_skinny_babybear(io.input, rows);
    }

    pub unsafe fn poseidon2_skinny_event_to_row_koalabear(
        io: &Poseidon2Event<BabyBear>,
        cols: *mut u8,
    ) {
        let io = &*(io as *const Poseidon2Event<BabyBear> as *const Poseidon2Event<KoalaBear>);
        let rows = std::slice::from_raw_parts_mut(
            cols as *mut Poseidon2SkinnyKbCols<KoalaBear>,
            KB_ROWS,
        );
        populate_poseidon2_skinny_koalabear(io.input, rows);
    }

    fn populate_poseidon2_skinny_babybear(
        input: [BabyBear; BB_WIDTH],
        rows: &mut [Poseidon2SkinnyCols<BabyBear>],
    ) {
        let mut state = input;
        let half_ext = BB_EXT / 2;

        for (r, row) in rows.iter_mut().enumerate() {
            row.state_in = state;
            if r < half_ext || r >= half_ext + BB_INT {
                let mut round_input = state;
                if r == 0 {
                    baby_external_linear_layer(&mut round_input);
                }
                for i in 0..BB_WIDTH {
                    round_input[i] += BabyBear::from_wrapped_u32(RC_16_30_U32[r][i]);
                    let x3 = round_input[i] * round_input[i] * round_input[i];
                    round_input[i] = x3 * x3 * round_input[i];
                }
                baby_external_linear_layer(&mut round_input);
                state = round_input;
            } else {
                let mut round_input = state;
                round_input[0] += BabyBear::from_wrapped_u32(RC_16_30_U32[r][0]);
                let x3 = round_input[0] * round_input[0] * round_input[0];
                round_input[0] = x3 * x3 * round_input[0];
                baby_internal_linear_layer(&mut round_input);
                state = round_input;
            }
            row.state_out = state;
        }
    }

    fn populate_poseidon2_skinny_koalabear(
        input: [KoalaBear; KB_WIDTH],
        rows: &mut [Poseidon2SkinnyKbCols<KoalaBear>],
    ) {
        let mut state = input;
        let external_pairs = [
            (0usize, true, 0usize),
            (1usize, true, 2usize),
            (3usize, false, 0usize),
            (4usize, false, 2usize),
        ];

        for &(row_idx, first_half, table_idx) in &external_pairs {
            let row = &mut rows[row_idx];
            row.state_in = state;

            let mut first_input = state;
            if row_idx == 0 {
                koala_external_linear_layer(&mut first_input);
            }
            for i in 0..KB_WIDTH {
                let c = if first_half {
                    KoalaBear_BEGIN_EXT_CONSTS[table_idx][i]
                } else {
                    KoalaBear_END_EXT_CONSTS[table_idx][i]
                };
                first_input[i] += KoalaBear::from_canonical_u32(c.as_canonical_u32());
                first_input[i] = first_input[i] * first_input[i] * first_input[i];
            }
            koala_external_linear_layer(&mut first_input);
            row.round_witness[..KB_WIDTH].copy_from_slice(&first_input);

            let mut second_input = first_input;
            for i in 0..KB_WIDTH {
                let c = if first_half {
                    KoalaBear_BEGIN_EXT_CONSTS[table_idx + 1][i]
                } else {
                    KoalaBear_END_EXT_CONSTS[table_idx + 1][i]
                };
                second_input[i] += KoalaBear::from_canonical_u32(c.as_canonical_u32());
                second_input[i] = second_input[i] * second_input[i] * second_input[i];
            }
            koala_external_linear_layer(&mut second_input);
            state = second_input;
            row.state_out = state;
        }

        let row = &mut rows[2];
        row.state_in = state;
        for k in 0..(KB_INT - 1) {
            let c = KoalaBear_PARTIAL_CONSTS[k];
            let sbox_in = state[0] + KoalaBear::from_canonical_u32(c.as_canonical_u32());
            state[0] = sbox_in * sbox_in * sbox_in;
            row.round_witness[k] = state[0];
            koala_internal_linear_layer(&mut state);
        }
        let c = KoalaBear_PARTIAL_CONSTS[KB_INT - 1];
        let sbox_in = state[0] + KoalaBear::from_canonical_u32(c.as_canonical_u32());
        state[0] = sbox_in * sbox_in * sbox_in;
        koala_internal_linear_layer(&mut state);
        row.state_out = state;
    }
}
