#include "bb31_t.hpp"
#include "kb31_t.hpp"
#include "sys.hpp"

namespace dt_recursion_core_sys {
using namespace poseidon2;

extern void alu_base_event_to_row_babybear(const BaseAluIo<BabyBearP3>* io,
                                           BaseAluValueCols<BabyBearP3>* cols) {
  alu_base::event_to_row<bb31_t>(
      *reinterpret_cast<const BaseAluIo<bb31_t>*>(io),
      *reinterpret_cast<BaseAluValueCols<bb31_t>*>(cols));
}
extern void alu_base_instr_to_row_babybear(
    const BaseAluInstr<BabyBearP3>* instr,
    BaseAluAccessCols<BabyBearP3>* access) {
  alu_base::instr_to_row<bb31_t>(
      *reinterpret_cast<const BaseAluInstr<bb31_t>*>(instr),
      *reinterpret_cast<BaseAluAccessCols<bb31_t>*>(access));
}

extern void alu_ext_event_to_row_babybear(const ExtAluIo<Block<BabyBearP3>>* io,
                                          ExtAluValueCols<BabyBearP3>* cols) {
  alu_ext::event_to_row<bb31_t>(
      *reinterpret_cast<const ExtAluIo<Block<bb31_t>>*>(io),
      *reinterpret_cast<ExtAluValueCols<bb31_t>*>(cols));
}
extern void alu_ext_instr_to_row_babybear(
    const ExtAluInstr<BabyBearP3>* instr,
    ExtAluAccessCols<BabyBearP3>* access) {
  alu_ext::instr_to_row<bb31_t>(
      *reinterpret_cast<const ExtAluInstr<bb31_t>*>(instr),
      *reinterpret_cast<ExtAluAccessCols<bb31_t>*>(access));
}

extern void public_values_event_to_row_babybear(
    const CommitPublicValuesEvent<BabyBearP3>* io, size_t digest_idx,
    PublicValuesCols<BabyBearP3>* cols) {
  public_values::event_to_row<bb31_t>(
      *reinterpret_cast<const CommitPublicValuesEvent<bb31_t>*>(io), digest_idx,
      *reinterpret_cast<PublicValuesCols<bb31_t>*>(cols));
}
extern void public_values_instr_to_row_babybear(
    const CommitPublicValuesInstr<BabyBearP3>* instr, size_t digest_idx,
    PublicValuesPreprocessedCols<BabyBearP3>* cols) {
  public_values::instr_to_row<bb31_t>(
      *reinterpret_cast<const CommitPublicValuesInstr<bb31_t>*>(instr),
      digest_idx,
      *reinterpret_cast<PublicValuesPreprocessedCols<bb31_t>*>(cols));
}

extern void select_event_to_row_babybear(const SelectEvent<BabyBearP3>* io,
                                         SelectCols<BabyBearP3>* cols) {
  select::event_to_row<bb31_t>(
      *reinterpret_cast<const SelectEvent<bb31_t>*>(io),
      *reinterpret_cast<SelectCols<bb31_t>*>(cols));
}
extern void select_instr_to_row_babybear(
    const SelectInstr<BabyBearP3>* instr,
    SelectPreprocessedCols<BabyBearP3>* cols) {
  select::instr_to_row<bb31_t>(
      *reinterpret_cast<const SelectInstr<bb31_t>*>(instr),
      *reinterpret_cast<SelectPreprocessedCols<bb31_t>*>(cols));
}

extern void poseidon2_skinny_event_to_row_babybear(
    const Poseidon2Event<BabyBearP3>* event,
    uint8_t* cols) {
  poseidon2_skinny::event_to_row<bb31_t>(
      *reinterpret_cast<const Poseidon2Event<bb31_t>*>(event),
      reinterpret_cast<bb31_t*>(cols));
}

extern "C" void poseidon2_wide_event_to_row_babybear(const BabyBearP3* input,
                                                     BabyBearP3* input_row,
                                                     bool sbox_state) {
  poseidon2_wide::event_to_row<bb31_t>(reinterpret_cast<const bb31_t*>(input),
                                       reinterpret_cast<bb31_t*>(input_row), 0,
                                       1, sbox_state);
}
extern void poseidon2_wide_instr_to_row_babybear(
    const Poseidon2WideInstr<BabyBearP3>* instr,
    Poseidon2PreprocessedColsWide<BabyBearP3>* cols) {
  poseidon2_wide::instr_to_row<bb31_t>(
      *reinterpret_cast<const Poseidon2WideInstr<bb31_t>*>(instr),
      *reinterpret_cast<Poseidon2PreprocessedColsWide<bb31_t>*>(cols));
}

extern void poly_eval_event_to_row_babybear(
    const PolyEvalEventFFI<BabyBearP3>* io, size_t i,
    PolyEvalCols<BabyBearP3>* cols) {
  poly_eval::event_to_row<bb31_t>(
      *reinterpret_cast<const PolyEvalEventFFI<bb31_t>*>(io), i,
      *reinterpret_cast<PolyEvalCols<bb31_t>*>(cols));
}
extern void poly_eval_instr_to_row_babybear(
    const PolyEvalInstrFFI<BabyBearP3>* instr, size_t i, size_t len,
    PolyEvalPreprocessedCols<BabyBearP3>* cols) {
  poly_eval::instr_to_row<bb31_t>(
      *reinterpret_cast<const PolyEvalInstrFFI<bb31_t>*>(instr), i, len,
      *reinterpret_cast<PolyEvalPreprocessedCols<bb31_t>*>(cols));
}

extern void ext_exp_reverse_bits_event_to_row_babybear(
    const ExtExpReverseBitsEventFFI<BabyBearP3>* io, size_t i,
    ExtExpReverseBitsCols<BabyBearP3>* cols) {
  ext_exp_reverse_bits::event_to_row<bb31_t>(
      *reinterpret_cast<const ExtExpReverseBitsEventFFI<bb31_t>*>(io), i,
      *reinterpret_cast<ExtExpReverseBitsCols<bb31_t>*>(cols));
}
extern void ext_exp_reverse_bits_instr_to_row_babybear(
    const ExtExpReverseBitsInstrFFI<BabyBearP3>* instr, size_t i, size_t len,
    ExtExpReverseBitsPreprocessedCols<BabyBearP3>* cols) {
  ext_exp_reverse_bits::instr_to_row<bb31_t>(
      *reinterpret_cast<const ExtExpReverseBitsInstrFFI<bb31_t>*>(instr), i, len,
      *reinterpret_cast<ExtExpReverseBitsPreprocessedCols<bb31_t>*>(cols));
}

// ======================== KoalaBear FFI functions ========================
// These mirror the BabyBear versions above but use kb31_t for modular arithmetic.
// BabyBearP3 and KoalaBear have identical memory layout (u32 wrapper), so we
// reuse the cbindgen-generated struct definitions and reinterpret_cast to kb31_t.

extern void alu_base_event_to_row_koalabear(const BaseAluIo<BabyBearP3>* io,
                                            BaseAluValueCols<BabyBearP3>* cols) {
  alu_base::event_to_row<kb31_t>(
      *reinterpret_cast<const BaseAluIo<kb31_t>*>(io),
      *reinterpret_cast<BaseAluValueCols<kb31_t>*>(cols));
}
extern void alu_base_instr_to_row_koalabear(
    const BaseAluInstr<BabyBearP3>* instr,
    BaseAluAccessCols<BabyBearP3>* access) {
  alu_base::instr_to_row<kb31_t>(
      *reinterpret_cast<const BaseAluInstr<kb31_t>*>(instr),
      *reinterpret_cast<BaseAluAccessCols<kb31_t>*>(access));
}

extern void alu_ext_event_to_row_koalabear(const ExtAluIo<Block<BabyBearP3>>* io,
                                           ExtAluValueCols<BabyBearP3>* cols) {
  alu_ext::event_to_row<kb31_t>(
      *reinterpret_cast<const ExtAluIo<Block<kb31_t>>*>(io),
      *reinterpret_cast<ExtAluValueCols<kb31_t>*>(cols));
}
extern void alu_ext_instr_to_row_koalabear(
    const ExtAluInstr<BabyBearP3>* instr,
    ExtAluAccessCols<BabyBearP3>* access) {
  alu_ext::instr_to_row<kb31_t>(
      *reinterpret_cast<const ExtAluInstr<kb31_t>*>(instr),
      *reinterpret_cast<ExtAluAccessCols<kb31_t>*>(access));
}

extern void public_values_event_to_row_koalabear(
    const CommitPublicValuesEvent<BabyBearP3>* io, size_t digest_idx,
    PublicValuesCols<BabyBearP3>* cols) {
  public_values::event_to_row<kb31_t>(
      *reinterpret_cast<const CommitPublicValuesEvent<kb31_t>*>(io), digest_idx,
      *reinterpret_cast<PublicValuesCols<kb31_t>*>(cols));
}
extern void public_values_instr_to_row_koalabear(
    const CommitPublicValuesInstr<BabyBearP3>* instr, size_t digest_idx,
    PublicValuesPreprocessedCols<BabyBearP3>* cols) {
  public_values::instr_to_row<kb31_t>(
      *reinterpret_cast<const CommitPublicValuesInstr<kb31_t>*>(instr),
      digest_idx,
      *reinterpret_cast<PublicValuesPreprocessedCols<kb31_t>*>(cols));
}

extern void select_event_to_row_koalabear(const SelectEvent<BabyBearP3>* io,
                                          SelectCols<BabyBearP3>* cols) {
  select::event_to_row<kb31_t>(
      *reinterpret_cast<const SelectEvent<kb31_t>*>(io),
      *reinterpret_cast<SelectCols<kb31_t>*>(cols));
}
extern void select_instr_to_row_koalabear(
    const SelectInstr<BabyBearP3>* instr,
    SelectPreprocessedCols<BabyBearP3>* cols) {
  select::instr_to_row<kb31_t>(
      *reinterpret_cast<const SelectInstr<kb31_t>*>(instr),
      *reinterpret_cast<SelectPreprocessedCols<kb31_t>*>(cols));
}

extern void poseidon2_skinny_event_to_row_koalabear(
    const Poseidon2Event<BabyBearP3>* event,
    uint8_t* cols) {
  // KoalaBear uses the "9-row per permutation" layout (poseidon2_skinny_kb), distinct from
  // BabyBear's classical one-round-per-row layout (poseidon2_skinny).
  poseidon2_skinny_kb::event_to_row<kb31_t>(
      *reinterpret_cast<const Poseidon2Event<kb31_t>*>(event),
      reinterpret_cast<kb31_t*>(cols));
}
extern "C" void poseidon2_wide_event_to_row_koalabear(const BabyBearP3* input,
                                                      BabyBearP3* input_row,
                                                      bool sbox_state) {
  poseidon2_wide::event_to_row<kb31_t>(reinterpret_cast<const kb31_t*>(input),
                                       reinterpret_cast<kb31_t*>(input_row), 0,
                                       1, sbox_state);
}
extern void poseidon2_wide_instr_to_row_koalabear(
    const Poseidon2WideInstr<BabyBearP3>* instr,
    uint8_t* cols) {
  poseidon2_wide::instr_to_row<kb31_t>(
      *reinterpret_cast<const Poseidon2WideInstr<kb31_t>*>(instr),
      *reinterpret_cast<Poseidon2PreprocessedColsWide<kb31_t>*>(cols));
}

extern void poly_eval_event_to_row_koalabear(
    const PolyEvalEventFFI<BabyBearP3>* io, size_t i,
    PolyEvalCols<BabyBearP3>* cols) {
  poly_eval::event_to_row<kb31_t>(
      *reinterpret_cast<const PolyEvalEventFFI<kb31_t>*>(io), i,
      *reinterpret_cast<PolyEvalCols<kb31_t>*>(cols));
}
extern void poly_eval_instr_to_row_koalabear(
    const PolyEvalInstrFFI<BabyBearP3>* instr, size_t i, size_t len,
    PolyEvalPreprocessedCols<BabyBearP3>* cols) {
  poly_eval::instr_to_row<kb31_t>(
      *reinterpret_cast<const PolyEvalInstrFFI<kb31_t>*>(instr), i, len,
      *reinterpret_cast<PolyEvalPreprocessedCols<kb31_t>*>(cols));
}

extern void ext_exp_reverse_bits_event_to_row_koalabear(
    const ExtExpReverseBitsEventFFI<BabyBearP3>* io, size_t i,
    ExtExpReverseBitsCols<BabyBearP3>* cols) {
  ext_exp_reverse_bits::event_to_row<kb31_t>(
      *reinterpret_cast<const ExtExpReverseBitsEventFFI<kb31_t>*>(io), i,
      *reinterpret_cast<ExtExpReverseBitsCols<kb31_t>*>(cols));
}
extern void ext_exp_reverse_bits_instr_to_row_koalabear(
    const ExtExpReverseBitsInstrFFI<BabyBearP3>* instr, size_t i, size_t len,
    ExtExpReverseBitsPreprocessedCols<BabyBearP3>* cols) {
  ext_exp_reverse_bits::instr_to_row<kb31_t>(
      *reinterpret_cast<const ExtExpReverseBitsInstrFFI<kb31_t>*>(instr), i, len,
      *reinterpret_cast<ExtExpReverseBitsPreprocessedCols<kb31_t>*>(cols));
}

}  // namespace dt_recursion_core_sys
