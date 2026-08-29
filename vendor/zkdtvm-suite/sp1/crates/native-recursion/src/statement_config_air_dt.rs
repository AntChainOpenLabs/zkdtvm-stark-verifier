use core::{
    borrow::{Borrow, BorrowMut},
    ops::Deref,
};

use dt_stark::{
    air::{FullAir, FullAirBuilder, MachineAir, PairCol},
    sumcheck::trace::{CompressedMatrix, PaddingRow},
};
use native_recursion_derive::AlignedBorrow;
use p3_air::BaseAir;
use p3_field::{AbstractField, Field};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use crate::{
    config::{DIGEST_SIZE, F},
    interaction_full_air_dt::RecursionFullAirBus,
    interaction_registry_dt::STATEMENT_CONFIG_SCHEMA,
    statement_dt::resolve_child_vk_class_with_memo,
    system_dt::{RecursionNativeProgram, RecursionRecord, StatementConfigRow},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StatementConfigBus {
    bus: RecursionFullAirBus,
}

impl StatementConfigBus {
    pub const fn new() -> Self {
        Self { bus: RecursionFullAirBus::new(STATEMENT_CONFIG_SCHEMA) }
    }

    pub const fn required_max_beta_power_floor(&self) -> usize {
        self.bus.required_max_beta_power_floor()
    }

    pub fn denominator<AB>(
        &self,
        builder: &AB,
        class_id: AB::VarMaybeExt,
        digest: [AB::VarMaybeExt; DIGEST_SIZE],
    ) -> AB::VarExt
    where
        AB: FullAirBuilder,
    {
        let mut values = Vec::with_capacity(1 + DIGEST_SIZE);
        values.push(class_id);
        values.extend(digest);
        self.bus.denominator(builder, values)
    }
}

impl Default for StatementConfigBus {
    fn default() -> Self {
        Self::new()
    }
}

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct StatementConfigPreprocessedCols<T> {
    pub class_id: T,
    pub digest: [T; DIGEST_SIZE],
    pub is_row: T,
}

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct StatementConfigCols<T> {
    pub mult: T,
}

pub const NUM_STATEMENT_CONFIG_PREPROCESSED_COLS: usize =
    StatementConfigPreprocessedCols::<u8>::width();
pub const NUM_STATEMENT_CONFIG_COLS: usize = StatementConfigCols::<u8>::width();

/// Preprocessed vk-candidate table: one row per accepted baked child vk class of this
/// machine's statement role. Sends `[class_id, digest[8]]` on global bus 12 with a demand-count
/// witness mult; soundness comes from the consumer's forced recv mult (StatementAir membership).
#[derive(Debug, Clone)]
pub struct StatementConfigAir {
    pub bus: StatementConfigBus,
    pub rows: Vec<StatementConfigRow>,
}

impl StatementConfigAir {
    pub fn new(rows: Vec<StatementConfigRow>) -> Self {
        Self { bus: StatementConfigBus::new(), rows }
    }

    fn trace_height(&self) -> usize {
        // The prove path indexes eq coefficients by log_height - 1, so height 1 chips are
        // unsupported; an empty (lift) table still pads to two rows.
        self.rows.len().max(2).next_power_of_two()
    }
}

impl<Fld: Field> BaseAir<Fld> for StatementConfigAir {
    fn width(&self) -> usize {
        NUM_STATEMENT_CONFIG_COLS
    }
}

impl<AB: FullAirBuilder> FullAir<AB> for StatementConfigAir {
    fn width(&self) -> usize {
        NUM_STATEMENT_CONFIG_COLS
    }

    fn required_max_beta_power(&self) -> usize {
        self.bus.required_max_beta_power_floor()
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        (0..NUM_STATEMENT_CONFIG_PREPROCESSED_COLS)
            .map(PairCol::Prep)
            .chain((0..NUM_STATEMENT_CONFIG_COLS).map(PairCol::Main))
            .collect()
    }

    fn precompute_lc(&self, builder: &mut AB) {
        let denominator = {
            let prep = builder.preprocessed();
            let local: &StatementConfigPreprocessedCols<AB::VarMaybeExt> = prep.borrow();
            self.bus.denominator(builder, local.class_id.clone(), local.digest.clone())
        };
        builder.retain_precomputed(denominator);
    }

    fn eval(&self, builder: &mut AB) {
        // A padding row (is_row == 0) must never serve a membership recv.
        let reserved = builder.reserved_poly();
        let local_binding = reserved.row_slice(0);
        let local: &[AB::VarMaybeExt] = local_binding.deref();
        let is_row = local[NUM_STATEMENT_CONFIG_PREPROCESSED_COLS - 1].clone();
        let mult = local[NUM_STATEMENT_CONFIG_PREPROCESSED_COLS].clone();
        builder.assert_zero(mult * (AB::one_maybe() - is_row));
    }

    fn lookup(&self, builder: &mut AB) {
        let reserved = builder.reserved_poly();
        let local_binding = reserved.row_slice(0);
        let local: &[AB::VarMaybeExt] = local_binding.deref();
        // Order matches precompute_lc: StatementConfig.
        builder.send(local[NUM_STATEMENT_CONFIG_PREPROCESSED_COLS].clone());
    }
}

impl MachineAir<F> for StatementConfigAir {
    type Record = RecursionRecord;
    type Program = RecursionNativeProgram<F>;

    fn name(&self) -> String {
        "NativeStatementConfig".to_string()
    }

    fn preprocessed_width(&self) -> usize {
        NUM_STATEMENT_CONFIG_PREPROCESSED_COLS
    }

    fn preprocessed_num_rows(&self, _program: &Self::Program, _instrs_len: usize) -> Option<usize> {
        Some(self.trace_height())
    }

    fn generate_preprocessed_trace(&self, program: &Self::Program) -> Option<CompressedMatrix<F>> {
        assert_eq!(
            program.statement_config, self.rows,
            "StatementConfigAir rows drifted from the program statement_config"
        );
        let height = self.trace_height();
        let mut values =
            vec![F::zero(); NUM_STATEMENT_CONFIG_PREPROCESSED_COLS * self.rows.len().max(1)];
        for (row_idx, row) in self.rows.iter().enumerate() {
            let cols: &mut StatementConfigPreprocessedCols<F> = values[row_idx *
                NUM_STATEMENT_CONFIG_PREPROCESSED_COLS..
                (row_idx + 1) * NUM_STATEMENT_CONFIG_PREPROCESSED_COLS]
                .borrow_mut();
            cols.class_id = F::from_canonical_usize(row.class_id);
            cols.digest = row.digest;
            cols.is_row = F::one();
        }
        let main = RowMajorMatrix::new(values, NUM_STATEMENT_CONFIG_PREPROCESSED_COLS);
        Some(CompressedMatrix::new(
            main,
            PaddingRow::General(vec![F::zero(); NUM_STATEMENT_CONFIG_PREPROCESSED_COLS]),
            height,
        ))
    }

    fn num_rows(&self, _input: &Self::Record) -> Option<usize> {
        Some(self.trace_height())
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        _output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        let mut mults = vec![0u32; self.rows.len().max(1)];
        for proof in &input.proof_records {
            if proof.proof_shape.is_empty() {
                continue;
            }
            if let Ok(crate::statement_dt::ChildVkClass::Baked(row_idx)) =
                resolve_child_vk_class_with_memo(
                    proof,
                    input.statement_vk_root,
                    &self.rows,
                    &input.poseidon2_memo,
                )
            {
                mults[row_idx] += 1;
            }
        }
        let height = self.trace_height();
        let mut values = vec![F::zero(); NUM_STATEMENT_CONFIG_COLS * self.rows.len().max(1)];
        for (row_idx, mult) in mults.iter().enumerate() {
            let cols: &mut StatementConfigCols<F> = values
                [row_idx * NUM_STATEMENT_CONFIG_COLS..(row_idx + 1) * NUM_STATEMENT_CONFIG_COLS]
                .borrow_mut();
            cols.mult = F::from_canonical_u32(*mult);
        }
        let main = RowMajorMatrix::new(values, NUM_STATEMENT_CONFIG_COLS);
        CompressedMatrix::new(
            main,
            PaddingRow::General(vec![F::zero(); NUM_STATEMENT_CONFIG_COLS]),
            height,
        )
    }

    fn included(&self, _record: &Self::Record) -> bool {
        true
    }

    fn local_only(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::D_EF;
    use polyair::Chip;

    #[test]
    fn symbolic_analysis() {
        let chip = Chip::<StatementConfigAir, F, D_EF>::new(StatementConfigAir::new(vec![
            StatementConfigRow { class_id: 0, digest: [F::one(); DIGEST_SIZE] },
        ]));
        assert_eq!(chip.num_lookup(), 1);
        assert!(chip.degree <= 3, "StatementConfigAir degree {}", chip.degree);
    }
}
