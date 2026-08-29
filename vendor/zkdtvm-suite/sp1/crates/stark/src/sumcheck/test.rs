use crate::{
    air::{
        AirInteraction, BaseAirBuilder, DTAirBuilder, InteractionScope, MachineAir, MachineProgram,
        DT_PROOF_NUM_PV_ELTS,
    },
    lookup::InteractionKind,
    opts::DTCoreOpts,
    sumcheck::trace::{CompressedMatrix, PaddingRow},
    MachineRecord,
};
use dt_derive::AlignedBorrow;
use hashbrown::HashMap;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use std::{
    borrow::{Borrow, BorrowMut},
    marker::PhantomData,
};

pub const DEFAULT_LOG_HEIGHT: usize = 10;
pub const DEFAULT_LOG_HEIGHT_THRESHOLD: usize = 6;
pub const DEFAULT_NUM_SKIP_ROUNDS: usize = 2;

/// Controls whether a chip sends, receives, or has no cross-table interactions.
#[derive(Default, Copy, Clone, PartialEq, Eq)]
pub enum InteractionMode {
    /// No interactions (no permutation trace).
    #[default]
    None,
    /// Send `(a, b)` with multiplicity `is_real`.
    Send,
    /// Receive `(a, b)` with multiplicity `is_real`.
    Receive,
}

/// A simple chip for testing the sumcheck protocol.
///
/// Constraints are local-only (no `when_transition` / next-row access).
/// Each row satisfies: when `is_add=1`, `a = b + c`; when `is_add=0`, `b = a + c`.
/// Additionally `d = a * b` when `is_add=1`, and `d = 1` when `is_add=0`.
///
/// When `interaction_mode` is `Send` or `Receive`, the chip also sends/receives
/// `(a, b)` with multiplicity `is_real`, generating a non-empty permutation trace.
#[derive(Default, Copy, Clone)]
pub struct SimpleAddChip {
    pub index: usize,
    pub log_height: Option<usize>,
    pub interaction_mode: InteractionMode,
    /// When true, padding rows use a non-zero pattern instead of all zeros.
    /// The padding row satisfies constraints: `is_add=0, is_real=0, a=1, b=3, c=2, d=0`.
    pub use_nonzero_padding: bool,
}

/// Columns: `a`, `b`, `c`, `is_add`, `d`, `is_real` (`is_real`: 1 = real row, 0 = padding row).
#[derive(AlignedBorrow, Default, Clone, Copy)]
#[repr(C)]
pub struct SimpleAddCols<T> {
    pub a: T,
    pub b: T,
    pub c: T,
    pub is_add: T,
    pub d: T,
    pub is_real: T,
}

// ---------------------------------------------------------------------------
// Dummy record / program (reused from test.rs pattern)
// ---------------------------------------------------------------------------

#[derive(Default, Clone)]
pub struct DummyRecord;

impl MachineRecord for DummyRecord {
    type Config = DTCoreOpts;

    fn stats(&self) -> HashMap<String, usize> {
        unimplemented!()
    }

    fn append(&mut self, _other: &mut Self) {}

    fn register_nonces(&mut self, _opts: &Self::Config) {}

    fn public_values<F: AbstractField>(&self) -> Vec<F> {
        vec![F::zero(); DT_PROOF_NUM_PV_ELTS]
    }
}

#[derive(Default)]
pub struct DummyProgram<F> {
    _phantom: PhantomData<F>,
}

impl<F: Field> MachineProgram<F> for DummyProgram<F> {
    fn pc_start(&self) -> F {
        F::zero()
    }
}

// ---------------------------------------------------------------------------
// MachineAir implementation
// ---------------------------------------------------------------------------

impl<F: Field> MachineAir<F> for SimpleAddChip {
    type Record = DummyRecord;
    type Program = DummyProgram<F>;

    fn name(&self) -> String {
        format!("SimpleAddChip_{}", self.index)
    }

    fn num_rows(&self, _input: &Self::Record) -> Option<usize> {
        let log_h = self.log_height.unwrap_or(DEFAULT_LOG_HEIGHT);
        Some(1 << log_h)
    }

    fn generate_trace(
        &self,
        _input: &Self::Record,
        _output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        use rand::{rngs::StdRng, Rng, SeedableRng};

        let height = <Self as MachineAir<F>>::num_rows(self, &DummyRecord).unwrap();
        let width = <Self as BaseAir<F>>::width(self);
        let modulus = 4u32;

        // Non-padding height = half total height + 1; the rest are padding rows.
        let main_height = height / 2 + 1;
        let num_padding = height - main_height;

        let seed = [42u8; 32];
        let mut rng = StdRng::from_seed(seed);

        let mut trace = RowMajorMatrix::new(vec![F::zero(); main_height * width], width);

        for i in 0..main_height {
            let row = trace.row_mut(i);
            let cols: &mut SimpleAddCols<F> = (*row).borrow_mut();
            cols.is_real = F::one();

            let is_add = rng.gen_range(0..2);
            cols.c = F::from_canonical_u32(rng.gen_range(0..modulus));

            if is_add == 1 {
                cols.is_add = F::one();
                cols.b = F::from_canonical_u32(rng.gen_range(0..modulus));
                cols.a = cols.b + cols.c;
                cols.d = cols.a * cols.b;
            } else {
                cols.is_add = F::zero();
                cols.a = F::from_canonical_u32(rng.gen_range(0..modulus));
                cols.b = cols.a + cols.c;
                cols.d = F::one();
            }
        }

        // Force first row: is_add = 0
        {
            let row = trace.row_mut(0);
            let cols: &mut SimpleAddCols<F> = (*row).borrow_mut();
            cols.is_add = F::zero();
            cols.c = F::from_canonical_u32(rng.gen_range(0..modulus));
            cols.a = F::from_canonical_u32(rng.gen_range(0..modulus));
            cols.b = cols.a + cols.c;
            cols.d = F::one();
        }

        // Force last real row: is_add = 1
        {
            let row = trace.row_mut(main_height - 1);
            let cols: &mut SimpleAddCols<F> = (*row).borrow_mut();
            cols.is_add = F::one();
            cols.c = F::from_canonical_u32(rng.gen_range(0..modulus));
            cols.b = F::from_canonical_u32(rng.gen_range(0..modulus));
            cols.a = cols.b + cols.c;
            cols.d = cols.a * cols.b;
        }

        let padding_row = if num_padding > 0 {
            if self.use_nonzero_padding {
                // Non-zero padding: a=1, b=3, c=2, is_add=0, d=0, is_real=0
                // Satisfies: is_add=0 → b = a + c (3 = 1 + 2), is_real=0 so d unconstrained.
                let mut row = vec![F::zero(); width];
                row[0] = F::one(); // a = 1
                row[1] = F::from_canonical_u32(3); // b = 3
                row[2] = F::from_canonical_u32(2); // c = 2
                                                   // is_add = 0, d = 0, is_real = 0 (already zero)
                PaddingRow::General(row)
            } else {
                PaddingRow::Zero { width }
            }
        } else {
            PaddingRow::None
        };

        CompressedMatrix::new(trace, padding_row, height)
    }

    fn generate_dependencies(&self, _input: &Self::Record, _output: &mut Self::Record) {}

    fn included(&self, _shard: &Self::Record) -> bool {
        true
    }

    fn preprocessed_width(&self) -> usize {
        if self.index & 1 == 0 {
            1
        } else {
            0
        }
    }

    fn preprocessed_num_rows(&self, _program: &Self::Program, _instrs_len: usize) -> Option<usize> {
        None
    }

    fn generate_preprocessed_trace(&self, _program: &Self::Program) -> Option<CompressedMatrix<F>> {
        if self.index & 1 == 0 {
            let height = <Self as MachineAir<F>>::num_rows(self, &DummyRecord).unwrap();
            let prep_width = <Self as MachineAir<F>>::preprocessed_width(self);
            // Same padding as main: non-padding height = half total + 1.
            let main_height = height / 2 + 1;
            let num_padding = height - main_height;
            let mat = RowMajorMatrix::new(vec![F::zero(); main_height * prep_width], prep_width);
            let padding_row = if num_padding > 0 {
                PaddingRow::Zero { width: prep_width }
            } else {
                PaddingRow::None
            };
            Some(CompressedMatrix::new(mat, padding_row, height))
        } else {
            None
        }
    }

    fn commit_scope(&self) -> InteractionScope {
        InteractionScope::Local
    }

    fn local_only(&self) -> bool {
        true
    }
}

// ---------------------------------------------------------------------------
// BaseAir
// ---------------------------------------------------------------------------

impl<F> BaseAir<F> for SimpleAddChip {
    fn width(&self) -> usize {
        std::mem::size_of::<SimpleAddCols<u8>>()
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        None
    }
}

// ---------------------------------------------------------------------------
// Air constraints — local-only, no when_transition
// ---------------------------------------------------------------------------

impl<AB> Air<AB> for SimpleAddChip
where
    AB: DTAirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &SimpleAddCols<AB::Var> = (*local).borrow();

        // is_add ∈ {0, 1}, is_real ∈ {0, 1} (1 = real row, 0 = padding)
        builder.assert_bool(local.is_add);
        builder.assert_bool(local.is_real);

        // when is_add=1: a = b + c
        let add_residual = local.b + local.c - local.a;
        builder.when(local.is_add).assert_zero(add_residual);

        // when is_add=0: b = a + c
        let sub_residual = local.a + local.c - local.b;
        builder.when_not(local.is_add).assert_zero(sub_residual);

        // is_add=0 on first row
        builder.when_first_row().assert_zero(local.is_add);

        // is_add=1 on last row
        builder.when_last_row().when(local.is_real).assert_one(local.is_add);

        // when is_add=1: d = a * b
        let mul_residual = local.a * local.b - local.d;
        builder.when(local.is_add).assert_zero(mul_residual);

        // when is_real=1 and is_add=0: d = 1
        let const_residual = local.d - AB::F::one();
        builder.when(local.is_real).when_not(local.is_add).assert_zero(const_residual);

        // Cross-table interactions for permutation trace generation.
        match self.interaction_mode {
            InteractionMode::Send => {
                builder.send(
                    AirInteraction::new(
                        vec![local.a.into(), local.b.into()],
                        local.is_real.into(),
                        InteractionKind::Alu,
                    ),
                    InteractionScope::Local,
                );
            }
            InteractionMode::Receive => {
                builder.receive(
                    AirInteraction::new(
                        vec![local.a.into(), local.b.into()],
                        local.is_real.into(),
                        InteractionKind::Alu,
                    ),
                    InteractionScope::Local,
                );
            }
            InteractionMode::None => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    #![allow(clippy::print_stdout)]

    use crate::{
        baby_bear_poseidon2::SCBabyBearPoseidon2,
        sumcheck::{
            config::SCStarkGenericConfig,
            prover::{SCMachineProver, *},
        },
        Chip, SCStarkMachine,
    };
    use p3_baby_bear::BabyBear;
    use p3_field::extension::BinomialExtensionField;

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    #[test]
    fn generate_compressed_trace() {
        let chip = SimpleAddChip {
            index: 0,
            log_height: Some(3),
            interaction_mode: InteractionMode::None,
            use_nonzero_padding: false,
        };
        let record = DummyRecord;
        let compressed: CompressedMatrix<F> = chip.generate_trace(&record, &mut DummyRecord);
        let height = 1 << 3;
        let main_height = height / 2 + 1;
        assert_eq!(compressed.height(), height);
        assert_eq!(compressed.stored_height(), main_height);
        println!(
            "compressed trace: height={}, stored={}, width={}",
            compressed.height(),
            compressed.stored_height(),
            compressed.main.width()
        );

        let full = compressed.decompress();
        println!("decompressed matrix: {} x {}", full.height(), full.width());
        for i in 0..full.height() {
            let row_vec: Vec<F> = full.row(i).collect();
            let tag = if i < main_height { "real" } else { "pad" };
            println!("  row {i} ({tag}): {row_vec:?}");
        }
    }

    #[test]
    fn simple_prove_and_verify() {
        let config = SCBabyBearPoseidon2::new();
        let mut challenger_prover = config.mlchallenger();
        let mut challenger_verifier = challenger_prover.clone();

        let log_heights = [9, 6];
        let (chips, chips_ext): (Vec<_>, Vec<_>) = log_heights
            .iter()
            .enumerate()
            .map(|(index, log_height)| {
                let chip = Chip::<F, SimpleAddChip>::new(SimpleAddChip {
                    index,
                    log_height: Some(*log_height),
                    ..Default::default()
                });
                let chip_ext = Chip::<EF, SimpleAddChip>::new(SimpleAddChip {
                    index,
                    log_height: Some(*log_height),
                    ..Default::default()
                });
                (chip, chip_ext)
            })
            .collect::<Vec<_>>()
            .into_iter()
            .unzip();

        let machine = SCStarkMachine::new(config, chips, chips_ext, 0, true);

        let prover = SumcheckProver { machine };
        let program = <SimpleAddChip as MachineAir<F>>::Program::default();
        let (pk, vk) = prover.setup(&program);

        let num_skip_rounds = 3;
        let chip_log_height_threshold = 6;

        let prove_result = prover.prove(
            &pk,
            vec![DummyRecord; 1],
            &mut challenger_prover,
            DTCoreOpts::default(),
            num_skip_rounds,
            chip_log_height_threshold,
        );
        if let Err(e) = prove_result {
            panic!("prove failed: {e}");
        }
        let proof = prove_result.unwrap();

        let verify_result = prover.machine().verify(
            &vk,
            &proof,
            &mut challenger_verifier,
            num_skip_rounds,
            chip_log_height_threshold,
        );
        if let Err(e) = verify_result {
            panic!("verify failed: {e}");
        }
    }

    #[test]
    fn prove_and_verify_with_varied_heights() {
        let config = SCBabyBearPoseidon2::new();
        let mut challenger_prover = config.mlchallenger();
        let mut challenger_verifier = challenger_prover.clone();

        let log_heights = [10, 8, 7, 6];
        let (chips, chips_ext): (Vec<_>, Vec<_>) = log_heights
            .iter()
            .enumerate()
            .map(|(index, log_height)| {
                let chip = Chip::<F, SimpleAddChip>::new(SimpleAddChip {
                    index,
                    log_height: Some(*log_height),
                    ..Default::default()
                });
                let chip_ext = Chip::<EF, SimpleAddChip>::new(SimpleAddChip {
                    index,
                    log_height: Some(*log_height),
                    ..Default::default()
                });
                (chip, chip_ext)
            })
            .collect::<Vec<_>>()
            .into_iter()
            .unzip();

        let machine = SCStarkMachine::new(config, chips, chips_ext, 0, true);

        let prover = SumcheckProver { machine };

        let (pk, vk) = prover.setup(&<SimpleAddChip as MachineAir<F>>::Program::default());

        let num_skip_rounds = 2;
        let chip_log_height_threshold = 6;

        let prove_result = prover.prove(
            &pk,
            vec![DummyRecord; 1],
            &mut challenger_prover,
            DTCoreOpts::default(),
            num_skip_rounds,
            chip_log_height_threshold,
        );
        if let Err(e) = prove_result {
            panic!("prove failed: {e}");
        }
        let proof = prove_result.unwrap();

        let verify_result = prover.machine().verify(
            &vk,
            &proof,
            &mut challenger_verifier,
            num_skip_rounds,
            chip_log_height_threshold,
        );
        if let Err(e) = verify_result {
            panic!("verify failed: {e}");
        }
    }

    /// Prove & verify with non-zero padding rows.
    ///
    /// This test exercises the bug fix where padding rows in the algebraic
    /// decomposition path must be multiplied by their corresponding eq polynomial
    /// coefficients, not by a flat constant.
    #[test]
    fn prove_and_verify_with_nonzero_padding() {
        // Print the decompressed trace matrices for each chip.
        for (index, &log_height) in [6usize, 4].iter().enumerate() {
            let chip = SimpleAddChip {
                index,
                log_height: Some(log_height),
                interaction_mode: InteractionMode::None,
                use_nonzero_padding: true,
            };
            let compressed: CompressedMatrix<F> =
                chip.generate_trace(&DummyRecord, &mut DummyRecord);
            let stored = compressed.stored_height();
            let total = compressed.height();
            let full = compressed.decompress();
            println!(
                "\n=== Chip {index} (log_height={log_height}) stored={stored} total={total} ==="
            );
            println!("  columns: a, b, c, is_add, d, is_real");
            for row in 0..full.height() {
                let vals: Vec<F> = full.row(row).collect();
                let tag = if row < stored { "real" } else { "pad " };
                println!("  row {row:3} [{tag}]: {vals:?}");
            }
        }

        let config = SCBabyBearPoseidon2::new();
        let mut challenger_prover = config.mlchallenger();
        let mut challenger_verifier = challenger_prover.clone();

        let log_heights = [6, 4];
        let (chips, chips_ext): (Vec<_>, Vec<_>) = log_heights
            .iter()
            .enumerate()
            .map(|(index, log_height)| {
                let chip = Chip::<F, SimpleAddChip>::new(SimpleAddChip {
                    index,
                    log_height: Some(*log_height),
                    interaction_mode: InteractionMode::None,
                    use_nonzero_padding: true,
                });
                let chip_ext = Chip::<EF, SimpleAddChip>::new(SimpleAddChip {
                    index,
                    log_height: Some(*log_height),
                    interaction_mode: InteractionMode::None,
                    use_nonzero_padding: true,
                });
                (chip, chip_ext)
            })
            .collect::<Vec<_>>()
            .into_iter()
            .unzip();

        let machine = SCStarkMachine::new(config, chips, chips_ext, 0, true);

        let prover = SumcheckProver { machine };
        let program = <SimpleAddChip as MachineAir<F>>::Program::default();
        let (pk, vk) = prover.setup(&program);

        let num_skip_rounds = 2;
        let chip_log_height_threshold = 4;

        let prove_result = prover.prove(
            &pk,
            vec![DummyRecord; 1],
            &mut challenger_prover,
            DTCoreOpts::default(),
            num_skip_rounds,
            chip_log_height_threshold,
        );
        if let Err(e) = prove_result {
            panic!("prove failed: {e}");
        }
        let proof = prove_result.unwrap();

        let verify_result = prover.machine().verify(
            &vk,
            &proof,
            &mut challenger_verifier,
            num_skip_rounds,
            chip_log_height_threshold,
        );
        if let Err(e) = verify_result {
            panic!("verify failed: {e}");
        }
    }
}

#[cfg(test)]
mod permutation_tests {
    #![allow(clippy::print_stdout)]

    use crate::{
        baby_bear_poseidon2::SCBabyBearPoseidon2,
        sumcheck::{
            config::SCStarkGenericConfig,
            prover::{SCMachineProver, *},
        },
        Chip, SCStarkMachine,
    };
    use p3_baby_bear::BabyBear;
    use p3_field::extension::BinomialExtensionField;

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    /// Multi-chip test with permutation traces: sender + receiver + plain chip.
    #[test]
    #[allow(clippy::too_many_lines)]
    fn prove_and_verify_with_permutation() {
        // Print main and permutation traces for each chip.
        {
            use p3_field::AbstractField;
            let random_elements: [EF; 2] = [EF::from_canonical_u32(7), EF::from_canonical_u32(13)];
            let chip_configs: Vec<(usize, usize, InteractionMode)> = vec![
                (0, 8, InteractionMode::Send),
                (1, 8, InteractionMode::Receive),
                (2, 6, InteractionMode::Send),
                (3, 6, InteractionMode::Receive),
            ];
            for (index, log_height, mode) in &chip_configs {
                let chip = Chip::<F, SimpleAddChip>::new(SimpleAddChip {
                    index: *index,
                    log_height: Some(*log_height),
                    interaction_mode: *mode,
                    use_nonzero_padding: false,
                });
                let main_compressed = chip.generate_trace(&DummyRecord, &mut DummyRecord);
                let prep_compressed =
                    chip.generate_preprocessed_trace(&DummyProgram::<F>::default());
                let (perm_compressed, local_cum_sum) = chip.generate_compressed_permutation_trace(
                    prep_compressed.as_ref(),
                    &main_compressed,
                    &random_elements,
                );
                let perm_full = perm_compressed.decompress();
                let main_full = main_compressed.decompress();
                let stored = main_compressed.stored_height();
                let mode_str = match mode {
                    InteractionMode::Send => "Send",
                    InteractionMode::Receive => "Receive",
                    InteractionMode::None => "None",
                };
                println!(
                    "\n=== Chip {} (log_height={}, mode={}) stored={} total={} local_cum_sum={:?} ===",
                    index, log_height, mode_str, stored, main_compressed.height(), local_cum_sum
                );
                println!("  Main trace ({} x {}):", main_full.height(), main_full.width());
                for row in 0..std::cmp::min(main_full.height(), 5) {
                    let vals: Vec<F> = main_full.row(row).collect();
                    println!("    row {row:3}: {vals:?}");
                }
                if main_full.height() > 5 {
                    println!("    ... ({} more rows)", main_full.height() - 5);
                }
                println!("  Permutation trace ({} x {}):", perm_full.height(), perm_full.width());
                for row in 0..std::cmp::min(perm_full.height(), 5) {
                    let vals: Vec<EF> = perm_full.row(row).collect();
                    println!("    row {row:3}: {vals:?}");
                }
                if perm_full.height() > 5 {
                    println!("    ... ({} more rows)", perm_full.height() - 5);
                }
                println!(
                    "  Perm padding stored_height={} total_height={}",
                    perm_compressed.stored_height(),
                    perm_compressed.height()
                );
            }
        }

        let config = SCBabyBearPoseidon2::new();
        let mut challenger_prover = config.mlchallenger();
        let mut challenger_verifier = challenger_prover.clone();

        // All chips must have interactions so that every permutation trace has non-zero width.
        // A Send chip and a Receive chip with the same height form a matching pair.
        let chip_configs: Vec<(usize, usize, InteractionMode)> = vec![
            (0, 8, InteractionMode::Send),
            (1, 8, InteractionMode::Receive),
            (2, 6, InteractionMode::Send),
            (3, 6, InteractionMode::Receive),
        ];

        let (chips, chips_ext): (Vec<_>, Vec<_>) = chip_configs
            .iter()
            .map(|(index, log_height, mode)| {
                let air = SimpleAddChip {
                    index: *index,
                    log_height: Some(*log_height),
                    interaction_mode: *mode,
                    use_nonzero_padding: false,
                };
                (Chip::<F, _>::new(air), Chip::<EF, _>::new(air))
            })
            .unzip();

        // Verify that all chips have non-empty interactions.
        for (i, chip) in chips.iter().enumerate() {
            assert!(
                !chip.sends().is_empty() || !chip.receives().is_empty(),
                "Chip {i} should have interactions"
            );
        }
        println!(
            "Chip 0 sends={}, Chip 1 receives={}, Chip 2 sends={}, Chip 3 receives={}",
            chips[0].sends().len(),
            chips[1].receives().len(),
            chips[2].sends().len(),
            chips[3].receives().len(),
        );

        let machine = SCStarkMachine::new(config, chips, chips_ext, 0, true);
        let prover = SumcheckProver { machine };
        let (pk, vk) = prover.setup(&DummyProgram::<F>::default());

        let num_skip_rounds = 2;
        let chip_log_height_threshold = 4;

        let prove_result = prover.prove(
            &pk,
            vec![DummyRecord; 1],
            &mut challenger_prover,
            DTCoreOpts::default(),
            num_skip_rounds,
            chip_log_height_threshold,
        );
        if let Err(e) = prove_result {
            panic!("prove failed: {e}");
        }
        let proof = prove_result.unwrap();

        let verify_result = prover.machine().verify(
            &vk,
            &proof,
            &mut challenger_verifier,
            num_skip_rounds,
            chip_log_height_threshold,
        );
        if let Err(e) = verify_result {
            panic!("verify failed: {e}");
        }
    }

    /// Multi-chip test with varied heights and permutation traces.
    #[test]
    fn prove_and_verify_with_permutation_varied_heights() {
        let config = SCBabyBearPoseidon2::new();
        let mut challenger_prover = config.mlchallenger();
        let mut challenger_verifier = challenger_prover.clone();

        // All chips have interactions; varied heights test the sumcheck with different-sized
        // permutation traces. Each Send/Receive pair shares the same height.
        let chip_configs: Vec<(usize, usize, InteractionMode)> = vec![
            (0, 10, InteractionMode::Send),
            (1, 10, InteractionMode::Receive),
            (2, 7, InteractionMode::Send),
            (3, 7, InteractionMode::Receive),
            (4, 8, InteractionMode::Send),
            (5, 8, InteractionMode::Receive),
        ];

        let (chips, chips_ext): (Vec<_>, Vec<_>) = chip_configs
            .iter()
            .map(|(index, log_height, mode)| {
                let air = SimpleAddChip {
                    index: *index,
                    log_height: Some(*log_height),
                    interaction_mode: *mode,
                    use_nonzero_padding: false,
                };
                (Chip::<F, _>::new(air), Chip::<EF, _>::new(air))
            })
            .unzip();

        let machine = SCStarkMachine::new(config, chips, chips_ext, 0, true);
        let prover = SumcheckProver { machine };
        let (pk, vk) = prover.setup(&DummyProgram::<F>::default());

        let num_skip_rounds = 2;
        let chip_log_height_threshold = 6;

        let prove_result = prover.prove(
            &pk,
            vec![DummyRecord; 1],
            &mut challenger_prover,
            DTCoreOpts::default(),
            num_skip_rounds,
            chip_log_height_threshold,
        );
        if let Err(e) = prove_result {
            panic!("prove failed: {e}");
        }
        let proof = prove_result.unwrap();

        let verify_result = prover.machine().verify(
            &vk,
            &proof,
            &mut challenger_verifier,
            num_skip_rounds,
            chip_log_height_threshold,
        );
        if let Err(e) = verify_result {
            panic!("verify failed: {e}");
        }
    }
}
