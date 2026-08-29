use core::{array, borrow::Borrow, ops::Deref};

use dt_stark::{
    air::{FullAir, FullAirBuilder, MachineAir, PairCol},
    sumcheck::trace::CompressedMatrix,
};
use p3_air::BaseAir;
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use crate::{
    config::{DIGEST_SIZE, F, POSEIDON2_WIDTH},
    system_dt::{RecursionNativeProgram, RecursionRecord},
    transcript_dt::{
        bus::Poseidon2PermuteBus,
        merkle_path::{
            bus::{
                MerkleCommitmentRootBus, MerkleDigestChainBus, MerkleLeafBlockBus,
                MerkleSpongeStateChainBus,
            },
            columns::{MerklePathCols, NUM_MERKLE_PATH_COLS},
            trace::{merkle_row_iter, MerklePathTraceGenerator},
        },
    },
};

#[derive(Debug, Clone, Copy)]
pub struct MerklePathAir {
    pub poseidon2_bus: Poseidon2PermuteBus,
    pub digest_chain_bus: MerkleDigestChainBus,
    pub sponge_state_chain_bus: MerkleSpongeStateChainBus,
    pub commitment_root_bus: MerkleCommitmentRootBus,
    pub leaf_block_bus: MerkleLeafBlockBus,
}

impl MerklePathAir {
    pub const fn new(
        poseidon2_bus: Poseidon2PermuteBus,
        digest_chain_bus: MerkleDigestChainBus,
        sponge_state_chain_bus: MerkleSpongeStateChainBus,
        commitment_root_bus: MerkleCommitmentRootBus,
        leaf_block_bus: MerkleLeafBlockBus,
    ) -> Self {
        Self {
            poseidon2_bus,
            digest_chain_bus,
            sponge_state_chain_bus,
            commitment_root_bus,
            leaf_block_bus,
        }
    }
}

impl Default for MerklePathAir {
    fn default() -> Self {
        Self::new(
            Poseidon2PermuteBus::new(),
            MerkleDigestChainBus::new(),
            MerkleSpongeStateChainBus::new(),
            MerkleCommitmentRootBus::new(),
            MerkleLeafBlockBus::new(),
        )
    }
}

impl<Fld: Field> BaseAir<Fld> for MerklePathAir {
    fn width(&self) -> usize {
        NUM_MERKLE_PATH_COLS
    }
}

impl<AB: FullAirBuilder> FullAir<AB> for MerklePathAir {
    fn width(&self) -> usize {
        NUM_MERKLE_PATH_COLS
    }

    fn required_max_beta_power(&self) -> usize {
        [
            self.poseidon2_bus.required_max_beta_power_floor(),
            self.digest_chain_bus.required_max_beta_power_floor(),
            self.sponge_state_chain_bus.required_max_beta_power_floor(),
            self.commitment_root_bus.required_max_beta_power_floor(),
            self.leaf_block_bus.required_max_beta_power_floor(),
        ]
        .into_iter()
        .max()
        .expect("non-empty bus list")
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        (0..NUM_MERKLE_PATH_COLS).map(PairCol::Main).collect()
    }

    fn precompute_lc(&self, builder: &mut AB) {
        let denominators = {
            let main = builder.main();
            let local: &MerklePathCols<AB::VarMaybeExt> = main.borrow();
            let proof_idx = local.proof_idx.clone();
            let output_digest = output_digest(local);
            let left_digest = input_digest(local, 0);
            let right_digest = input_digest(local, DIGEST_SIZE);
            let next_level =
                local.level.clone() + local.is_valid.clone() - local.is_leaf_absorb.clone();
            let right_idx = local.left_idx.clone() + AB::one_maybe() - local.is_inject.clone();

            vec![
                self.poseidon2_bus.denominator(builder, local.input.clone(), local.output.clone()),
                self.digest_chain_bus.denominator(
                    builder,
                    proof_idx.clone(),
                    local.commit_id.clone(),
                    local.level.clone(),
                    left_digest,
                    local.left_idx.clone(),
                ),
                self.digest_chain_bus.denominator(
                    builder,
                    proof_idx.clone(),
                    local.commit_id.clone(),
                    local.level.clone(),
                    right_digest,
                    right_idx,
                ),
                self.digest_chain_bus.denominator(
                    builder,
                    proof_idx.clone(),
                    local.commit_id.clone(),
                    next_level,
                    output_digest.clone(),
                    local.idx.clone(),
                ),
                self.sponge_state_chain_bus.denominator(
                    builder,
                    proof_idx.clone(),
                    local.unit_key.clone(),
                    local.idx.clone(),
                    local.block_idx.clone(),
                    local.prev_state.clone(),
                ),
                self.sponge_state_chain_bus.denominator(
                    builder,
                    proof_idx.clone(),
                    local.unit_key.clone(),
                    local.idx.clone(),
                    local.block_idx.clone() + AB::one_maybe(),
                    local.output.clone(),
                ),
                self.commitment_root_bus.denominator(
                    builder,
                    proof_idx.clone(),
                    local.commit_id.clone(),
                    output_digest,
                ),
                self.leaf_block_bus.denominator(
                    builder,
                    proof_idx.clone(),
                    local.commit_id.clone(),
                    local.unit_key.clone(),
                    local.idx.clone(),
                    local.block_idx.clone(),
                    local.chunk_mask.clone(),
                    local.chunk.clone(),
                ),
            ]
        };
        for denominator in denominators {
            builder.retain_precomputed(denominator);
        }
    }

    fn eval(&self, builder: &mut AB) {
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        let local: &MerklePathCols<AB::VarMaybeExt> = local_binding.deref().borrow();

        assert_bool(builder, local.is_valid.clone());
        assert_bool(builder, local.is_leaf_absorb.clone());
        assert_bool(builder, local.is_inject.clone());
        assert_bool(builder, local.is_last.clone());
        assert_bool(builder, local.is_leaf_first.clone());
        assert_bool(builder, local.is_leaf_last.clone());
        for bit in local.chunk_mask.iter() {
            assert_bool(builder, bit.clone());
        }

        let one = AB::one_maybe();
        let is_node = local.is_valid.clone() - local.is_leaf_absorb.clone();
        assert_bool(builder, is_node.clone());
        assert_flag_implies(builder, local.is_inject.clone(), is_node.clone());
        assert_flag_implies(builder, local.is_last.clone(), is_node.clone());
        assert_flag_implies(builder, local.is_leaf_first.clone(), local.is_leaf_absorb.clone());
        assert_flag_implies(builder, local.is_leaf_last.clone(), local.is_leaf_absorb.clone());
        // Blocks are absorbed only on leaf rows; the count itself is
        // balance-forced against the producer send multiset.
        builder
            .assert_zero(local.absorb_cnt.clone() * (one.clone() - local.is_leaf_absorb.clone()));

        let two = AB::VarMaybeExt::from(AB::F::from_canonical_usize(2));
        let plain_node = is_node.clone() - local.is_inject.clone();
        builder
            .assert_zero(plain_node * (local.left_idx.clone() - local.idx.clone() * two.clone()));
        builder.assert_zero(local.is_inject.clone() * (local.left_idx.clone() - local.idx.clone()));
        builder.assert_zero(local.left_cnt.clone() * (one.clone() - is_node.clone()));
        builder.assert_zero(local.right_cnt.clone() * (one.clone() - is_node.clone()));
        builder.assert_zero(local.root_cnt.clone() * (one.clone() - local.is_last.clone()));
        builder.assert_eq(
            local.absorb_cnt.clone() * local.left_idx.clone() +
                local.root_cnt.clone() * local.block_idx.clone(),
            local.is_leaf_absorb.clone() + local.is_last.clone(),
        );

        for value in local.prev_state.iter() {
            builder.assert_zero(local.is_leaf_first.clone() * value.clone());
        }

        for i in 0..POSEIDON2_WIDTH {
            let expected = if i < DIGEST_SIZE {
                select_maybe::<AB>(
                    local.chunk_mask[i].clone(),
                    local.chunk[i].clone(),
                    local.prev_state[i].clone(),
                )
            } else {
                local.prev_state[i].clone()
            };
            builder.assert_zero(local.is_leaf_absorb.clone() * (local.input[i].clone() - expected));
        }
    }

    fn lookup(&self, builder: &mut AB) {
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        let local: &MerklePathCols<AB::VarMaybeExt> = local_binding.deref().borrow();

        // Order matches precompute_lc: poseidon recv, left edge recv, right edge recv,
        // edge send, sponge recv, sponge send, commitment-root recv, leaf block recv.
        builder.recv(local.is_valid.clone());
        builder.recv(local.left_cnt.clone());
        builder.recv(local.right_cnt.clone());
        builder.send(
            local.is_leaf_last.clone() + local.is_valid.clone() -
                local.is_leaf_absorb.clone() -
                local.is_last.clone(),
        );
        builder.recv(local.is_leaf_absorb.clone() - local.is_leaf_first.clone());
        builder.send(local.is_leaf_absorb.clone() - local.is_leaf_last.clone());
        builder.recv(local.root_cnt.clone());
        builder.recv(local.absorb_cnt.clone());
    }
}

impl MachineAir<F> for MerklePathAir {
    type Record = RecursionRecord;
    type Program = RecursionNativeProgram<F>;

    fn name(&self) -> String {
        "NativeMerklePath".to_string()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        Some(MerklePathTraceGenerator::trace_height(input))
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        _output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        MerklePathTraceGenerator::generate_trace_compressed(input)
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        for row in merkle_row_iter(input) {
            output.poseidon2.record_poseidon2(row.input);
        }
    }

    fn included(&self, _record: &Self::Record) -> bool {
        true
    }

    fn local_only(&self) -> bool {
        true
    }
}

fn output_digest<T: Clone>(local: &MerklePathCols<T>) -> [T; DIGEST_SIZE] {
    array::from_fn(|i| local.output[i].clone())
}

fn input_digest<T: Clone>(local: &MerklePathCols<T>, offset: usize) -> [T; DIGEST_SIZE] {
    array::from_fn(|i| local.input[offset + i].clone())
}

fn assert_bool<AB: FullAirBuilder>(builder: &mut AB, value: AB::VarMaybeExt) {
    builder.assert_zero(value.clone() * (value - AB::one_maybe()));
}

fn assert_flag_implies<AB: FullAirBuilder>(
    builder: &mut AB,
    flag: AB::VarMaybeExt,
    condition: AB::VarMaybeExt,
) {
    builder.assert_zero(flag * (AB::one_maybe() - condition));
}

fn select_maybe<AB: FullAirBuilder>(
    bit: AB::VarMaybeExt,
    when_one: AB::VarMaybeExt,
    when_zero: AB::VarMaybeExt,
) -> AB::VarMaybeExt {
    when_zero.clone() + bit * (when_one - when_zero)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::{D_EF, F},
        system_dt::{RecursionMerklePathRow, RecursionProofRecord},
        transcript_dt::{merkle_path::trace::trace_row, poseidon2::RecursionPoseidon2Memo},
    };
    use p3_matrix::Matrix;

    #[test]
    fn symbolic_analysis() {
        let air = MerklePathAir::default();
        let chip = polyair::Chip::<MerklePathAir, F, D_EF>::new(air);
        assert_eq!(chip.width(), 81);
        assert_eq!(chip.reserved_poly().len(), 81);
        assert_eq!(chip.num_precompute(), 8);
        assert_eq!(chip.perm_width(), 4);
        assert_eq!(chip.symbolic_builder.gate.len(), 58);
        assert_eq!(chip.num_alpha, 63);
        assert_eq!(chip.num_lookup(), 8);
        assert_eq!(chip.required_max_beta_power(), 34);
        assert!(chip.degree <= 3);
    }

    #[test]
    fn trace_and_dependencies_for_short_path() {
        let poseidon2_memo = RecursionPoseidon2Memo::default();
        let chunk = array::from_fn(|i| F::from_canonical_usize(i + 1));
        let mask = [true; DIGEST_SIZE];
        let leaf = RecursionMerklePathRow::leaf_absorb(
            0,
            7,
            17,
            0,
            2,
            1,
            false,
            true,
            true,
            [F::zero(); POSEIDON2_WIDTH],
            chunk,
            mask,
            &poseidon2_memo,
        );
        let sibling0 = array::from_fn(|i| F::from_canonical_usize(100 + i));
        let row0 = RecursionMerklePathRow::path_compress(
            0,
            17,
            0,
            2,
            output_digest_record(&leaf),
            sibling0,
            false,
            &poseidon2_memo,
        );
        let sibling1 = array::from_fn(|i| F::from_canonical_usize(200 + i));
        let row1 = RecursionMerklePathRow::path_compress(
            0,
            17,
            1,
            1,
            output_digest_record(&row0),
            sibling1,
            true,
            &poseidon2_memo,
        );

        let mut record = RecursionRecord::default();
        record.proof_records.push(RecursionProofRecord::default());
        record.proof_records[0].merkle_path.push_row(leaf);
        record.proof_records[0].merkle_path.push_row(row0);
        record.proof_records[0].merkle_path.push_row(row1);

        let trace = MerklePathTraceGenerator::generate_trace_row_major(&record);
        assert_eq!(trace.width(), NUM_MERKLE_PATH_COLS);
        assert_eq!(trace.height(), 4);

        let mut deps = RecursionRecord::default();
        MerklePathAir::default().generate_dependencies(&record, &mut deps);
        assert_eq!(deps.poseidon2.unique_count(), 3);
        assert_eq!(deps.poseidon2.total_count(), 3);
    }

    #[test]
    fn materialized_components_require_leaf_and_root_demand() {
        let poseidon2_memo = RecursionPoseidon2Memo::default();
        let chunk = array::from_fn(|i| F::from_canonical_usize(i + 1));
        let leaf = RecursionMerklePathRow::leaf_absorb(
            0,
            7,
            17,
            0,
            2,
            3,
            false,
            true,
            true,
            [F::zero(); POSEIDON2_WIDTH],
            chunk,
            [true; DIGEST_SIZE],
            &poseidon2_memo,
        );
        let root = RecursionMerklePathRow::path_compress(
            0,
            17,
            0,
            2,
            output_digest_record(&leaf),
            array::from_fn(|i| F::from_canonical_usize(100 + i)),
            true,
            &poseidon2_memo,
        );

        let record_for = |leaf: RecursionMerklePathRow, root: RecursionMerklePathRow| {
            let mut record = RecursionRecord::default();
            record.proof_records.push(RecursionProofRecord::default());
            record.proof_records[0].merkle_path.push_row(leaf);
            record.proof_records[0].merkle_path.push_row(root);
            record
        };

        let honest = MerklePathTraceGenerator::generate_trace_compressed(&record_for(leaf, root));
        let leaf_binding = honest.main.row_slice(0);
        let leaf_cols: &MerklePathCols<F> = leaf_binding.as_ref().borrow();
        assert_eq!(leaf_cols.left_idx, F::from_canonical_usize(3).inverse());
        assert_eq!(leaf_cols.absorb_cnt * leaf_cols.left_idx - leaf_cols.is_leaf_absorb, F::zero());
        let root_binding = honest.main.row_slice(1);
        let root_cols: &MerklePathCols<F> = root_binding.as_ref().borrow();
        assert_eq!(root_cols.block_idx, F::one());
        assert_eq!(root_cols.root_cnt * root_cols.block_idx - root_cols.is_last, F::zero());

        let mut no_leaf_trace = honest.clone();
        let absorb_offset = core::mem::offset_of!(MerklePathCols<u8>, absorb_cnt);
        let left_idx_offset = core::mem::offset_of!(MerklePathCols<u8>, left_idx);
        no_leaf_trace.main.values[absorb_offset] = F::zero();
        no_leaf_trace.main.values[left_idx_offset] = F::zero();
        let no_leaf_binding = no_leaf_trace.main.row_slice(0);
        let no_leaf_cols: &MerklePathCols<F> = no_leaf_binding.as_ref().borrow();
        assert_ne!(
            no_leaf_cols.absorb_cnt * no_leaf_cols.left_idx - no_leaf_cols.is_leaf_absorb,
            F::zero()
        );

        let mut no_root_trace = honest.clone();
        let root_row_start = NUM_MERKLE_PATH_COLS;
        let root_cnt_offset = core::mem::offset_of!(MerklePathCols<u8>, root_cnt);
        let block_idx_offset = core::mem::offset_of!(MerklePathCols<u8>, block_idx);
        no_root_trace.main.values[root_row_start + root_cnt_offset] = F::zero();
        no_root_trace.main.values[root_row_start + block_idx_offset] = F::zero();
        let no_root_binding = no_root_trace.main.row_slice(1);
        let no_root_cols: &MerklePathCols<F> = no_root_binding.as_ref().borrow();
        assert_ne!(
            no_root_cols.root_cnt * no_root_cols.block_idx - no_root_cols.is_last,
            F::zero()
        );
    }

    #[test]
    fn partial_leaf_block_keeps_full_previous_state() {
        let poseidon2_memo = RecursionPoseidon2Memo::default();
        let prev_state = array::from_fn(|i| F::from_canonical_usize(300 + i));
        let chunk = array::from_fn(|i| F::from_canonical_usize(400 + i));
        let mut mask = [false; DIGEST_SIZE];
        mask[0] = true;
        mask[1] = true;
        let row = RecursionMerklePathRow::leaf_absorb(
            0,
            1,
            4,
            1,
            5,
            1,
            false,
            false,
            true,
            prev_state,
            chunk,
            mask,
            &poseidon2_memo,
        );

        let trace = trace_row(&row);
        let cols: &MerklePathCols<F> = trace.as_slice().borrow();
        assert_eq!(cols.input[0], chunk[0]);
        assert_eq!(cols.input[1], chunk[1]);
        for i in 2..POSEIDON2_WIDTH {
            assert_eq!(cols.input[i], prev_state[i]);
        }
    }

    #[test]
    fn inject_compress_uses_leaf_digest_source() {
        let poseidon2_memo = RecursionPoseidon2Memo::default();
        // NATIVE_REC_TODO_DELETE: real-MMCS fixture — land when recording-challenger pipeline
        // testable.
        let mask = [true; DIGEST_SIZE];
        let main_chunk = array::from_fn(|i| F::from_canonical_usize(10 + i));
        let main_leaf = RecursionMerklePathRow::leaf_absorb(
            0,
            21,
            51,
            0,
            4,
            1,
            false,
            true,
            true,
            [F::zero(); POSEIDON2_WIDTH],
            main_chunk,
            mask,
            &poseidon2_memo,
        );
        let sibling0 = array::from_fn(|i| F::from_canonical_usize(100 + i));
        let path0 = RecursionMerklePathRow::path_compress(
            0,
            51,
            0,
            4,
            output_digest_record(&main_leaf),
            sibling0,
            false,
            &poseidon2_memo,
        );

        let injected_chunk = array::from_fn(|i| F::from_canonical_usize(200 + i));
        let injected_leaf = RecursionMerklePathRow::leaf_absorb_at_level(
            0,
            32,
            51,
            1,
            0,
            2,
            1,
            false,
            true,
            true,
            [F::zero(); POSEIDON2_WIDTH],
            injected_chunk,
            mask,
            &poseidon2_memo,
        );
        let inject = RecursionMerklePathRow::inject_compress(
            0,
            51,
            1,
            2,
            output_digest_record(&path0),
            output_digest_record(&injected_leaf),
            false,
            &poseidon2_memo,
        );
        let sibling1 = array::from_fn(|i| F::from_canonical_usize(300 + i));
        let root = RecursionMerklePathRow::path_compress(
            0,
            51,
            2,
            2,
            output_digest_record(&inject),
            sibling1,
            true,
            &poseidon2_memo,
        );

        let injected_trace = trace_row(&injected_leaf);
        let injected_cols: &MerklePathCols<F> = injected_trace.as_slice().borrow();
        assert_eq!(injected_cols.level, F::from_canonical_usize(1));

        let inject_trace = trace_row(&inject);
        let inject_cols: &MerklePathCols<F> = inject_trace.as_slice().borrow();
        assert_eq!(inject_cols.is_inject, F::one());
        assert_eq!(&inject_cols.input[..DIGEST_SIZE], &output_digest_record(&path0));
        assert_eq!(&inject_cols.input[DIGEST_SIZE..], &output_digest_record(&injected_leaf));

        let mut record = RecursionRecord::default();
        record.proof_records.push(RecursionProofRecord::default());
        for row in [main_leaf, path0, injected_leaf, inject, root] {
            record.proof_records[0].merkle_path.push_row(row);
        }

        let trace = MerklePathTraceGenerator::generate_trace_row_major(&record);
        assert_eq!(trace.width(), NUM_MERKLE_PATH_COLS);
        assert_eq!(trace.height(), 8);

        let mut deps = RecursionRecord::default();
        MerklePathAir::default().generate_dependencies(&record, &mut deps);
        assert_eq!(deps.poseidon2.unique_count(), 5);
        assert_eq!(deps.poseidon2.total_count(), 5);
    }

    fn output_digest_record(row: &RecursionMerklePathRow) -> [F; DIGEST_SIZE] {
        array::from_fn(|i| row.output[i])
    }
}
