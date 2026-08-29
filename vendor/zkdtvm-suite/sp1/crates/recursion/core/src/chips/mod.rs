pub mod alu_base;
pub mod alu_ext;
pub mod ext_exp_reverse_bits;
pub mod mem;
pub mod poly;
pub mod poseidon2_skinny;
#[cfg(feature = "koalabear")]
pub mod poseidon2_skinny_kb;
pub mod poseidon2_wide;
#[cfg(feature = "koalabear")]
pub mod poseidon2_wide_kb;
pub mod prefix_sum_checks;
pub mod public_values;
pub mod select;
pub mod sumcheck_round;

#[cfg(test)]
pub mod test_fixtures {
    use crate::*;
    use dt_stark::inner_perm;
    use p3_baby_bear::BabyBear;
    use p3_field::{AbstractField, Field, PrimeField32};
    use p3_symmetric::Permutation;
    use rand::{prelude::SliceRandom, rngs::StdRng, Rng, SeedableRng};
    use std::{array, borrow::Borrow};

    const SEED: u64 = 12345;
    pub const MIN_TEST_CASES: usize = 1000;
    const MAX_TEST_CASES: usize = 10000;

    pub fn shard() -> ExecutionRecord<BabyBear> {
        ExecutionRecord {
            base_alu_events: base_alu_events(),
            ext_alu_events: ext_alu_events(),
            commit_pv_hash_events: public_values_events(),
            select_events: select_events(),
            poseidon2_events: poseidon2_events(),
            poly_eval_events: poly_eval_events(),
            ..Default::default()
        }
    }

    pub fn program() -> RecursionProgram<BabyBear> {
        let mut instructions = [
            base_alu_instructions(),
            ext_alu_instructions(),
            public_values_instructions(),
            select_instructions(),
            poseidon2_instructions(),
            poly_eval_instructions(),
        ]
        .concat();

        let mut rng = StdRng::seed_from_u64(SEED);
        instructions.shuffle(&mut rng);

        linear_program(instructions).unwrap()
    }

    pub fn default_execution_record() -> ExecutionRecord<BabyBear> {
        ExecutionRecord::<BabyBear>::default()
    }

    fn initialize() -> (StdRng, usize) {
        let mut rng = StdRng::seed_from_u64(SEED);
        let num_test_cases = rng.gen_range(MIN_TEST_CASES..=MAX_TEST_CASES);
        (rng, num_test_cases)
    }

    fn base_alu_events() -> Vec<BaseAluIo<BabyBear>> {
        let (mut rng, num_test_cases) = initialize();
        let mut events = Vec::with_capacity(num_test_cases);
        for _ in 0..num_test_cases {
            let in1 = BabyBear::from_wrapped_u32(rng.gen());
            let in2 = BabyBear::from_wrapped_u32(rng.gen());
            let out = match rng.gen_range(0..4) {
                0 => in1 + in2, // Add
                1 => in1 - in2, // Sub
                2 => in1 * in2, // Mul
                _ => {
                    let in2 = if in2.is_zero() { BabyBear::one() } else { in2 };
                    in1 / in2
                }
            };
            events.push(BaseAluIo { out, in1, in2 });
        }
        events
    }

    fn ext_alu_events() -> Vec<ExtAluIo<Block<BabyBear>>> {
        let (_, num_test_cases) = initialize();
        let mut events = Vec::with_capacity(num_test_cases);
        for _ in 0..num_test_cases {
            events.push(ExtAluIo {
                out: BabyBear::one().into(),
                in1: BabyBear::one().into(),
                in2: BabyBear::one().into(),
            });
        }
        events
    }

    fn public_values_events() -> Vec<CommitPublicValuesEvent<BabyBear>> {
        let (mut rng, num_test_cases) = initialize();
        let mut events = Vec::with_capacity(num_test_cases);
        for _ in 0..num_test_cases {
            let random_felts: [BabyBear; air::RECURSIVE_PROOF_NUM_PV_ELTS] =
                array::from_fn(|_| BabyBear::from_wrapped_u32(rng.gen()));
            events
                .push(CommitPublicValuesEvent { public_values: *random_felts.as_slice().borrow() });
        }
        events
    }

    fn select_events() -> Vec<SelectIo<BabyBear>> {
        let (mut rng, num_test_cases) = initialize();
        let mut events = Vec::with_capacity(num_test_cases);
        for _ in 0..num_test_cases {
            let bit = if rng.gen_bool(0.5) { BabyBear::one() } else { BabyBear::zero() };
            let in1 = BabyBear::from_wrapped_u32(rng.gen());
            let in2 = BabyBear::from_wrapped_u32(rng.gen());
            let (out1, out2) = if bit == BabyBear::one() { (in1, in2) } else { (in2, in1) };
            events.push(SelectIo { bit, out1, out2, in1, in2 });
        }
        events
    }

    fn poseidon2_events() -> Vec<Poseidon2Event<BabyBear>> {
        let (mut rng, num_test_cases) = initialize();
        let mut events = Vec::with_capacity(num_test_cases);
        for _ in 0..num_test_cases {
            let input = array::from_fn(|_| BabyBear::from_wrapped_u32(rng.gen()));
            let permuter = inner_perm();
            let output = permuter.permute(input);

            events.push(Poseidon2Event { input, output });
        }
        events
    }

    fn poly_eval_events() -> Vec<PolyEvalEvent<BabyBear>> {
        let (mut rng, num_test_cases) = initialize();
        let mut events = Vec::with_capacity(num_test_cases);
        for _ in 0..num_test_cases {
            let point = BabyBear::from_wrapped_u32(rng.gen());
            let len = rng.gen_range(1..4); // Random length between 1 and 4 bits
            let coeff: Vec<BabyBear> =
                (0..len).map(|_| BabyBear::from_canonical_u32(rng.gen_range(0..3))).collect();
            let out = coeff[1..].iter().fold(coeff[0], |acc, &x| acc * point + x);

            events.push(PolyEvalEvent { point, coeff, out });
        }
        events
    }

    fn base_alu_instructions() -> Vec<Instruction<BabyBear>> {
        let (mut rng, num_test_cases) = initialize();
        let mut instructions = Vec::with_capacity(num_test_cases);
        for _ in 0..num_test_cases {
            let opcode = match rng.gen_range(0..4) {
                0 => BaseAluOpcode::AddF,
                1 => BaseAluOpcode::SubF,
                2 => BaseAluOpcode::MulF,
                _ => BaseAluOpcode::DivF,
            };
            instructions.push(Instruction::BaseAlu(BaseAluInstr {
                opcode,
                mult: BabyBear::from_wrapped_u32(rng.gen()),
                addrs: BaseAluIo {
                    out: Address(BabyBear::from_wrapped_u32(rng.gen())),
                    in1: Address(BabyBear::from_wrapped_u32(rng.gen())),
                    in2: Address(BabyBear::from_wrapped_u32(rng.gen())),
                },
            }));
        }
        instructions
    }

    fn ext_alu_instructions() -> Vec<Instruction<BabyBear>> {
        let (mut rng, num_test_cases) = initialize();
        let mut instructions = Vec::with_capacity(num_test_cases);
        for _ in 0..num_test_cases {
            let opcode = match rng.gen_range(0..4) {
                0 => ExtAluOpcode::AddE,
                1 => ExtAluOpcode::SubE,
                2 => ExtAluOpcode::MulE,
                _ => ExtAluOpcode::DivE,
            };
            instructions.push(Instruction::ExtAlu(ExtAluInstr {
                opcode,
                mult: BabyBear::from_wrapped_u32(rng.gen()),
                addrs: ExtAluIo {
                    out: Address(BabyBear::from_wrapped_u32(rng.gen())),
                    in1: Address(BabyBear::from_wrapped_u32(rng.gen())),
                    in2: Address(BabyBear::from_wrapped_u32(rng.gen())),
                },
            }));
        }
        instructions
    }

    fn public_values_instructions() -> Vec<Instruction<BabyBear>> {
        let (mut rng, num_test_cases) = initialize();
        let mut instructions = Vec::with_capacity(num_test_cases);
        for _ in 0..num_test_cases {
            let public_values_a: [u32; air::RECURSIVE_PROOF_NUM_PV_ELTS] =
                array::from_fn(|_| BabyBear::from_wrapped_u32(rng.gen()).as_canonical_u32());
            let public_values: &RecursionPublicValues<u32> = public_values_a.as_slice().borrow();
            instructions.push(runtime::instruction::commit_public_values(public_values));
        }
        instructions
    }

    fn select_instructions() -> Vec<Instruction<BabyBear>> {
        let (mut rng, num_test_cases) = initialize();
        let mut instructions = Vec::with_capacity(num_test_cases);
        for _ in 0..num_test_cases {
            instructions.push(Instruction::Select(SelectInstr {
                addrs: SelectIo {
                    bit: Address(BabyBear::from_wrapped_u32(rng.gen())),
                    out1: Address(BabyBear::from_wrapped_u32(rng.gen())),
                    out2: Address(BabyBear::from_wrapped_u32(rng.gen())),
                    in1: Address(BabyBear::from_wrapped_u32(rng.gen())),
                    in2: Address(BabyBear::from_wrapped_u32(rng.gen())),
                },
                mult1: BabyBear::from_wrapped_u32(rng.gen()),
                mult2: BabyBear::from_wrapped_u32(rng.gen()),
            }));
        }
        instructions
    }

    fn poseidon2_instructions() -> Vec<Instruction<BabyBear>> {
        let (mut rng, num_test_cases) = initialize();
        let mut instructions = Vec::with_capacity(num_test_cases);

        for _ in 0..num_test_cases {
            let input = array::from_fn(|_| Address(BabyBear::from_wrapped_u32(rng.gen())));
            let output = array::from_fn(|_| Address(BabyBear::from_wrapped_u32(rng.gen())));
            let mults = array::from_fn(|_| BabyBear::from_wrapped_u32(rng.gen()));

            instructions.push(Instruction::Poseidon2(Box::new(Poseidon2Instr {
                addrs: Poseidon2Io { input, output },
                mults,
            })));
        }
        instructions
    }

    fn poly_eval_instructions() -> Vec<Instruction<BabyBear>> {
        let (mut rng, num_test_cases) = initialize();
        let mut instructions = Vec::with_capacity(num_test_cases);
        for _ in 0..num_test_cases {
            let len = rng.gen_range(1..4); // Random length between 1 and 4 bits
            let coeff: Vec<Address<BabyBear>> =
                (0..len).map(|_| Address(BabyBear::from_wrapped_u32(rng.gen()))).collect();
            let point = Address(BabyBear::from_wrapped_u32(rng.gen()));
            let out = Address(BabyBear::from_wrapped_u32(rng.gen()));
            let mult = BabyBear::from_wrapped_u32(rng.gen());
            let num_chains = if len > 1 { len - 1 } else { 0 };
            instructions.push(Instruction::PolyEval(PolyEvalInstr {
                addrs: PolyEvalIo { point, coeff, out },
                mult,
                chain_accum_addrs: (0..num_chains)
                    .map(|i| Address(BabyBear::from_canonical_usize(900000 + i)))
                    .collect(),
            }));
        }
        instructions
    }
}
