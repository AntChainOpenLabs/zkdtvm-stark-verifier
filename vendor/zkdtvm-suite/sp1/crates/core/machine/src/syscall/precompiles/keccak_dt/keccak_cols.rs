use std::ops::Index;

use dt_core_executor::events::ByteRecord;
use dt_derive::AlignedBorrow;
use dt_stark::{air::DTAirBuilder, Word};
use p3_air::AirBuilder;
use p3_field::{AbstractField, Field};
use typenum::U2;

use crate::{
    operations_dt::{CompactWord, CompactWordToWordWitness, XorNOperation},
    syscall::precompiles::keccak_dt::STATE_SIZE,
};

pub const NUM_KECCAK_COLS: usize = size_of::<KeccakCols<u8>>();

#[derive(AlignedBorrow, Debug, Clone, Copy)]
#[repr(C)]
pub struct KeccakCols<T> {
    pub step: T,
    pub step_low_one_hot: [T; 4],
    pub step_high_one_hot: [T; 6],

    pub a: [[[CompactWord<T>; 2]; 5]; 5],

    /// c[x] = xor(a[0][x], a[1][x], a[2][x], a[3][x], a[4][x])
    pub c: [[T; 64]; 5],

    /// c'[x] = xor(c[x], c[x - 1], c[x + 1].rotate_left(1))
    pub c_prime: [[T; 64]; 5],

    /// a'[y][x] = xor(a[y][x], c[x], c'[x]) = xor(a[y][x], c[x - 1], c[x + 1].rotate_left(1))
    pub a_prime: [[[T; 64]; 5]; 5],

    /// b[y][x] = perm(a'[y][x])
    /// a''[y][x] = xor(b[y][x], and(not(b[y][x + 1]), b[y][x + 2]))
    pub a_prime_prime: [[[CompactWord<T>; 2]; 5]; 5],

    pub a_prime_prime_0_0_witness: [CompactWordToWordWitness<T>; 2],

    pub rc: [CompactWord<T>; 2],
    pub rc_witness: [CompactWordToWordWitness<T>; 2],

    /// a'''[0][0] = xor(a''[0][0], rc)
    pub a_prime_prime_prime_0_0: [XorNOperation<T, U2>; 2],
}

impl<F: Field> KeccakCols<F> {
    pub fn populate(
        &mut self,
        blu: &mut impl ByteRecord,
        step: usize,
        state: &mut [u64; STATE_SIZE],
    ) {
        self.step = F::from_canonical_u32(step as u32);
        self.step_low_one_hot[step & 0x3] = F::one();
        self.step_high_one_hot[step >> 2] = F::one();

        let mut c = [0u64; 5];
        for i in 0..5 {
            let state = &state[i * 5..];
            for j in 0..5 {
                self.a[i][j][0] = ((state[j] & 0xFFFFFFFFu64) as u32).into();
                self.a[i][j][1] = ((state[j] >> 32) as u32).into();

                c[j] ^= state[j];
            }
        }

        let mut c_prime = [0u64; 5];
        for i in 0..5 {
            for j in 0..64 {
                self.c[i][j] = F::from_bool((c[i] & (1u64 << j)) > 0);
            }

            c_prime[i] = c[(i + 4) % 5] ^ c[(i + 1) % 5].rotate_left(1);
        }

        for i in 0..5 {
            {
                let c_prime = c[i] ^ c_prime[i];
                for j in 0..64 {
                    self.c_prime[i][j] = F::from_bool((c_prime & (1u64 << j)) > 0);
                }
            }
            let state = &mut state[i * 5..];
            for j in 0..5 {
                state[j] ^= c_prime[j];

                for k in 0..64 {
                    self.a_prime[i][j][k] = F::from_bool((state[j] & (1u64 << k)) > 0);
                }
            }
        }

        let mut last = state[1];
        for i in 0..24 {
            let temp = state[PI[i]];
            state[PI[i]] = last.rotate_left(RHO[i]);
            last = temp;
        }

        for i in 0..5 {
            let state = &mut state[5 * i..];
            let array: [_; 5] = std::array::from_fn(|j| state[j]);

            for j in 0..5 {
                state[j] ^= (!array[(j + 1) % 5]) & array[(j + 2) % 5];

                self.a_prime_prime[i][j][0] = ((state[j] & 0xFFFFFFFFu64) as u32).into();
                self.a_prime_prime[i][j][1] = ((state[j] >> 32) as u32).into();
            }
        }

        self.a_prime_prime_0_0_witness[0] = ((state[0] & 0xFFFFFFFFu64) as u32).into();
        self.a_prime_prime_0_0_witness[1] = ((state[0] >> 32) as u32).into();

        self.rc[0] = ((RC[step] & 0xFFFFFFFFu64) as u32).into();
        self.rc[1] = ((RC[step] >> 32) as u32).into();

        self.rc_witness[0] = ((RC[step] & 0xFFFFFFFFu64) as u32).into();
        self.rc_witness[1] = ((RC[step] >> 32) as u32).into();

        self.a_prime_prime_prime_0_0[0]
            .populate(blu, [(state[0] & 0xFFFFFFFFu64) as u32, (RC[step] & 0xFFFFFFFFu64) as u32]);
        self.a_prime_prime_prime_0_0[1]
            .populate(blu, [(state[0] >> 32) as u32, (RC[step] >> 32) as u32]);

        state[0] ^= RC[step];
    }

    pub fn eval<AB: DTAirBuilder<F = F>>(
        cols: &KeccakCols<AB::Var>,
        builder: &mut AB,
        is_real: impl Into<AB::Expr>,
    ) -> [Word<AB::Expr>; 2] {
        let is_real = is_real.into();

        cols.step_low_one_hot
            .iter()
            .chain(cols.step_high_one_hot.iter())
            .for_each(|b| builder.assert_bool(*b));
        builder.when(is_real.clone()).assert_eq(
            cols.step,
            cols.step_low_one_hot
                .iter()
                .enumerate()
                .map(|(i, b)| *b * AB::F::from_canonical_u32(i as u32))
                .chain(
                    cols.step_high_one_hot
                        .iter()
                        .enumerate()
                        .map(|(i, b)| *b * AB::F::from_canonical_u32((i << 2) as u32)),
                )
                .sum::<AB::Expr>(),
        );
        builder.assert_bool(
            cols.step_low_one_hot
                .iter()
                .map(|b| <AB::Var as Into<AB::Expr>>::into(*b))
                .sum::<AB::Expr>(),
        );
        builder.assert_bool(
            cols.step_high_one_hot
                .iter()
                .map(|b| <AB::Var as Into<AB::Expr>>::into(*b))
                .sum::<AB::Expr>(),
        );

        cols.c.as_flattened().iter().for_each(|c| builder.assert_bool(*c));
        // cols.c_prime.as_flattened().iter().for_each(|c| builder.assert_bool(*c));
        cols.a_prime.as_flattened().as_flattened().iter().for_each(|c| builder.assert_bool(*c));

        for i in 0..5 {
            for j in 0..64 {
                builder.assert_eq(
                    cols.c_prime[i][j],
                    xor(
                        xor(cols.c[i][j], cols.c[(i + 4) % 5][j]),
                        cols.c[(i + 1) % 5][(j + 63) % 64],
                    ),
                );
            }
        }

        for i in 0..5 {
            let c_xor_c_prime: [_; 64] =
                std::array::from_fn(|j| xor(cols.c[i][j], cols.c_prime[i][j]));

            for j in 0..5 {
                let a: [_; 64] =
                    std::array::from_fn(|k| xor(c_xor_c_prime[k].clone(), cols.a_prime[j][i][k]));

                for k in 0..4 {
                    builder.assert_eq(
                        a[k * 16..(k + 1) * 16]
                            .iter()
                            .rev()
                            .fold(AB::Expr::zero(), |acc, a| acc.clone() + acc + a.clone()),
                        cols.a[j][i][k >> 1][k & 0x1],
                    );
                }
            }
        }

        for i in 0..5 {
            for j in 0..64 {
                let sum = (0..5).map(|k| cols.a_prime[k][i][j].into()).sum::<AB::Expr>();
                let diff = sum - cols.c_prime[i][j];
                builder.assert_zero(
                    diff.clone() *
                        (diff.clone() - AB::F::two()) *
                        (diff - AB::F::from_canonical_u32(4)),
                );
            }
        }

        for i in 0..5 {
            let b: [_; 5] = std::array::from_fn(|j| cols.b(i, j));
            for j in 0..5 {
                let a_prime_prime: [_; 64] = std::array::from_fn(|k| {
                    xor(b[j][k], and(not(b[(j + 1) % 5][k]), b[(j + 2) % 5][k]))
                });

                for k in 0..4 {
                    builder.assert_eq(
                        a_prime_prime[k * 16..(k + 1) * 16]
                            .iter()
                            .rev()
                            .fold(AB::Expr::zero(), |acc, a| acc.clone() + acc + a.clone()),
                        cols.a_prime_prime[i][j][k >> 1][k & 0x1],
                    );
                }
            }
        }

        let a_prime_prime_0_0 = [
            CompactWord::into_word(cols.a_prime_prime[0][0][0], cols.a_prime_prime_0_0_witness[0]),
            CompactWord::into_word(cols.a_prime_prime[0][0][1], cols.a_prime_prime_0_0_witness[1]),
        ];

        let one_hot: [AB::Expr; 24] = std::array::from_fn(|i| {
            cols.step_low_one_hot[i & 0x3] * cols.step_high_one_hot[i >> 2]
        });

        for i in 0..4 {
            builder.assert_eq(
                one_hot
                    .iter()
                    .zip(RC.iter())
                    .map(|(b, rc)| {
                        b.clone() * AB::F::from_canonical_u32(((rc >> (16 * i)) & 0xFFFFu64) as u32)
                    })
                    .sum::<AB::Expr>(),
                cols.rc[i >> 1][i & 0x1],
            );
        }

        [
            XorNOperation::<AB::F, U2>::eval(
                &cols.a_prime_prime_prime_0_0[0],
                builder,
                [
                    a_prime_prime_0_0[0].clone(),
                    CompactWord::into_word(cols.rc[0], cols.rc_witness[0]),
                ],
                is_real.clone(),
            ),
            XorNOperation::<AB::F, U2>::eval(
                &cols.a_prime_prime_prime_0_0[1],
                builder,
                [
                    a_prime_prime_0_0[1].clone(),
                    CompactWord::into_word(cols.rc[1], cols.rc_witness[1]),
                ],
                is_real,
            ),
        ]
    }
}

struct B<'a, T> {
    b: &'a [T; 64],
    r: usize,
}

impl<'a, T> Index<usize> for B<'a, T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.b[(index + 64 - self.r) % 64]
    }
}

impl<T> KeccakCols<T> {
    fn b<'a>(&'a self, i: usize, j: usize) -> B<'a, T> {
        let a = j;
        let b = (j + 3 * i) % 5;
        let r = R[a][b];

        B { b: &self.a_prime[a][b], r }
    }
}

pub(super) const PI: [usize; 24] =
    [10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1];

pub(super) const RHO: [u32; 24] =
    [1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44];

pub(crate) const R: [[usize; 5]; 5] = [
    [0, 1, 62, 28, 27],
    [36, 44, 6, 55, 20],
    [3, 10, 43, 25, 39],
    [41, 45, 15, 21, 8],
    [18, 2, 61, 56, 14],
];

const RC: [u64; 24] = [
    0x0000000000000001,
    0x0000000000008082,
    0x800000000000808A,
    0x8000000080008000,
    0x000000000000808B,
    0x0000000080000001,
    0x8000000080008081,
    0x8000000000008009,
    0x000000000000008A,
    0x0000000000000088,
    0x0000000080008009,
    0x000000008000000A,
    0x000000008000808B,
    0x800000000000008B,
    0x8000000000008089,
    0x8000000000008003,
    0x8000000000008002,
    0x8000000000000080,
    0x000000000000800A,
    0x800000008000000A,
    0x8000000080008081,
    0x8000000000008080,
    0x0000000080000001,
    0x8000000080008008,
];

fn xor<Expr: AbstractField>(a: impl Into<Expr>, b: impl Into<Expr>) -> Expr {
    let a = a.into();
    let b = b.into();
    a.clone() + b.clone() - a * b * Expr::two()
}

fn and<Expr: AbstractField>(a: impl Into<Expr>, b: impl Into<Expr>) -> Expr {
    a.into() * b.into()
}

fn not<Expr: AbstractField>(a: impl Into<Expr>) -> Expr {
    Expr::one() - a.into()
}
