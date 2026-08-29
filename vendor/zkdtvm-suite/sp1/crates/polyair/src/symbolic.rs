//! Symbolic execution for AIR constraint degree analysis.
//!
//! This module provides [`SymbolicAirBuilder`] which implements [`FullAirBuilder`]
//! using symbolic expressions. This allows analyzing constraint degrees and
//! counting constraints before actual proving.

use core::fmt::{self, Debug};

use std::rc::Rc;

use core::{
    iter::{Product, Sum},
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};
use dt_stark::air::PolyAirExtendable;
use p3_air::BaseAir;
use p3_field::{AbstractExtensionField, AbstractField, Field};
use p3_matrix::dense::{DenseMatrix, RowMajorMatrix};

use dt_stark::air::{FullAir, FullAirBuilder, MachineAir, PairCol};

/// Trait for types that can report their degree multiple.
///
/// Used during symbolic execution to track the degree of constraint expressions.
pub trait DegreeMultiple: Clone + Debug {
    /// Returns the degree multiple of this expression.
    fn degree_multiple(&self) -> usize;
}

/// A symbolic expression tree for tracking constraint degrees.
///
/// This enum represents expressions symbolically, allowing degree analysis
/// without concrete evaluation.
#[derive(Clone)]
pub enum SymbolicExpression<VAR, F: Field + PolyAirExtendable<D>, const D: usize> {
    /// A variable reference.
    VARiable(VAR),
    /// A base field constant.
    Constant(F),
    /// An extension field constant.
    ConstantExt(<F as PolyAirExtendable<D>>::EF),
    /// Addition of two expressions.
    Add { x: Rc<Self>, y: Rc<Self>, degree_multiple: usize },
    /// Subtraction of two expressions.
    Sub { x: Rc<Self>, y: Rc<Self>, degree_multiple: usize },
    /// Negation of an expression.
    Neg { x: Rc<Self>, degree_multiple: usize },
    /// Multiplication of two expressions.
    Mul { x: Rc<Self>, y: Rc<Self>, degree_multiple: usize },
}

impl<VAR, F: Field + PolyAirExtendable<D>, const D: usize> SymbolicExpression<VAR, F, D> {
    pub fn cache_key(&self) -> Option<[u64; 3]> {
        match self {
            SymbolicExpression::Add { x, y, degree_multiple: _ } => Some([
                0,
                Rc::<SymbolicExpression<VAR, F, D>>::as_ptr(x) as u64,
                Rc::<SymbolicExpression<VAR, F, D>>::as_ptr(y) as u64,
            ]),
            SymbolicExpression::Sub { x, y, degree_multiple: _ } => Some([
                1,
                Rc::<SymbolicExpression<VAR, F, D>>::as_ptr(x) as u64,
                Rc::<SymbolicExpression<VAR, F, D>>::as_ptr(y) as u64,
            ]),
            SymbolicExpression::Neg { x, degree_multiple: _ } => {
                Some([2, Rc::<SymbolicExpression<VAR, F, D>>::as_ptr(x) as u64, 0])
            }
            SymbolicExpression::Mul { x, y, degree_multiple: _ } => Some([
                3,
                Rc::<SymbolicExpression<VAR, F, D>>::as_ptr(x) as u64,
                Rc::<SymbolicExpression<VAR, F, D>>::as_ptr(y) as u64,
            ]),
            _ => None,
        }
    }
    fn fmt_with_depth(&self, f: &mut fmt::Formatter<'_>, depth: usize) -> fmt::Result
    where
        VAR: fmt::Debug,
        F: fmt::Debug,
    {
        for _ in 0..depth {
            write!(f, "\t")?;
        }

        match self {
            SymbolicExpression::VARiable(v) => write!(f, "VARiable({v:?})"),
            SymbolicExpression::Constant(c) => write!(f, "Constant({c:?})"),
            SymbolicExpression::ConstantExt(c) => write!(f, "ConstantExt({c:?})"),
            SymbolicExpression::Add { x, y, degree_multiple } => {
                writeln!(f, "Add(degree_multiple={degree_multiple})")?;
                x.fmt_with_depth(f, depth + 1)?;
                writeln!(f)?;
                y.fmt_with_depth(f, depth + 1)
            }
            SymbolicExpression::Sub { x, y, degree_multiple } => {
                writeln!(f, "Sub(degree_multiple={degree_multiple})")?;
                x.fmt_with_depth(f, depth + 1)?;
                writeln!(f)?;
                y.fmt_with_depth(f, depth + 1)
            }
            SymbolicExpression::Neg { x, degree_multiple } => {
                writeln!(f, "Neg(degree_multiple={degree_multiple})")?;
                x.fmt_with_depth(f, depth + 1)
            }
            SymbolicExpression::Mul { x, y, degree_multiple } => {
                writeln!(f, "Mul(degree_multiple={degree_multiple})")?;
                x.fmt_with_depth(f, depth + 1)?;
                writeln!(f)?;
                y.fmt_with_depth(f, depth + 1)
            }
        }
    }
}

impl<VAR, F: Field + PolyAirExtendable<D>, const D: usize> fmt::Debug
    for SymbolicExpression<VAR, F, D>
where
    VAR: fmt::Debug,
    F: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_with_depth(f, 0)
    }
}

unsafe impl<VAR, F: Field + PolyAirExtendable<D>, const D: usize> Send
    for SymbolicExpression<VAR, F, D>
{
}

unsafe impl<VAR, F: Field + PolyAirExtendable<D>, const D: usize> Sync
    for SymbolicExpression<VAR, F, D>
{
}

impl<VAR: DegreeMultiple, F: Field + PolyAirExtendable<D>, const D: usize>
    SymbolicExpression<VAR, F, D>
{
    /// Iterates over all variables in this expression.
    pub fn iter_all_var(&self) -> Vec<&VAR> {
        match self {
            SymbolicExpression::VARiable(x) => Some(x).into_iter().collect(),
            SymbolicExpression::Constant(_) | SymbolicExpression::ConstantExt(_) => {
                None.into_iter().collect()
            }
            SymbolicExpression::Add { x, y, degree_multiple: _ } |
            SymbolicExpression::Sub { x, y, degree_multiple: _ } |
            SymbolicExpression::Mul { x, y, degree_multiple: _ } => {
                x.iter_all_var().into_iter().chain(y.iter_all_var()).collect()
            }
            SymbolicExpression::Neg { x, degree_multiple: _ } => x.iter_all_var(),
        }
    }
}

impl<VAR: DegreeMultiple, F: Field + PolyAirExtendable<D>, const D: usize> DegreeMultiple
    for SymbolicExpression<VAR, F, D>
{
    fn degree_multiple(&self) -> usize {
        match self {
            Self::VARiable(v) => v.degree_multiple(),
            Self::Constant(_) => 0,
            Self::ConstantExt(_) => 0,
            Self::Add { degree_multiple, .. } |
            Self::Sub { degree_multiple, .. } |
            Self::Neg { degree_multiple, .. } |
            Self::Mul { degree_multiple, .. } => *degree_multiple,
        }
    }
}

impl<VAR, F: Field + PolyAirExtendable<D>, const D: usize> Default
    for SymbolicExpression<VAR, F, D>
{
    fn default() -> Self {
        Self::Constant(F::zero())
    }
}

impl<VAR, F: Field + PolyAirExtendable<D>, const D: usize> From<F>
    for SymbolicExpression<VAR, F, D>
{
    fn from(value: F) -> Self {
        Self::Constant(value)
    }
}

impl<VAR: DegreeMultiple, F: Field + PolyAirExtendable<D>, const D: usize> AbstractField
    for SymbolicExpression<VAR, F, D>
{
    type F = F;

    fn zero() -> Self {
        Self::Constant(F::zero())
    }
    fn one() -> Self {
        Self::Constant(F::one())
    }
    fn two() -> Self {
        Self::Constant(F::two())
    }
    fn neg_one() -> Self {
        Self::Constant(F::neg_one())
    }

    #[inline]
    fn from_f(f: Self::F) -> Self {
        Self::Constant(f)
    }

    fn from_bool(b: bool) -> Self {
        Self::Constant(F::from_bool(b))
    }

    fn from_canonical_u8(n: u8) -> Self {
        Self::Constant(F::from_canonical_u8(n))
    }

    fn from_canonical_u16(n: u16) -> Self {
        Self::Constant(F::from_canonical_u16(n))
    }

    fn from_canonical_u32(n: u32) -> Self {
        Self::Constant(F::from_canonical_u32(n))
    }

    fn from_canonical_u64(n: u64) -> Self {
        Self::Constant(F::from_canonical_u64(n))
    }

    fn from_canonical_usize(n: usize) -> Self {
        Self::Constant(F::from_canonical_usize(n))
    }

    fn from_wrapped_u32(n: u32) -> Self {
        Self::Constant(F::from_wrapped_u32(n))
    }

    fn from_wrapped_u64(n: u64) -> Self {
        Self::Constant(F::from_wrapped_u64(n))
    }

    fn generator() -> Self {
        Self::Constant(F::generator())
    }
}

impl<VAR, F: Field + PolyAirExtendable<D>, const D: usize> SymbolicExpression<VAR, F, D> {
    pub fn from_ext(value: <F as PolyAirExtendable<D>>::EF) -> Self {
        Self::ConstantExt(value)
    }

    fn is_zero_const(&self) -> bool {
        match self {
            Self::Constant(c) => c.is_zero(),
            Self::ConstantExt(c) => c.is_zero(),
            _ => false,
        }
    }

    fn is_one_const(&self) -> bool {
        match self {
            Self::Constant(c) => c.is_one(),
            Self::ConstantExt(c) => c.is_one(),
            _ => false,
        }
    }
}

impl<VAR: DegreeMultiple, F: Field + PolyAirExtendable<D>, const D: usize, T> Add<T>
    for SymbolicExpression<VAR, F, D>
where
    T: Into<Self>,
{
    type Output = Self;

    fn add(self, rhs: T) -> Self {
        match (self, rhs.into()) {
            (Self::Constant(lhs), Self::Constant(rhs)) => Self::Constant(lhs + rhs),
            (Self::Constant(lhs), Self::ConstantExt(rhs)) => {
                Self::ConstantExt(F::ef_from_base(lhs) + rhs)
            }
            (Self::ConstantExt(lhs), Self::Constant(rhs)) => Self::ConstantExt(lhs + rhs),
            (Self::ConstantExt(lhs), Self::ConstantExt(rhs)) => Self::ConstantExt(lhs + rhs),
            (lhs, rhs) if lhs.is_zero_const() => rhs,
            (lhs, rhs) if rhs.is_zero_const() => lhs,
            (lhs, rhs) => Self::Add {
                degree_multiple: lhs.degree_multiple().max(rhs.degree_multiple()),
                x: Rc::new(lhs),
                y: Rc::new(rhs),
            },
        }
    }
}

impl<VAR: DegreeMultiple, F: Field + PolyAirExtendable<D>, const D: usize, T> AddAssign<T>
    for SymbolicExpression<VAR, F, D>
where
    T: Into<Self>,
{
    fn add_assign(&mut self, rhs: T) {
        *self = self.clone() + rhs.into();
    }
}

impl<VAR: DegreeMultiple, F: Field + PolyAirExtendable<D>, const D: usize, T> Sum<T>
    for SymbolicExpression<VAR, F, D>
where
    T: Into<Self>,
{
    fn sum<I: Iterator<Item = T>>(iter: I) -> Self {
        iter.map(Into::into).reduce(|x, y| x + y).unwrap_or_default()
    }
}

impl<VAR: DegreeMultiple, F: Field + PolyAirExtendable<D>, const D: usize, T: Into<Self>> Sub<T>
    for SymbolicExpression<VAR, F, D>
{
    type Output = Self;

    fn sub(self, rhs: T) -> Self {
        match (self, rhs.into()) {
            (Self::Constant(lhs), Self::Constant(rhs)) => Self::Constant(lhs - rhs),
            (Self::Constant(lhs), Self::ConstantExt(rhs)) => {
                Self::ConstantExt(F::ef_from_base(lhs) - rhs)
            }
            (Self::ConstantExt(lhs), Self::Constant(rhs)) => Self::ConstantExt(lhs - rhs),
            (Self::ConstantExt(lhs), Self::ConstantExt(rhs)) => Self::ConstantExt(lhs - rhs),
            (lhs, rhs) if rhs.is_zero_const() => lhs,
            (lhs, rhs) if lhs.is_zero_const() => -rhs,
            (lhs, rhs) => Self::Sub {
                degree_multiple: lhs.degree_multiple().max(rhs.degree_multiple()),
                x: Rc::new(lhs),
                y: Rc::new(rhs),
            },
        }
    }
}

impl<VAR: DegreeMultiple, F: Field + PolyAirExtendable<D>, const D: usize, T> SubAssign<T>
    for SymbolicExpression<VAR, F, D>
where
    T: Into<Self>,
{
    fn sub_assign(&mut self, rhs: T) {
        *self = self.clone() - rhs.into();
    }
}

impl<VAR: DegreeMultiple, F: Field + PolyAirExtendable<D>, const D: usize> Neg
    for SymbolicExpression<VAR, F, D>
{
    type Output = Self;

    fn neg(self) -> Self {
        match self {
            Self::Constant(c) => Self::Constant(-c),
            Self::ConstantExt(c) => Self::ConstantExt(-c),
            expr => Self::Neg { degree_multiple: expr.degree_multiple(), x: Rc::new(expr) },
        }
    }
}

impl<VAR: DegreeMultiple, F: Field + PolyAirExtendable<D>, const D: usize, T: Into<Self>> Mul<T>
    for SymbolicExpression<VAR, F, D>
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self {
        match (self, rhs.into()) {
            (Self::Constant(lhs), Self::Constant(rhs)) => Self::Constant(lhs * rhs),
            (Self::Constant(lhs), Self::ConstantExt(rhs)) => {
                Self::ConstantExt(F::ef_from_base(lhs) * rhs)
            }
            (Self::ConstantExt(lhs), Self::Constant(rhs)) => Self::ConstantExt(lhs * rhs),
            (Self::ConstantExt(lhs), Self::ConstantExt(rhs)) => Self::ConstantExt(lhs * rhs),
            (lhs, _) if lhs.is_zero_const() => lhs,
            (_, rhs) if rhs.is_zero_const() => rhs,
            (lhs, rhs) if lhs.is_one_const() => rhs,
            (lhs, rhs) if rhs.is_one_const() => lhs,
            (lhs, rhs) => Self::Mul {
                degree_multiple: lhs.degree_multiple() + rhs.degree_multiple(),
                x: Rc::new(lhs),
                y: Rc::new(rhs),
            },
        }
    }
}

impl<VAR: DegreeMultiple, F: Field + PolyAirExtendable<D>, const D: usize, T> MulAssign<T>
    for SymbolicExpression<VAR, F, D>
where
    T: Into<Self>,
{
    fn mul_assign(&mut self, rhs: T) {
        *self = self.clone() * rhs.into();
    }
}

impl<VAR: DegreeMultiple, F: Field + PolyAirExtendable<D>, const D: usize, T: Into<Self>> Product<T>
    for SymbolicExpression<VAR, F, D>
{
    fn product<I: Iterator<Item = T>>(iter: I) -> Self {
        iter.map(Into::into).reduce(|x, y| x * y).unwrap_or(F::one().into())
    }
}

/// Rotation indicator for accessing adjacent rows.
#[derive(Debug, Clone, Copy)]
pub enum Rotation {
    /// Current row.
    Local,
    /// Next row.
    Next,
}

/// Symbolic variable types for AIR constraints.
#[derive(Debug, Clone, Copy)]
pub enum SymbolicVar {
    /// Preprocessed column at index.
    Preprocessed(usize),
    /// Main trace column at index.
    Main(usize),
    /// Public value at index.
    Public(usize),
    /// Alpha challenge.
    Alpha,
    /// Beta power at index.
    BetaPowers(usize),
    /// Beta septix challenge.
    BetaSeptix,
    /// Precomputed linear combination at index with rotation.
    Precomputed(usize, Rotation),
    /// Reserved polynomial at index with rotation.
    ReservedPoly(usize, Rotation),
    /// First row selector.
    IsFirstRow,
    /// Last row selector.
    IsLastRow,
    /// Transition selector.
    IsTransition,
}

impl DegreeMultiple for SymbolicVar {
    fn degree_multiple(&self) -> usize {
        match self {
            SymbolicVar::Preprocessed(_) |
            SymbolicVar::Main(_) |
            SymbolicVar::Precomputed(_, _) |
            SymbolicVar::ReservedPoly(_, _) |
            SymbolicVar::IsFirstRow |
            SymbolicVar::IsLastRow |
            SymbolicVar::IsTransition => 1,
            _ => 0,
        }
    }
}

/// Information about a lookup argument during symbolic execution.
#[derive(Debug, Clone)]
pub struct LookupInfo<F: Field + PolyAirExtendable<D>, const D: usize> {
    /// The multiplicity expression.
    pub multiplicity: SymbolicExpression<SymbolicVar, F, D>,
    /// Whether this is a send (true) or receive (false).
    pub is_send: bool,
}

/// Symbolic AIR builder for constraint degree analysis.
///
/// This builder records all operations symbolically, allowing analysis of
/// constraint degrees and counts before actual proving.
#[derive(Debug, Clone)]
pub struct SymbolicAirBuilder<F: Field + PolyAirExtendable<D>, const D: usize> {
    pub preprocessed: Vec<SymbolicExpression<SymbolicVar, F, D>>,
    pub main: Vec<SymbolicExpression<SymbolicVar, F, D>>,
    pub public: Vec<SymbolicExpression<SymbolicVar, F, D>>,
    pub reserved_poly_input: DenseMatrix<SymbolicExpression<SymbolicVar, F, D>>,
    pub precomputed_lc_input: DenseMatrix<SymbolicExpression<SymbolicVar, F, D>>,
    pub beta_powers: Vec<SymbolicExpression<SymbolicVar, F, D>>,

    pub reserved_poly_output: Vec<PairCol>,
    pub precomputed_lc_output: Vec<SymbolicExpression<SymbolicVar, F, D>>,
    pub lookup_infos: Vec<LookupInfo<F, D>>,
    pub gate: Vec<SymbolicExpression<SymbolicVar, F, D>>,
}

impl<F: Field + PolyAirExtendable<D>, const D: usize> SymbolicAirBuilder<F, D> {
    /// Creates a new empty symbolic builder.
    pub fn new_empty() -> Self {
        Self {
            preprocessed: vec![],
            main: vec![],
            public: vec![],
            beta_powers: vec![],
            reserved_poly_input: DenseMatrix::new(vec![], 0),
            precomputed_lc_input: DenseMatrix::new(vec![], 0),

            reserved_poly_output: vec![],
            precomputed_lc_output: vec![],
            lookup_infos: vec![],
            gate: vec![],
        }
    }

    /// Sets the preprocessed trace width.
    pub fn with_preprocessed_width(&mut self, preprocessed_width: usize) {
        self.preprocessed = (0..preprocessed_width)
            .map(|i| SymbolicExpression::VARiable(SymbolicVar::Preprocessed(i)))
            .collect();
    }

    /// Sets the main trace width.
    pub fn with_main_width(&mut self, main_width: usize) {
        self.main =
            (0..main_width).map(|i| SymbolicExpression::VARiable(SymbolicVar::Main(i))).collect();
    }

    /// Sets the public values width.
    pub fn with_public_width(&mut self, public_width: usize) {
        self.public = (0..public_width)
            .map(|i| SymbolicExpression::VARiable(SymbolicVar::Public(i)))
            .collect();
    }

    /// Sets the maximum beta power needed.
    pub fn width_max_beta_power(&mut self, max_beta_power: usize) {
        self.beta_powers = (0..(max_beta_power + 1))
            .map(|i| SymbolicExpression::VARiable(SymbolicVar::BetaPowers(i)))
            .collect();
    }

    /// Generates the reserved polynomial input matrix.
    pub fn generete_reserved_poly_input(&mut self) {
        let values: Vec<_> = (0..self.reserved_poly_output.len())
            .map(|i| SymbolicExpression::VARiable(SymbolicVar::ReservedPoly(i, Rotation::Local)))
            // .chain((0..self.reserved_poly_output.len()).map(|i| {
            //     SymbolicExpression::VARiable(SymbolicVar::ReservedPoly(i, Rotation::Next))
            // }))
            .collect();
        self.reserved_poly_input = RowMajorMatrix::new(values, self.reserved_poly_output.len())
    }

    /// Generates the precomputed LC input matrix.
    pub fn generate_precomputed_lc_input(&mut self) {
        let values: Vec<_> = (0..self.precomputed_lc_output.len())
            .map(|i| SymbolicExpression::VARiable(SymbolicVar::Precomputed(i, Rotation::Local)))
            // .chain((0..self.precomputed_lc_output.len()).map(|i| {
            //     SymbolicExpression::VARiable(SymbolicVar::Precomputed(i, Rotation::Next))
            // }))
            .collect();
        self.precomputed_lc_input = RowMajorMatrix::new(values, self.precomputed_lc_output.len())
    }

    /// Returns the maximum degree of all constraints.
    pub fn get_max_degree(&self) -> usize {
        for (i, precomputed) in self.precomputed_lc_output.iter().enumerate() {
            assert!(
                precomputed.degree_multiple() == 1,
                "degree != 1 for precomputed_lc[{i}] {:?}",
                self.precomputed_lc_output[i]
            )
        }
        for (i, lookup) in self.lookup_infos.iter().enumerate() {
            assert!(
                lookup.multiplicity.degree_multiple() <= 1,
                "degree > 1 for lookup[{i}].multiplicity"
            )
        }
        let mut max_degree = 0;
        for (i, gate) in self.gate.iter().enumerate() {
            if gate.degree_multiple() <= 1 {
                tracing::debug!("degree <= 1 for gate[{i}]");
            }
            // if gate.degree_multiple() > 3 {
            //     tracing::error!("degree > 3 for gate[{i}]: {:?}", self.gate[i]);
            // }
            max_degree = max_degree.max(gate.degree_multiple());
        }
        max_degree
    }

    /// Returns the number of constraints given a batch size.
    pub fn get_num_constraint(&self, batch_size: usize) -> usize {
        // For the cumulative sum constraint computed separately in the sumcheck prover.
        self.gate.len() + self.lookup_infos.len().div_ceil(batch_size) + 1
    }

    /// Build a symbolic representation from an AIR.
    pub fn from_air<A: FullAir<Self> + MachineAir<F>>(air: &A) -> Self {
        assert_eq!(FullAir::width(air), BaseAir::width(air));
        let mut builder = Self::new_empty();
        builder.with_preprocessed_width(air.preprocessed_width());
        builder.with_main_width(FullAir::width(air));
        builder.with_public_width(air.num_public_values());

        builder.width_max_beta_power(air.required_max_beta_power());

        builder.reserved_poly_output = air.reserved_poly();
        builder.generete_reserved_poly_input();

        air.precompute_lc(&mut builder);
        builder.generate_precomputed_lc_input();

        air.lookup(&mut builder);
        air.eval(&mut builder);

        builder
    }
}

impl<F: Field + PolyAirExtendable<D>, const D: usize> FullAirBuilder for SymbolicAirBuilder<F, D> {
    type F = F;

    type EF = F;

    type VarBase = SymbolicExpression<SymbolicVar, F, D>;

    type VarMaybeExt = SymbolicExpression<SymbolicVar, F, D>;

    type VarExt = SymbolicExpression<SymbolicVar, F, D>;

    type MatMaybeExt = RowMajorMatrix<Self::VarMaybeExt>;

    type MatExt = RowMajorMatrix<Self::VarExt>;

    fn preprocessed(&self) -> &[Self::VarMaybeExt] {
        &self.preprocessed
    }

    fn main(&self) -> &[Self::VarMaybeExt] {
        &self.main
    }

    fn public(&self) -> &[Self::VarBase] {
        &self.public
    }

    fn alpha(&self) -> Self::VarExt {
        SymbolicExpression::VARiable(SymbolicVar::Alpha)
    }

    fn beta_powers(&self) -> &[Self::VarExt] {
        &self.beta_powers
    }

    fn beta_septix(&self) -> Self::VarExt {
        SymbolicExpression::VARiable(SymbolicVar::BetaSeptix)
    }

    fn retain_precomputed(&mut self, x: Self::VarExt) {
        self.precomputed_lc_output.push(x);
    }

    fn precomputed(&self) -> Self::MatExt {
        self.precomputed_lc_input.clone()
    }

    fn reserved_poly(&self) -> Self::MatMaybeExt {
        self.reserved_poly_input.clone()
    }

    fn local_lookup(&mut self, multiplicity: Self::VarMaybeExt, is_send: bool) {
        self.lookup_infos.push(LookupInfo { multiplicity, is_send });
    }

    fn is_first_row(&self) -> Self::VarMaybeExt {
        SymbolicExpression::VARiable(SymbolicVar::IsFirstRow)
    }

    fn is_last_row(&self) -> Self::VarMaybeExt {
        SymbolicExpression::VARiable(SymbolicVar::IsLastRow)
    }

    fn is_transition(&self) -> Self::VarMaybeExt {
        SymbolicExpression::VARiable(SymbolicVar::IsTransition)
    }

    fn from_ef(ef: Self::EF) -> Self::VarExt {
        SymbolicExpression::Constant(ef)
    }

    fn pack_ext_limbs(limbs: &[Self::VarMaybeExt]) -> Self::VarExt {
        let degree = <<F as PolyAirExtendable<D>>::EF as AbstractExtensionField<F>>::D;
        assert!(
            !limbs.is_empty() && limbs.len() <= degree,
            "extension limb count must be in 1..={degree}, got {}",
            limbs.len()
        );
        let theta = SymbolicExpression::from_ext(
            <<F as PolyAirExtendable<D>>::EF as AbstractExtensionField<F>>::monomial(1),
        );
        let mut limbs = limbs.iter().rev();
        let mut packed = limbs.next().expect("checked non-empty").clone();
        for limb in limbs {
            packed = theta.clone() * packed + limb.clone();
        }
        packed
    }

    fn assert_zero<I: Into<Self::VarMaybeExt>>(&mut self, x: I) {
        self.gate.push(x.into());
    }

    fn assert_zero_ext<I: Into<Self::VarExt>>(&mut self, x: I) {
        self.gate.push(x.into());
    }
}
