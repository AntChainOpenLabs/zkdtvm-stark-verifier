use dt_recursion_compiler::ir::{Config, Ext, Felt, SymbolicExt};
use dt_stark::air::FullAirBuilder;
use p3_field::{AbstractField, ExtensionField, Field};
use p3_matrix::dense::RowMajorMatrixView;

pub struct RecursivePolyAirPrecomputeRowBuilder<'a, C: Config> {
    pub preprocessed: &'a [SymbolicExt<C::F, C::EF>],
    pub main: &'a [SymbolicExt<C::F, C::EF>],
    pub beta_powers: &'a [SymbolicExt<C::F, C::EF>],
    pub public: &'a [Felt<C::F>],
    pub alpha: SymbolicExt<C::F, C::EF>,
    pub beta_septix: SymbolicExt<C::F, C::EF>,
    pub row: &'a mut Vec<SymbolicExt<C::F, C::EF>>,
    pub col_index: usize,
}

impl<'a, C: Config> FullAirBuilder for RecursivePolyAirPrecomputeRowBuilder<'a, C>
where
    C::F: Field,
    C::EF: ExtensionField<C::F>,
{
    type F = C::F;
    type EF = C::EF;
    type VarBase = Felt<C::F>;
    type VarMaybeExt = SymbolicExt<C::F, C::EF>;
    type VarExt = SymbolicExt<C::F, C::EF>;
    type MatMaybeExt = RowMajorMatrixView<'a, SymbolicExt<C::F, C::EF>>;
    type MatExt = RowMajorMatrixView<'a, SymbolicExt<C::F, C::EF>>;

    fn preprocessed(&self) -> &[Self::VarMaybeExt] {
        self.preprocessed
    }

    fn main(&self) -> &[Self::VarMaybeExt] {
        self.main
    }

    fn public(&self) -> &[Self::VarBase] {
        self.public
    }

    fn alpha(&self) -> Self::VarExt {
        self.alpha
    }

    fn beta_powers(&self) -> &[Self::VarExt] {
        self.beta_powers
    }

    fn beta_septix(&self) -> Self::VarExt {
        self.beta_septix
    }

    fn retain_precomputed(&mut self, x: Self::VarExt) {
        if self.col_index < self.row.len() {
            self.row[self.col_index] = x;
        } else {
            self.row.push(x);
        }
        self.col_index += 1;
    }

    fn precomputed(&self) -> Self::MatExt {
        unreachable!("precomputed is not ready in precompute linear combination phase")
    }

    fn reserved_poly(&self) -> Self::MatMaybeExt {
        unreachable!("reserved_poly should not be used in precompute linear combination phase")
    }

    fn local_lookup(&mut self, _: Self::VarMaybeExt, _: bool) {
        unreachable!("local_lookup is not ready in precompute linear combination phase")
    }

    fn is_first_row(&self) -> Self::VarMaybeExt {
        unreachable!("is_first_row should not be used in precompute linear combination phase")
    }

    fn is_last_row(&self) -> Self::VarMaybeExt {
        unreachable!("is_last_row should not be used in precompute linear combination phase")
    }

    fn is_transition(&self) -> Self::VarMaybeExt {
        unreachable!("is_transition should not be used in precompute linear combination phase")
    }

    fn from_ef(ef: Self::EF) -> Self::VarExt {
        SymbolicExt::from_f(ef)
    }

    fn assert_zero<I: Into<Self::VarMaybeExt>>(&mut self, _: I) {
        unreachable!("assert_zero should not be used in precompute linear combination phase")
    }

    fn assert_zero_ext<I: Into<Self::VarExt>>(&mut self, _: I) {
        unreachable!("assert_zero_ext should not be used in precompute linear combination phase")
    }
}
