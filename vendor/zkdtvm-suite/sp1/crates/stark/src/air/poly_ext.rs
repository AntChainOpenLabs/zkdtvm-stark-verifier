use p3_field::{ExtensionField, Field};

pub trait PolyAirExtendable<const D: usize>: Field {
    type EF: ExtensionField<Self>
        + From<Self>
        + core::ops::Add<Output = Self::EF>
        + core::ops::Sub<Output = Self::EF>
        + core::ops::Mul<Output = Self::EF>
        + core::ops::Neg<Output = Self::EF>
        + Clone
        + Send
        + Sync;

    fn ef_from_base(f: Self) -> Self::EF {
        Self::EF::from(f)
    }
}

#[cfg(feature = "babybear")]
impl PolyAirExtendable<4> for p3_baby_bear::BabyBear {
    type EF = p3_field::extension::BinomialExtensionField<p3_baby_bear::BabyBear, 4>;
}

#[cfg(all(feature = "koalabear", not(feature = "ext5")))]
impl PolyAirExtendable<4> for p3_koala_bear::KoalaBear {
    type EF = p3_field::extension::BinomialExtensionField<p3_koala_bear::KoalaBear, 4>;
}

#[cfg(feature = "ext5")]
impl PolyAirExtendable<5> for p3_koala_bear::KoalaBear {
    type EF = p3_field::extension::QuinticTrinomialExtensionField<p3_koala_bear::KoalaBear>;
}
