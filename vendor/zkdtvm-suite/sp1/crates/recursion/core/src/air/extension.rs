use dt_stark::air::BinomialExtension;
#[cfg(not(feature = "ext5"))]
use p3_field::extension::{BinomialExtensionField, BinomiallyExtendable};
#[cfg(feature = "ext5")]
use p3_field::extension::{QuinticTrinomialExtendable, QuinticTrinomialExtensionField};
use p3_field::{AbstractExtensionField, Field};

use super::Block;

#[cfg(not(feature = "ext5"))]
use crate::runtime::D;

pub trait BinomialExtensionUtils<T> {
    fn from_block(block: Block<T>) -> Self;

    fn as_block(&self) -> Block<T>;
}

impl<T: Clone> BinomialExtensionUtils<T> for BinomialExtension<T> {
    fn from_block(block: Block<T>) -> Self {
        Self(block.0)
    }

    fn as_block(&self) -> Block<T> {
        Block(self.0.clone())
    }
}

#[cfg(not(feature = "ext5"))]
impl<AF> BinomialExtensionUtils<AF> for BinomialExtensionField<AF, D>
where
    AF: Field,
    AF::F: BinomiallyExtendable<D>,
{
    fn from_block(block: Block<AF>) -> Self {
        Self::from_base_slice(&block.0)
    }

    fn as_block(&self) -> Block<AF> {
        Block(self.as_base_slice().try_into().unwrap())
    }
}

#[cfg(feature = "ext5")]
impl<AF> BinomialExtensionUtils<AF> for QuinticTrinomialExtensionField<AF>
where
    AF: Field,
    AF::F: QuinticTrinomialExtendable,
{
    fn from_block(block: Block<AF>) -> Self {
        Self::from_base_slice(&block.0)
    }

    fn as_block(&self) -> Block<AF> {
        Block(self.as_base_slice().try_into().unwrap())
    }
}
