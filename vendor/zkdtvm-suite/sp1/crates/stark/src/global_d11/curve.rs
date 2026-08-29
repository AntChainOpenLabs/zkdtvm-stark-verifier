use p3_field::{Field, PrimeField32};

use super::{
    constants::CURVE_B_COEFFICIENTS,
    field::{D11Sparse7, D11},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct D11AffinePointV1<F: Field> {
    pub x: D11<F>,
    pub y: D11<F>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct D11ProjectivePointV1<F: Field> {
    pub x: D11<F>,
    pub y: D11<F>,
    pub z: D11<F>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProjectivePointError {
    AllZeroTriple,
    InvalidInfinityEncoding,
    OffCurve,
}

#[must_use]
pub fn curve_b<F: PrimeField32>() -> D11<F> {
    D11::from_canonical_u32(CURVE_B_COEFFICIENTS)
}

impl<F: PrimeField32> D11AffinePointV1<F> {
    #[must_use]
    pub fn is_on_curve(&self) -> bool {
        self.y.square() == self.x.square() * self.x - self.x * F::from_canonical_u32(3) + curve_b()
    }

    #[must_use]
    pub fn negated(&self) -> Self {
        Self { x: self.x, y: -self.y }
    }

    #[must_use]
    pub fn to_projective(&self) -> D11ProjectivePointV1<F> {
        D11ProjectivePointV1 { x: self.x, y: self.y, z: D11::one() }
    }
}

impl<F: PrimeField32> D11ProjectivePointV1<F> {
    #[must_use]
    pub fn identity() -> Self {
        Self { x: D11::zero(), y: D11::one(), z: D11::zero() }
    }

    #[must_use]
    pub fn is_zero_triple(&self) -> bool {
        self.x.is_zero() && self.y.is_zero() && self.z.is_zero()
    }

    /// Projective infinity has many legal scalings: `X=Z=0,Y!=0`.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.x.is_zero() && self.z.is_zero() && !self.y.is_zero()
    }

    #[must_use]
    pub fn is_on_curve(&self) -> bool {
        if self.is_zero_triple() {
            return false;
        }
        let z2 = self.z.square();
        let left = self.y.square() * self.z;
        let right = self.x.square() * self.x - self.x * z2 * F::from_canonical_u32(3) +
            curve_b::<F>() * z2 * self.z;
        left == right
    }

    pub fn validate(&self) -> Result<(), ProjectivePointError> {
        if self.is_zero_triple() {
            return Err(ProjectivePointError::AllZeroTriple);
        }
        if self.z.is_zero() && !self.is_identity() {
            return Err(ProjectivePointError::InvalidInfinityEncoding);
        }
        if !self.is_on_curve() {
            return Err(ProjectivePointError::OffCurve);
        }
        Ok(())
    }

    #[must_use]
    pub fn negated(&self) -> Self {
        Self { x: self.x, y: -self.y, z: self.z }
    }

    #[must_use]
    pub fn rescaled(&self, scale: D11<F>) -> Self {
        Self { x: self.x * scale, y: self.y * scale, z: self.z * scale }
    }

    pub fn to_affine(&self) -> Result<Option<D11AffinePointV1<F>>, ProjectivePointError> {
        self.validate()?;
        if self.is_identity() {
            return Ok(None);
        }
        let inverse_z = self.z.inverse();
        Ok(Some(D11AffinePointV1 { x: self.x * inverse_z, y: self.y * inverse_z }))
    }

    #[must_use]
    pub fn equivalent(&self, other: &Self) -> bool {
        if self.validate().is_err() || other.validate().is_err() {
            return false;
        }
        if self.is_identity() || other.is_identity() {
            return self.is_identity() && other.is_identity();
        }
        self.x * other.z == other.x * self.z && self.y * other.z == other.y * self.z
    }

    /// Complete homogeneous addition for `a=-3`, RCB Algorithm 4.  This is
    /// distinct from the mixed formula used in each Global row.
    #[must_use]
    pub fn add_complete(&self, other: &Self) -> Self {
        self.add_complete_with_products(other).0
    }

    /// Complete homogeneous addition together with the exact products used by
    /// the native D11 aggregate AIR.  Keeping witness products and the output
    /// under one formula authority avoids recomputing the addition in tracegen.
    #[must_use]
    pub fn add_complete_with_products(&self, other: &Self) -> (Self, [D11<F>; 6], [D11<F>; 6]) {
        let xx = self.x * other.x;
        let yy = self.y * other.y;
        let zz = self.z * other.z;
        let xy_product = (self.x + self.y) * (other.x + other.y);
        let yz_product = (self.y + self.z) * (other.y + other.z);
        let xz_product = (self.x + self.z) * (other.x + other.z);
        let xy_pairs = xy_product - (xx + yy);
        let yz_pairs = yz_product - (yy + zz);
        let xz_pairs = xz_product - (xx + zz);

        let bzz_part = xz_pairs - zz.mul_by_z_plus_36();
        let bzz3_part = bzz_part.double();
        let bzz3_part = bzz3_part + bzz_part;
        let yy_m_bzz3 = yy - bzz3_part;
        let yy_p_bzz3 = yy + bzz3_part;

        let zz3 = zz.double() + zz;
        let bxz_part = xz_pairs.mul_by_z_plus_36() - (zz3 + xx);
        let bxz3_part = bxz_part.double() + bxz_part;
        let xx3_m_zz3 = xx.double() + xx - zz3;

        let final_products = [
            yy_p_bzz3 * xy_pairs,
            yz_pairs * bxz3_part,
            yy_p_bzz3 * yy_m_bzz3,
            xx3_m_zz3 * bxz3_part,
            yy_m_bzz3 * yz_pairs,
            xy_pairs * xx3_m_zz3,
        ];
        (
            Self {
                x: final_products[0] - final_products[1],
                y: final_products[2] + final_products[3],
                z: final_products[4] + final_products[5],
            },
            [xx, yy, zz, xy_product, yz_product, xz_product],
            final_products,
        )
    }

    /// Complete mixed homogeneous addition used by the row accumulator.  The
    /// five returned products are exactly the materialized AIR products.
    #[must_use]
    pub fn add_mixed_complete(&self, other: &D11AffinePointV1<F>) -> (Self, [D11<F>; 5]) {
        let u0 = self.x * other.x;
        let u1 = self.y * other.y;
        let u3 = (self.x + self.y) * (other.x + other.y);
        let u4 = other.x * self.z;
        let u5 = other.y * self.z;
        self.add_mixed_from_products(u0, u1, u3, u4, u5)
    }

    /// Production specialization for the protocol's sparse-seven packed X.
    /// The RCB formula is shared with [`Self::add_mixed_complete`]; only the
    /// two dense×X products select the 77-product sparse kernel.
    #[must_use]
    pub fn add_mixed_complete_sparse(
        &self,
        other_x: &D11Sparse7<F>,
        other_y: &D11<F>,
    ) -> (Self, [D11<F>; 5]) {
        let other_x_dense = other_x.to_d11();
        let u0 = self.x.mul_sparse_7(other_x);
        let u1 = self.y * *other_y;
        let u3 = (self.x + self.y) * (other_x_dense + *other_y);
        let u4 = self.z.mul_sparse_7(other_x);
        let u5 = self.z * *other_y;
        self.add_mixed_from_products(u0, u1, u3, u4, u5)
    }

    fn add_mixed_from_products(
        &self,
        u0: D11<F>,
        u1: D11<F>,
        u3: D11<F>,
        u4: D11<F>,
        u5: D11<F>,
    ) -> (Self, [D11<F>; 5]) {
        let sxy = u3 - u0 - u1;
        let sxz = self.x + u4;
        let syz = self.y + u5;

        let delta = (sxz - self.z.mul_by_z_plus_36()) * F::from_canonical_u32(3);
        let l0 = u1 + delta;
        let l3 = u1 - delta;
        let l2 = (u0 - self.z) * F::from_canonical_u32(3);
        let l1 = (sxz.mul_by_z_plus_36() - u0 - self.z * F::from_canonical_u32(3)) *
            F::from_canonical_u32(3);

        (
            Self { x: sxy * l0 - syz * l1, y: l2 * l1 + l3 * l0, z: syz * l3 + sxy * l2 },
            [u0, u1, u3, u4, u5],
        )
    }
}
