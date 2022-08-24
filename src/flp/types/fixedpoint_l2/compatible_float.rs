// SPDX-License-Identifier: MPL-2.0

//! Implementations of encoding fixed point types as field elements and field elements as floats
//! for the [`FixedPointL2BoundedVecSum`](crate::flp::types::fixedpoint_l2::FixedPointBoundedL2VecSum)
//! type.

use crate::field::{Field64, FieldElement};
use fixed::FixedI16;
use std::fmt::Debug;

/// Assign a `Float` type to this type and describe how to represent this type as an integer of the
/// given field, and how to represent a field element as the assigned `Float` type.
pub trait CompatibleFloat<F: FieldElement> {
    /// The float type F can be converted into.
    type Float: Debug + Clone;
    /// Represent a field element as `Float`, given the number of clients `c`.
    fn to_float(t: F, c: usize) -> Self::Float;
    /// Represent a value of this type as an integer in the given field.
    fn to_field_integer(t: Self) -> <F as FieldElement>::Integer;
}

impl<U> CompatibleFloat<Field64> for FixedI16<U> {
    type Float = f64;
    fn to_float(d: Field64, c: usize) -> f64 {
        // get integer representation of field element
        let i: u64 = <Field64 as FieldElement>::Integer::from(d);
        // interpret integer as float
        let f = i as f64;
        // to decode a single integer, we'd use the function
        // dec(y) = (y - 2^(n-1)) * 2^(1-n) = y * 2^(1-n) - 1
        // as f is the sum of c encoded vector entries where c is the number of
        // clients, we compute f * 2^(1-n) - c
        f * f64::powi(2.0, -15) - (c as f64)
    }
    fn to_field_integer(fp: FixedI16<U>) -> <Field64 as FieldElement>::Integer {
        //signed two's complement integer representation
        let i: i16 = fp.to_bits();
        // reinterpret as unsigned
        let u = i as u16;
        // invert the left-most bit to de-two-complement
        u64::from(u ^ (1 << 15))
    }
}
