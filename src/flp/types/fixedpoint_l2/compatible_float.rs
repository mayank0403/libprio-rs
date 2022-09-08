// SPDX-License-Identifier: MPL-2.0

//! Implementations of encoding fixed point types as field elements and field elements as floats
//! for the [`FixedPointBoundedL2VecSum`](crate::flp::types::fixedpoint_l2::FixedPointBoundedL2VecSum) type.

use crate::field::{Field128, FieldElement};
use fixed::types::extra::{U15, U31, U63};
use fixed::{FixedI16, FixedI32, FixedI64};

/// Assign a `Float` type to this type and describe how to represent this type as an integer of the
/// given field, and how to represent a field element as the assigned `Float` type.
pub trait CompatibleFloat<F: FieldElement> {
    /// Represent a field element as `Float`, given the number of clients `c`.
    fn to_float(t: F, c: usize) -> f64;

    /// Represent a value of this type as an integer in the given field.
    fn to_field_integer(&self) -> <F as FieldElement>::Integer;
}

impl CompatibleFloat<Field128> for FixedI16<U15> {
    fn to_float(d: Field128, c: usize) -> f64 {
        to_float_bits(d, c, 15)
    }

    fn to_field_integer(&self) -> <Field128 as FieldElement>::Integer {
        //signed two's complement integer representation
        let i: i16 = self.to_bits();
        // reinterpret as unsigned
        let u = i as u16;
        // invert the left-most bit to de-two-complement
        u128::from(u ^ (1 << 15))
    }
}

impl CompatibleFloat<Field128> for FixedI32<U31> {
    fn to_float(d: Field128, c: usize) -> f64 {
        to_float_bits(d, c, 31)
    }

    fn to_field_integer(&self) -> <Field128 as FieldElement>::Integer {
        //signed two's complement integer representation
        let i: i32 = self.to_bits();
        // reinterpret as unsigned
        let u = i as u32;
        // invert the left-most bit to de-two-complement
        u128::from(u ^ (1 << 31))
    }
}

impl CompatibleFloat<Field128> for FixedI64<U63> {
    fn to_float(d: Field128, c: usize) -> f64 {
        to_float_bits(d, c, 63)
    }

    fn to_field_integer(&self) -> <Field128 as FieldElement>::Integer {
        //signed two's complement integer representation
        let i: i64 = self.to_bits();
        // reinterpret as unsigned
        let u = i as u64;
        // invert the left-most bit to de-two-complement
        u128::from(u ^ (1 << 63))
    }
}

/// Return an `f64` representation of the field element `d`, assuming it is the computation result
/// of a `c`-client fixed point vector summation with `n` fractional bits.
fn to_float_bits(d: Field128, c: usize, n: i32) -> f64 {
    // get integer representation of field element
    let i: u128 = <Field128 as FieldElement>::Integer::from(d);
    // interpret integer as float
    let f = i as f64;
    // to decode a single integer, we'd use the function
    // dec(y) = (y - 2^(n-1)) * 2^(1-n) = y * 2^(1-n) - 1
    // as f is the sum of c encoded vector entries where c is the number of
    // clients, we compute f * 2^(1-n) - c
    f * f64::powi(2.0, -n) - (c as f64)
}
