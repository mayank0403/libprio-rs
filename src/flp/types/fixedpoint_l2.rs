// SPDX-License-Identifier: MPL-2.0

//! A [`Type`](crate::flp::Type) for summing vectors of fixed point numbers whose L2 norm is
//! bounded by 1.

pub mod compatible_float;

use crate::field::{FieldElement, FieldElementExt};
use crate::flp::gadgets::PolyEval;
use crate::flp::types::call_gadget_on_vec_entries;
use crate::flp::types::fixedpoint_l2::compatible_float::CompatibleFloat;
use crate::flp::{FlpError, Gadget, Type};
use crate::polynomial::poly_range_check;
use fixed::traits::Fixed;

use std::{convert::TryInto, fmt::Debug, marker::PhantomData};

/// The fixed point vector sum data type. Each measurement is a vector of fixed point numbers of
/// type `T`, and the aggregate result is the float vector of the sum of the measurements.
///
/// The validity circuit verifies that the L2 norm of each measurement is bounded by 1.
///
/// The [*fixed* crate] is used for fixed point numbers, in particular, exactly the following types
/// are supported: `FixedI8<U7>`, `FixedI16<U15>`, `FixedI32<U31>`, `FixedI64<U63>` and
/// `FixedI128<U127>`.
///
/// Depending on the size of the vector that needs to be transmitted, a corresponding field type has
/// to be chosen for `F`. For a `n`-bit fixed point type and a `d`-dimensional vector, the field
/// modulus needs to be larger than `d * 2^(2n-2)` so there are no overflows during norm validity
/// computation.
///
/// [*fixed* crate]: https://crates.io/crates/fixed
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FixedPointL2BoundedVecSum<T: Fixed, F: FieldElement> {
    bits_per_entry: usize,
    entries: usize,
    bits_for_norm: usize,
    range_01_checker: Vec<F>,
    square_computer: Vec<F>,
    phantom: PhantomData<T>,
    // range/position constants
    range_norm_begin: usize,
    range_norm_end: usize,
}

/* In the following a high level overview over the inner workings of this type
 * is given and implementation details are discussed. It is not necessary for
 * using the type, but it should be very helpful when trying to understand the
 * implementation.
 *
 *
 * --- Overview ---
 * Clients submit a vector of numbers whose values semantically lie in [-1,1),
 * together with a norm in the range [0,1). The validation circuit checks that
 * the norm of the vector is equal to the submitted norm, while the encoding
 * guarantees that the submitted norm lies in the correct range.
 *
 *
 * --- Different number encodings ---
 * Let n denote the number of bits of the chosen fixed-point type.
 * Numbers occur in 5 different representations:
 *  (1) Clients have a vector whose entries are fixed point numbers. Only those
 *      fixed point types are supported where the numbers lie in the range
 *      [-1,1).
 *  (2) Because norm computation happens in the validation circuit, it is done
 *      on entries encoded as field elements. That is, the same vector entries
 *      are now represented by integers in the range [0,2^n), where -1 is
 *      represented by 0 and +1 by 2^n.
 *  (3) Because the field is not necessarily exactly of size 2^n, but might be
 *      larger, it is not enough to encode a vector entry as in (2) and submit
 *      it to the aggregator. Instead, in order to make sure that all submitted
 *      values are in the correct range, they are bit-encoded. (This is the same
 *      as what happens in the `Sum` type.)
 *      This means that instead of sending a field element in the range [0,2^n),
 *      we send n field elements representing the bit encoding. The validation
 *      circuit can verify that all submitted "bits" are indeed either 0 or 1.
 *  (4) The computed and submitted norms are treated very similar to the vector
 *      entries, but they have a different number of bits, namely 2n-2.
 *  (5) As the aggregation result is a pointwise sum of the client vectors,
 *      the numbers no longer (semantically) lie in the range [-1,1), and cannot
 *      be represented by the same fixed point type as the input. Instead the
 *      decoding happens directly into a vector of floats.
 *
 * --- Fixed point encoding ---
 * Submissions consist of encoded fixed-point numbers in [-1,1) represented as
 * field elements in [0, 2^n), where n is the number of bits the fixed-point
 * representation has. Encoding and decoding is handled by the associated
 * functions of the `CompatibleFloat` trait.
 *
 * Semantically, the following function describes how a fixed-point value `x` is
 * converted to a field integer:
 *   enc : [-1,1) -> [0,2^n)
 *   enc(x) = 2^(n-1) * x + 2^(n-1)
 * The inverse is:
 *   dec : [0,2^n) -> [-1,1)
 *   dec(y) = (y - 2^(n-1)) * 2^(1-n)
 * Note that these functions only make sense when interpreting all occuring
 * numbers as real numbers. Since our signed fixed-point numbers are encoded as
 * two's complement integers, the computation that happens in
 * `CompatibleFloat::to_field_integer` is actually simpler.
 *
 *
 * --- Norm computation ---
 * The L2 norm of a vector xs of numbers in [-1,1) is given by:
 *   norm(xs) = sqrt(sum_{x in xs} x^2)
 * Instead of computing the norm, we make two simplifications:
 *  (1) We ignore the square root, which means that we are actually computing
 *      the square of the norm.
 *  (2) Since we work with integers, fractional values should not occur in the
 *      computation. This is done by working with a factor of 2^(2n-2).
 * This means that what is actually computed in this type is the following:
 *   our_norm(xs) = 2^(2n-2) * norm(xs)^2
 * Given a vector ys of numbers in the field integer encoding (in [0,2^n]),
 * this gives the following equation:
 *   our_norm_on_encoded(ys) = our_norm([dec(y) for y in ys])
 *                           = sum_{y in ys} y^2 - (2^n)*y + 2^(2n-2)
 * The constant and linear terms in the sum appear because the decoding function
 * is not linear but only affine.
 *
 * Let `d` denote the number of the vector entries. The maximal value the result
 * of `our_norm_on_encoded()` can take occurs in the case where all entries are
 * `2^n-1`, in which case `d * 2^(2n-2)` is an upper bound to the result. The
 * field type must be such that this number fits inside.
 *
 * For validating that the norm of the submitted vector lies in the correct
 * range, consider the following:
 *  - The result of `norm(xs)` should be in [0,1).
 *  - Thus, the result of `our_norm(xs)` should be in [0,2^(2n-2)).
 *  - The result of `our_norm_on_encoded(ys)` should be in the same range.
 * This means that the valid norms are exactly those representable with `2n-2`
 * bits.
 *
 *
 * --- Differences in the computation because of distribution ---
 * Computation of the norm in the validation circuit happens distributed, which
 * means that every aggregator computes the circuit on an additive share of the
 * client's actual vector entries and norm. This has the slight problem that
 * the constant part of the computation done in `our_norm_on_encoded()` occurs
 * `num_shares` times in the final aggregated result. The implementation
 * of the norm computation, in `compute_norm_of_entries()`, has an additional
 * parameter `constant_part_multiplier` which is set to 1/num_shares when the
 * norm is computed in the validation circuit.
 * Something similar happens in the decoding of the aggregated result (in
 * `decode_result()`), where instead of the `dec()` function from above, the
 * following function is used:
 *   dec'(x) = d * 2^(1-n) - c
 * Here, `c` is the number of clients.
 *
 *
 * --- Naming in the implementation ---
 * The following names are used:
 *  - `self.bits_per_entry` is n
 *  - `self.entries`        is d
 *  - `self.bits_for_norm`  is 2n-2
 *
 *
 * --- Submission layout ---
 * The client submissions contain a share of their vector and the norm
 * they claim it has.
 * The submission is a vector of field elements laid out as follows:
 * |---- bits_per_entry * entries ----|---- bits_for_norm ----|
 *  ^                                  ^
 *  \- the input vector entries        |
 *                                     \- the encoded norm
 *
 *
 * --- Validity ---
 * In addition to checking that every submission entry is 0 or 1, the validation
 * circuit of this type computes the norm and compares to what the client
 * claimed.
 *
 *
 * --- Value 1 ---
 * We actually do not allow the submitted norm or vector entries to be
 * exactly 1, but rather require them to be strictly less. Supporting 1 would
 * entail a more fiddly encoding and is not necessary for our usecase.
 */

impl<T: Fixed, F: FieldElement> FixedPointL2BoundedVecSum<T, F> {
    /// Return a new [`FixedPointL2BoundedVecSum`] type parameter. Each value of this type is a
    /// fixed point vector with `entries` entries.
    pub fn new(entries: usize) -> Result<Self, FlpError> {
        // (I) Check that the fixed type `F` is compatible.

        // We only support fixed types that encode values in [-1,1].
        // These have a single integer bit.
        if <T as Fixed>::INT_NBITS != 1 {
            return Err(FlpError::Encode(format!(
                "Expected fixed point type with one integer bit, but got {}.",
                <T as Fixed>::INT_NBITS,
            )));
        }

        // Compute number of bits of an entry, and check that an entry fits
        // into the field.
        let bits_per_entry: usize = (<T as Fixed>::INT_NBITS + <T as Fixed>::FRAC_NBITS)
            .try_into()
            .map_err(|_| FlpError::Encode("Could not convert u32 into usize.".to_string()))?;
        if !F::valid_integer_bitlength(bits_per_entry) {
            return Err(FlpError::Encode(format!(
                "fixed point type bit length ({}) too large for field modulus",
                bits_per_entry,
            )));
        }

        // (II) Check that the field is large enough for the norm.

        // Valid norms encoded as field integers lie in [0,2^(2*bits - 2)).
        let bits_for_norm = 2 * bits_per_entry - 2;
        if !F::valid_integer_bitlength(bits_for_norm) {
            return Err(FlpError::Encode(format!(
                "maximal norm bit length ({}) too large for field modulus",
                bits_for_norm,
            )));
        }

        // In order to compare the actual norm of the vector with the claimed
        // norm, the field needs to be able to represent all numbers that can
        // occur during the computation of the norm of any submitted vector,
        // even if its norm is not bounded by 1. Because of our encoding, an
        // upper bound to that value is `entries * 2^(2*bits - 2)` (see docs of
        // compute_norm_of_entries for details). It has to fit into the field.
        let usize_max_norm_value: usize = match entries.checked_mul(1 << bits_for_norm) {
            Some(val) => val,
            None => {
                return Err(FlpError::Encode(format!(
                    "number of entries ({}) not compatible with field size",
                    entries,
                )))
            }
        };
        F::valid_integer_try_from(usize_max_norm_value)?;

        Ok(Self {
            bits_per_entry,
            entries,
            bits_for_norm,
            range_01_checker: poly_range_check(0, 2),
            square_computer: vec![F::zero(), F::zero(), F::one()],
            phantom: PhantomData,

            // range constants
            range_norm_begin: entries * bits_per_entry,
            range_norm_end: entries * bits_per_entry + bits_for_norm,
        })
    }
}

impl<T: Fixed, F: FieldElement> Type for FixedPointL2BoundedVecSum<T, F>
where
    T: CompatibleFloat<F>,
{
    type Measurement = Vec<T>;
    type AggregateResult = Vec<<T as CompatibleFloat<F>>::Float>;
    type Field = F;

    fn encode_measurement(&self, fp_entries: &Vec<T>) -> Result<Vec<F>, FlpError> {
        // Convert the fixed-point encoded input values to field integers. We do
        // this once here because we need them for encoding but also for
        // computing the norm.
        let integer_entries: Vec<_> = fp_entries
            .iter()
            .map(|x| <T as CompatibleFloat<F>>::to_field_integer(*x))
            .collect();

        // (I) Vector entries.
        // Encode the integer entries bitwise, and write them into the `encoded`
        // vector.
        let mut encoded: Vec<F> =
            vec![F::zero(); self.bits_per_entry * self.entries + self.bits_for_norm];
        for (l, entry) in integer_entries.iter().enumerate() {
            F::encode_into_bitvector_representation_slice(
                entry,
                &mut encoded[l * self.bits_per_entry..(l + 1) * self.bits_per_entry],
            )?;
        }

        // (II) Vector norm.
        // Compute the norm of the input vector.
        let field_entries = integer_entries.iter().map(|&x| F::from(x));
        let norm =
            compute_norm_of_entries(field_entries, self.bits_per_entry, F::one(), &mut |x| {
                Ok(x * x)
            })?;
        let norm_int = <F as FieldElement>::Integer::from(norm);

        // Write the norm into the `entries` vector.
        F::encode_into_bitvector_representation_slice(
            &norm_int,
            &mut encoded[self.range_norm_begin..self.range_norm_end],
        )?;

        Ok(encoded)
    }

    fn decode_result(
        &self,
        data: &[F],
        num_measurements: usize,
    ) -> Result<Vec<<T as CompatibleFloat<F>>::Float>, FlpError> {
        if data.len() != self.entries {
            return Err(FlpError::Decode("unexpected input length".into()));
        }
        let mut res = vec![];
        for d in data {
            let decoded = <T as CompatibleFloat<F>>::to_float(*d, num_measurements);
            res.push(decoded);
        }
        Ok(res)
    }

    fn gadget(&self) -> Vec<Box<dyn Gadget<F>>> {
        // This gadgets checks that a field element is zero or one.
        // It is called for all the "bits" of the encoded entries
        // and of the encoded norm.
        let gadget0 = PolyEval::new(
            self.range_01_checker.clone(),
            self.bits_per_entry * self.entries + self.bits_for_norm,
        );

        // This gadget computes the square of a field element.
        // It is called on each entry during norm computation.
        let gadget1 = PolyEval::new(self.square_computer.clone(), self.entries);

        let res: Vec<Box<dyn Gadget<F>>> = vec![Box::new(gadget0), Box::new(gadget1)];
        res
    }

    fn valid(
        &self,
        g: &mut Vec<Box<dyn Gadget<F>>>,
        input: &[F],
        joint_rand: &[F],
        num_shares: usize,
    ) -> Result<F, FlpError> {
        self.valid_call_check(input, joint_rand)?;

        // Ensure that all submitted field elements are either 0 or 1.
        // This is done for:
        //  - all vector entries (each of them encoded in `self.bits_per_entry`
        //    field elements)
        //  - the submitted norm (encoded in `self.bits_for_norm` field
        //    elements)
        //
        // Since all input vector entry (field-)bits, as well as the norm bits,
        // are contiguous, we do the check directly for all bits from 0 to
        // entries*bits_per_entry + bits_for_norm.
        //
        // Check that each element is a 0 or 1:
        let range_check =
            call_gadget_on_vec_entries(&mut g[0], &input[0..self.range_norm_end], joint_rand[0])?;

        // Compute the norm of the entries and ensure that it is the same as the
        // submitted norm. There are exactly enough bits such that a submitted
        // norm is always a valid norm (semantically in the range [0,1]). By
        // comparing submitted with actual, we make sure the actual norm is
        // valid.

        // Computing the norm is done using `compute_norm_of_entries()`. This
        // needs some setup, in particular there is:
        //  - `decoded_entries` is an iterator over `self.entries` many field
        //    elements representing the vector entries
        //  - `constant_part_multiplier` is required because this validation
        //    function is executed by each aggregator and the result is summed.
        //    In the computation there is a constant part which would be added
        //    `num_of_clients` times, even though we only want it to be added
        //    once. To mitigate, we pass in the `constant_part_multiplier`,
        //    which is the inverse of the numbers of clients.
        //  - `squaring_fun` is a function which calls the squaring gadget (and
        //    mutates it).
        let decoded_entries: Result<Vec<_>, _> = input[0..self.entries * self.bits_per_entry]
            .chunks(self.bits_per_entry)
            .map(F::decode_from_bitvector_representation)
            .collect();

        let num_of_clients = F::valid_integer_try_from(num_shares)?;
        let constant_part_multiplier = F::one() / F::from(num_of_clients);

        let squaring_fun = &mut |x| g[1].call(std::slice::from_ref(&x));

        let computed_norm = compute_norm_of_entries(
            decoded_entries?,
            self.bits_per_entry,
            constant_part_multiplier,
            squaring_fun,
        )?;

        // The submitted norm is also decoded from its bit-encoding, and
        // compared with the computed norm.
        let submitted_norm_enc = &input[self.range_norm_begin..self.range_norm_end];
        let submitted_norm = F::decode_from_bitvector_representation(submitted_norm_enc)?;

        let norm_check = computed_norm - submitted_norm;

        // Finally, we require both checks to be successfull by computing a
        // random linear combination of them.
        let out = joint_rand[1] * range_check + (joint_rand[1] * joint_rand[1]) * norm_check;
        Ok(out)
    }

    fn truncate(&self, input: Vec<F>) -> Result<Vec<Self::Field>, FlpError> {
        self.truncate_call_check(&input)?;

        let mut decoded_vector = vec![];

        for i_entry in 0..self.entries {
            let start = i_entry * self.bits_per_entry;
            let end = (i_entry + 1) * self.bits_per_entry;

            let decoded = F::decode_from_bitvector_representation(&input[start..end])?;
            decoded_vector.push(decoded);
        }
        Ok(decoded_vector)
    }

    fn input_len(&self) -> usize {
        self.bits_per_entry * self.entries + self.bits_for_norm
    }

    fn proof_len(&self) -> usize {
        // computed via
        // `gadget.arity() + gadget.degree()
        //   * ((1 + gadget.calls()).next_power_of_two() - 1) + 1;`
        let proof_gadget_0 = 2
            * ((1 + (self.bits_per_entry * self.entries + self.bits_for_norm)).next_power_of_two()
                - 1)
            + 2;
        let proof_gadget_1 = 2 * ((1 + self.entries).next_power_of_two() - 1) + 2;
        proof_gadget_0 + proof_gadget_1
    }

    fn verifier_len(&self) -> usize {
        5
    }

    fn output_len(&self) -> usize {
        self.entries
    }

    fn joint_rand_len(&self) -> usize {
        2
    }

    fn prove_rand_len(&self) -> usize {
        2
    }

    fn query_rand_len(&self) -> usize {
        2
    }
}

/// Compute the square of the L2 norm of a vector of fixed-point numbers encoded as field elements.
///
/// * `entries` - Iterator over the vector entries.
/// * `bits_per_entry` - Number of bits one entry has.
/// * `constant_part_multiplier` - A share of 1.
/// * `sq` - The function used to compute the square of an entry.
fn compute_norm_of_entries<F, Fs, SquareFun>(
    entries: Fs,
    bits_per_entry: usize,
    constant_part_multiplier: F,
    sq: &mut SquareFun,
) -> Result<F, FlpError>
where
    F: FieldElement,
    Fs: IntoIterator<Item = F>,
    SquareFun: FnMut(F) -> Result<F, FlpError>,
{
    // Check out the Norm computation bit in the explanatory comment block
    // to understand what this function does.

    // Initialize `norm_accumulator`.
    let mut norm_accumulator = F::zero();

    // constants
    let constant_part = F::valid_integer_try_from(1 << (2 * bits_per_entry - 2))?; // = 2^(2n-2)
    let linear_part = F::valid_integer_try_from(1 << (bits_per_entry))?; // = 2^n

    // Add term for a given `entry` to `norm_accumulator`.
    // `constant_part` is distributed among clients for verification, so we
    // multiply with a share of 1.
    for entry in entries.into_iter() {
        let summand = sq(entry)? + F::from(constant_part) * constant_part_multiplier
            - F::from(linear_part) * (entry);
        norm_accumulator += summand;
    }
    Ok(norm_accumulator)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::{random_vector, Field64 as TestField};
    use crate::flp::types::test_utils::{flp_validity_test, ValidityTestCase};
    use fixed::types::extra::{U127, U14, U15};
    use fixed::{FixedI128, FixedI16};
    use fixed_macro::fixed;

    #[test]
    fn test_bounded_fpvec_sum() {
        let vsum: FixedPointL2BoundedVecSum<FixedI16<U15>, TestField> =
            FixedPointL2BoundedVecSum::new(3).unwrap();
        let one = TestField::one();

        let fp_4_inv = fixed!(0.25: I1F15);
        let fp_8_inv = fixed!(0.125: I1F15);
        let fp_16_inv = fixed!(0.0625: I1F15);

        let fp_vec1 = vec![fp_4_inv, fp_8_inv, fp_16_inv];

        // Round trip
        assert_eq!(
            vsum.decode_result(
                &vsum
                    .truncate(vsum.encode_measurement(&fp_vec1).unwrap())
                    .unwrap(),
                1
            )
            .unwrap(),
            vec!(0.25, 0.125, 0.0625)
        );

        // encoded norm does not match computed norm
        let mut input: Vec<TestField> = vsum.encode_measurement(&fp_vec1).unwrap();
        input[0] = one; // it was zero
        flp_validity_test(
            &vsum,
            &input,
            &ValidityTestCase::<TestField> {
                expect_valid: false,
                expected_output: Some(vec![
                    TestField::from(40961),
                    TestField::from(36864),
                    TestField::from(34816),
                ]),
            },
        )
        .unwrap();

        // encoding contains entries that are not zero or one
        let mut input2: Vec<TestField> = vsum.encode_measurement(&fp_vec1).unwrap();
        input2[0] = one + one;
        flp_validity_test(
            &vsum,
            &input2,
            &ValidityTestCase::<TestField> {
                expect_valid: false,
                expected_output: Some(vec![
                    TestField::from(40962),
                    TestField::from(36864),
                    TestField::from(34816),
                ]),
            },
        )
        .unwrap();

        // norm is too big
        flp_validity_test(
            &vsum,
            &vec![one; 78],
            &ValidityTestCase::<TestField> {
                expect_valid: false,
                expected_output: Some(vec![
                    TestField::from(65535),
                    TestField::from(65535),
                    TestField::from(65535),
                ]),
            },
        )
        .unwrap();

        // invalid submission length
        let joint_rand = random_vector(vsum.joint_rand_len()).unwrap();
        vsum.valid(&mut vsum.gadget(), &vec![one; 77], &joint_rand, 1)
            .unwrap_err();

        // test that the zero vector has correct norm, where zero is encoded as:
        // enc(0) = 2^(n-1) * 0 + 2^(n-1)
        //        = 32768
        {
            let entries = vec![
                TestField::from(32768),
                TestField::from(32768),
                TestField::from(32768),
            ];
            let norm =
                compute_norm_of_entries(entries, vsum.bits_per_entry, TestField::one(), &mut |x| {
                    Ok(x * x)
                })
                .unwrap();
            let expected_norm = TestField::from(0);
            assert_eq!(norm, expected_norm);
        }

        // ensure that no overflow occurs with largest possible norm
        {
            // the largest possible entries (2^n-1)
            let entries = vec![
                TestField::from(65535),
                TestField::from(65535),
                TestField::from(65535),
            ];
            let norm =
                compute_norm_of_entries(entries, vsum.bits_per_entry, TestField::one(), &mut |x| {
                    Ok(x * x)
                })
                .unwrap();
            let expected_norm = TestField::from(3221028867);
            assert_eq!(norm, expected_norm);

            // the smallest possible entries (0)
            let entries = vec![TestField::from(0), TestField::from(0), TestField::from(0)];
            let norm =
                compute_norm_of_entries(entries, vsum.bits_per_entry, TestField::one(), &mut |x| {
                    Ok(x * x)
                })
                .unwrap();
            let expected_norm = TestField::from(3221225472);
            assert_eq!(norm, expected_norm);
        }

        // invalid initialization
        // fixed point too large
        <FixedPointL2BoundedVecSum<FixedI128<U127>, TestField>>::new(3).unwrap_err();
        // vector too large
        <FixedPointL2BoundedVecSum<FixedI16<U15>, TestField>>::new(30000000000).unwrap_err();
        // fixed point type has more than one int bit
        <FixedPointL2BoundedVecSum<FixedI16<U14>, TestField>>::new(3).unwrap_err();
    }
}
