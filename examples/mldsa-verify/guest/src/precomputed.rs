//! Serialization helpers for values the host precomputes and ships to the guest.
//!
//! The guest receives `A_hat = ExpandA(rho)` (~30 KiB) and `c = SampleInBall(c_tilde)`
//! (256 coefficients in {-1, 0, +1}) as raw bytes, since the macro input format is
//! `&[u8]` / `[u8; N]`. These helpers move the typed `NttMatrix` / `Polynomial`
//! across that boundary in a self-describing way that both sides agree on.
//!
//! Every coefficient is field-reduced internally (`Elem<F>(F::Int)`), so the
//! serialization is just little-endian `u32`. For `c` we use a more compact
//! signed-byte encoding because coefficients are always in `{-1, 0, +1}`.

extern crate alloc;

use alloc::vec::Vec;
use hybrid_array::{Array, ArraySize};
use ml_dsa::{Elem, NttMatrix, NttPolynomial, NttVector, Polynomial};

// `NttMatrix<K, L>`, `NttVector<K>`, `NttPolynomial`, and `Polynomial` are
// `pub type` aliases for `module_lattice` types in `ml_dsa`. They expose
// `pub const fn new(inner)` constructors and a public `.0` field for direct
// access — both used below.

/// Number of coefficients per `Polynomial` / `NttPolynomial` (FIPS 204 fixes this at 256).
pub const COEFFS_PER_POLY: usize = 256;

/// Bytes per `Elem<F>` when serialized as little-endian `F::Int = u32`.
pub const ELEM_BYTES: usize = 4;

/// FIPS 204 modulus `q = 2^23 - 2^13 + 1`. Hardcoded here so the helper stays
/// `no_std` friendly without needing to expose a private `BaseField` constant.
pub const FIELD_Q: u32 = 8_380_417;

/// Bytes needed to serialize an `NttMatrix<K, L>`: `K * L * 256 * 4`.
#[must_use]
pub const fn a_hat_size(k: usize, l: usize) -> usize {
    k * l * COEFFS_PER_POLY * ELEM_BYTES
}

/// Serialize an `NttMatrix<K, L>` as `K * L * 256` little-endian `u32` coefficients.
#[must_use]
pub fn serialize_a_hat<K: ArraySize, L: ArraySize>(a_hat: &NttMatrix<K, L>) -> Vec<u8> {
    let mut out = Vec::with_capacity(a_hat_size(K::USIZE, L::USIZE));
    for row in a_hat.0.iter() {
        for poly in row.0.iter() {
            for elem in poly.0.iter() {
                out.extend_from_slice(&elem.0.to_le_bytes());
            }
        }
    }
    out
}

/// Inverse of [`serialize_a_hat`]. Reads `K * L * 256 * 4` bytes back into an `NttMatrix<K, L>`.
///
/// Returns `None` if `bytes.len()` doesn't match the expected size. The host
/// is trusted to supply byte-correct input; mismatched coefficients propagate
/// through the lattice check and cause the final `c̃' == c̃` comparison to
/// fail, which the caller surfaces as a spoiled proof.
#[must_use]
pub fn deserialize_a_hat<K: ArraySize, L: ArraySize>(bytes: &[u8]) -> Option<NttMatrix<K, L>> {
    if bytes.len() != a_hat_size(K::USIZE, L::USIZE) {
        return None;
    }
    let mut iter = bytes.chunks_exact(ELEM_BYTES);
    let matrix = NttMatrix::new(Array::from_fn(|_| {
        NttVector::new(Array::from_fn(|_| {
            NttPolynomial::new(Array::from_fn(|_| {
                let chunk = iter.next().expect("chunks_exact length checked");
                let raw = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                Elem::new(raw)
            }))
        }))
    }));
    Some(matrix)
}

/// Encode a `Polynomial` whose coefficients are all in `{-1, 0, +1}` as 256
/// signed bytes (`-1 → 0xFF`, `0 → 0x00`, `+1 → 0x01`). Out-of-range
/// coefficients are encoded as `0`, which will cause downstream verify to
/// reject — the host is responsible for only calling this on
/// `SampleInBall(c̃)` outputs.
#[must_use]
pub fn polynomial_to_signed_bytes(c: &Polynomial) -> [u8; COEFFS_PER_POLY] {
    let mut out = [0u8; COEFFS_PER_POLY];
    for (i, elem) in c.0.iter().enumerate() {
        out[i] = match elem.0 {
            0 => 0x00,
            1 => 0x01,
            v if v == FIELD_Q - 1 => 0xFF,
            _ => 0x00,
        };
    }
    out
}

/// Inverse of [`polynomial_to_signed_bytes`]. Maps `0x00 → 0`, `0x01 → +1`,
/// `0xFF → -1` (mod q); any other byte value is treated as `0`.
#[must_use]
pub fn polynomial_from_signed_bytes(bytes: &[u8; COEFFS_PER_POLY]) -> Polynomial {
    let coeffs = Array::from_fn(|i| match bytes[i] {
        0x00 => Elem::new(0),
        0x01 => Elem::new(1),
        0xFF => Elem::new(FIELD_Q - 1),
        _ => Elem::new(0),
    });
    Polynomial::new(coeffs)
}
