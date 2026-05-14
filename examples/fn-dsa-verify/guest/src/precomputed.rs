//! Serialization helpers for values the host precomputes and ships to the guest.
//!
//! The guest receives `c = hash_to_point(nonce, hashed_key, ctx, id, hv)`
//! (512 u16 coefficients < q=12289) as raw bytes, since the macro input
//! format is `&[u8]`. This module moves the typed `[u16; 512]` across
//! that boundary.

extern crate alloc;

use alloc::vec::Vec;

/// FN-DSA-512 polynomial degree.
pub const N_512: usize = 512;

/// Bytes per coefficient when serialized as little-endian u16.
pub const COEFF_BYTES: usize = 2;

/// Serialize an `n`-element `u16` polynomial as `n * 2` little-endian bytes.
#[must_use]
pub fn serialize_c(c: &[u16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(c.len() * COEFF_BYTES);
    for &v in c {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Deserialize `n * 2` little-endian bytes back into an `n`-element `u16`
/// polynomial. Returns `None` if the byte slice length is not a multiple
/// of 2.
#[must_use]
pub fn deserialize_c(bytes: &[u8], n: usize) -> Option<Vec<u16>> {
    if bytes.len() != n * COEFF_BYTES {
        return None;
    }
    let mut out = Vec::with_capacity(n);
    for chunk in bytes.chunks_exact(COEFF_BYTES) {
        out.push(u16::from_le_bytes([chunk[0], chunk[1]]));
    }
    Some(out)
}
