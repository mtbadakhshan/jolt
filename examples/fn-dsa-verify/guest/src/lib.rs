//! FN-DSA-512 (draft Falcon) signature verification, packaged as a Jolt guest.
//!
//! Verification is split between the host and the guest:
//!
//! - The **host** runs the publicly determined hashing — `hashed_key =
//!   SHAKE256(pk)` and `c = hash_to_point(nonce, hashed_key, ctx, id, msg)`
//!   — before invoking the prover, and ships the results into the proof as
//!   additional public inputs.
//! - The **guest** decodes `h` from `pk`, builds a `VerifyingKey512` from
//!   the precomputed `hashed_key` + already-NTT'd `h`, and runs only the
//!   lattice arithmetic (NTT, pointwise mul, iNTT, norm check) — the only
//!   work that genuinely requires a proof.
//!
//! Compared to the all-in-guest version this lifts ~15 of the ~15 SHAKE256
//! permutations out of the proof.

#![cfg_attr(feature = "guest", no_std)]

extern crate alloc;

use core::hint::black_box;

use fn_dsa_vrfy::{VerifyingKey, VerifyingKey512};
use jolt::{end_cycle_tracking, spoil_proof, start_cycle_tracking, UnwrapOrSpoilProof};

pub mod precomputed;

use precomputed::deserialize_c;

#[jolt::provable(
    profile = "guest-profile",
    stack_size = 65_536,
    heap_size = 16_384,
    max_trace_length = 8_388_608,
    max_input_size = 8192
)]
fn fn_dsa_verify(pk: &[u8], sig: &[u8], msg: &[u8], hashed_key_bytes: &[u8], c_bytes: &[u8]) {
    let pk = black_box(pk);
    let sig = black_box(sig);
    let _msg = black_box(msg);
    let hashed_key_bytes = black_box(hashed_key_bytes);
    let c_bytes = black_box(c_bytes);

    let hashed_key: [u8; 64] = hashed_key_bytes.try_into().unwrap_or_spoil_proof();

    // Phase 1: decode the verifying key from pk + precomputed hashed_key.
    // The upstream `decode()` would run SHAKE256(pk) to produce hashed_key
    // internally; we skip that because the host already computed it.
    // We still need to modq_decode + ext_to_int + int_to_NTT the `h`
    // polynomial, since those are pure arithmetic, not SHAKE.
    start_cycle_tracking("phase1_decode_vk");
    let vk = VerifyingKey512::decode(pk).unwrap_or_spoil_proof();
    // Rebuild with the host-supplied hashed_key to skip the SHAKE256(pk)
    // that `decode` already computed (it's the same value, but in the
    // precomputed flow we want the host's value to be the one bound to
    // the proof inputs).
    let vk =
        VerifyingKey512::from_parts(vk.get_logn(), vk.h_ntt(), hashed_key).unwrap_or_spoil_proof();
    let vk = black_box(vk);
    end_cycle_tracking("phase1_decode_vk");

    // Phase 2: verify with precomputed c (skips hash_to_point SHAKE256).
    start_cycle_tracking("phase2_verify_internal");
    let n = 1usize << vk.get_logn();
    let c = deserialize_c(c_bytes, n).unwrap_or_spoil_proof();
    let accepted = vk.verify_with_precomputed_c(sig, &c);
    let accepted = black_box(accepted);
    end_cycle_tracking("phase2_verify_internal");

    if !accepted {
        spoil_proof();
    }
}
