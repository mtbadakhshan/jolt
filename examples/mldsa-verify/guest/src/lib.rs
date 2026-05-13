//! ML-DSA-65 (FIPS 204) signature verification, packaged as a Jolt guest.
//!
//! The verifier is the upstream `ml-dsa` crate (RustCrypto). All Keccak
//! permutations bottom out in `keccak::f1600`, which we redirect to the Jolt
//! Keccak inline via a workspace-level `[patch.crates-io]` entry pointing at
//! `crates/keccak-jolt`. The guest code itself stays trivial.

#![cfg_attr(feature = "guest", no_std)]

extern crate alloc;

use core::hint::black_box;

use jolt::{end_cycle_tracking, start_cycle_tracking, UnwrapOrSpoilProof};
use ml_dsa::{KeyInit, MlDsa65, Signature, VerifyingKey};
use signature::Verifier;

// Working set for ML-DSA-65: matrix A_hat alone is 6 × 5 × 256 i32 ≈ 30 KiB,
// plus polynomial vectors for z, t1, c, etc. The actual trace lands around
// 5.5 M cycles; the headroom here is so we can swap in larger parameter sets
// without re-tuning.
#[jolt::provable(
    profile = "guest-profile",
    stack_size = 262_144,
    heap_size = 4_194_304,
    max_trace_length = 16_777_216,
    max_input_size = 8192
)]
fn mldsa_verify(pk: &[u8], msg: &[u8], sig: &[u8]) {
    // `black_box` on inputs and intermediate values prevents the optimizer
    // from constant-folding across phase boundaries or eliding work between
    // cycle-tracking markers.
    let pk = black_box(pk);
    let msg = black_box(msg);
    let sig_bytes = black_box(sig);

    // Phase 1: decode pk = (rho, t1); bit-unpack t1; expand A_hat via SHAKE128
    // rejection sampling (K·L = 30 polys); precompute tr = H(pk) via SHAKE256
    // and t1_2d_hat = NTT(t1 · 2^13). The bulk of the SHAKE128 work in the
    // whole verifier happens here (ExpandA).
    start_cycle_tracking("phase1_decode_pk_and_expand_a");
    let vk = decode_vk(pk).unwrap_or_spoil_proof();
    let vk = black_box(vk);
    end_cycle_tracking("phase1_decode_pk_and_expand_a");

    // Phase 2: decode sig = (c_tilde, z, h); bit-unpack z (5 polys × 20 bits/
    // coeff); decode the sparse hint vector h; enforce the norm check
    // ‖z‖∞ < γ1 - β. Returns Err on any malformed bytes or norm failure.
    start_cycle_tracking("phase2_decode_signature");
    let sig = Signature::<MlDsa65>::try_from(sig_bytes).unwrap_or_spoil_proof();
    let sig = black_box(sig);
    end_cycle_tracking("phase2_decode_signature");

    // Phase 3: the actual ML-DSA.Verify_internal:
    //   μ = SHAKE256(tr ‖ msg)
    //   c = SampleInBall(c_tilde)               (SHAKE256-driven)
    //   ẑ = NTT(z); ĉ = NTT(c)
    //   Aẑ = A_hat · ẑ        (K·L = 30 pointwise mul + accumulations)
    //   ĉt1 = ĉ · t1_2d_hat   (K = 6 pointwise muls)
    //   w'_approx = NTT^{-1}(Aẑ - ĉt1)
    //   w1' = UseHint(h, w'_approx)
    //   c̃' = SHAKE256(μ ‖ w1Encode(w1'))
    //   accept iff c̃' == c_tilde
    start_cycle_tracking("phase3_verify_internal");
    vk.verify(msg, &sig).unwrap_or_spoil_proof();
    end_cycle_tracking("phase3_verify_internal");
}

fn decode_vk(bytes: &[u8]) -> Result<VerifyingKey<MlDsa65>, signature::Error> {
    let arr = <&ml_dsa::EncodedVerifyingKey<MlDsa65>>::try_from(bytes)
        .map_err(|_| signature::Error::new())?;
    Ok(VerifyingKey::<MlDsa65>::new(arr))
}
