//! ML-DSA-65 (FIPS 204) signature verification, packaged as a Jolt guest.
//!
//! Verification is split between the host and the guest:
//!
//! - The **host** runs the publicly determined hashing — `Â = ExpandA(rho)`,
//!   `tr = SHAKE256(pk)`, `μ = SHAKE256(tr ‖ M)`, `c = SampleInBall(c̃)` —
//!   before invoking the prover, and ships the results into the proof as
//!   additional public inputs.
//! - The **guest** decodes those bytes, builds a `VerifyingKey` from the
//!   precomputed parts, and runs only the lattice arithmetic plus the final
//!   `c̃' = H(μ ‖ w1Encode(w₁'))` SHAKE256 (the only hash that genuinely
//!   depends on the lattice computation).
//!
//! Compared to the all-in-guest version this lifts ~192 of the ~200 verify
//! Keccak permutations out of the proof. The host MUST recompute the
//! precomputed values from `(pk, msg, sig)` itself before calling
//! `Jolt.verify(...)`; otherwise an attacker can supply mismatched values
//! and the proof will accept a forgery.

#![cfg_attr(feature = "guest", no_std)]
// The `#[jolt::provable]` macro expands to several wrapper functions that
// inherit our 7-arg signature; suppress the lint at the crate level rather
// than annotating each macro-generated function individually.
#![expect(clippy::too_many_arguments)]

extern crate alloc;

use core::hint::black_box;

use jolt::{end_cycle_tracking, spoil_proof, start_cycle_tracking, UnwrapOrSpoilProof};
use ml_dsa::{EncodedVerifyingKey, MlDsa65, Signature, VerifyingKey, VerifyingKeyParams};

pub mod precomputed;

use precomputed::{deserialize_a_hat, polynomial_from_signed_bytes};

// Working set for ML-DSA-65: matrix A_hat alone is 6 × 5 × 256 i32 ≈ 30 KiB,
// plus polynomial vectors for z, t1, c, etc. A_hat is now received as bytes
// (~30 KiB) and reconstructed in place; max_input_size grows accordingly.
#[jolt::provable(
    profile = "guest-profile",
    stack_size = 262_144,
    heap_size = 4_194_304,
    max_trace_length = 16_777_216,
    max_input_size = 65_536
)]
fn mldsa_verify(
    pk: &[u8],
    msg: &[u8],
    sig: &[u8],
    a_hat_bytes: &[u8],
    tr_bytes: &[u8],
    mu_bytes: &[u8],
    c_bytes: &[u8],
) {
    // `black_box` on inputs and intermediate values prevents the optimizer
    // from constant-folding across phase boundaries or eliding work between
    // cycle-tracking markers.
    let pk = black_box(pk);
    let _msg = black_box(msg);
    let sig_bytes = black_box(sig);
    let a_hat_bytes = black_box(a_hat_bytes);
    let tr_bytes = black_box(tr_bytes);
    let mu_bytes = black_box(mu_bytes);
    let c_bytes = black_box(c_bytes);

    // The host always sends fixed-size payloads; bail out (spoil) on any
    // length mismatch rather than indexing into a malformed slice.
    let tr: [u8; 64] = tr_bytes.try_into().unwrap_or_spoil_proof();
    let mu: [u8; 64] = mu_bytes.try_into().unwrap_or_spoil_proof();
    let c_bytes: &[u8; 256] = c_bytes.try_into().unwrap_or_spoil_proof();

    // Phase 1: decode pk = (rho, t1) and reconstruct VerifyingKey using the
    // precomputed A_hat and tr that the host computed via SHAKE128/SHAKE256.
    // No SHAKE work happens in this phase anymore — ExpandA and H(pk) both
    // ran on the host.
    start_cycle_tracking("phase1_decode_pk_with_precomputed");
    let vk_enc = <&EncodedVerifyingKey<MlDsa65>>::try_from(pk).unwrap_or_spoil_proof();
    let (rho, t1_enc) = MlDsa65::split_vk(vk_enc);
    let t1 = MlDsa65::decode_t1(t1_enc);
    let a_hat = deserialize_a_hat(a_hat_bytes).unwrap_or_spoil_proof();
    let vk = VerifyingKey::<MlDsa65>::new_with_precomputed(*rho, t1, a_hat, tr.into());
    let vk = black_box(vk);
    end_cycle_tracking("phase1_decode_pk_with_precomputed");

    // Phase 2: decode the signature (`c̃`, `z`, `h`); bit-unpack `z`; decode
    // the sparse hint vector; enforce `‖z‖∞ < γ1 - β`. Same as before — this
    // phase has no SHAKE work to skip.
    start_cycle_tracking("phase2_decode_signature");
    let sig = Signature::<MlDsa65>::try_from(sig_bytes).unwrap_or_spoil_proof();
    let sig = black_box(sig);
    end_cycle_tracking("phase2_decode_signature");

    // Phase 3: the lattice-only part of ML-DSA.Verify_internal. With μ and
    // c (= SampleInBall(c̃)) already supplied by the host, the only SHAKE
    // call left in-proof is the final `c̃' = H(μ ‖ w1Encode(w1'))`, which
    // genuinely depends on the lattice arithmetic.
    //   ẑ = NTT(z); ĉ = NTT(c)                     (host-supplied c)
    //   Aẑ = A_hat · ẑ        (K·L = 30 pointwise mul + accumulations)
    //   ĉt1 = ĉ · t1_2d_hat   (K = 6 pointwise muls)
    //   w'_approx = NTT^{-1}(Aẑ - ĉt1)
    //   w1' = UseHint(h, w'_approx)
    //   c̃' = SHAKE256(μ ‖ w1Encode(w1'))           (~8 SHAKE256 perms)
    //   accept iff c̃' == c_tilde
    start_cycle_tracking("phase3_verify_internal");
    let c = polynomial_from_signed_bytes(c_bytes);
    let accepted = vk.raw_verify_with_precomputed(&mu.into(), &c, &sig);
    let accepted = black_box(accepted);
    end_cycle_tracking("phase3_verify_internal");

    if !accepted {
        spoil_proof();
    }
}
