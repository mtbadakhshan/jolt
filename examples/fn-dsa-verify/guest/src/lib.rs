//! FN-DSA-512 (draft Falcon) signature verification, packaged as a Jolt guest.
//!
//! The verifier is the upstream `fn-dsa-vrfy` crate (Thomas Pornin). FN-DSA
//! verification is integer-only (no floating-point — those code paths live
//! in `fn-dsa-sign` / `fn-dsa-kgen` and are cfg-gated to x86/aarch64), which
//! is exactly what RV64IMAC can execute.
//!
//! Unlike the `mldsa-verify` example, the Keccak-f[1600] permutation here
//! is NOT redirected through Jolt's Keccak inline: `fn-dsa-comm`'s
//! `KeccakState::process()` is hand-rolled inside the crate and there is no
//! external crate boundary to patch. The trace therefore runs the SHAKE256
//! permutation as a few hundred kilobytes of plain RV64IMAC instructions.
//! See the example README for the planned acceleration path (a vendored
//! `fn-dsa-comm` mirror following the `crates/keccak-jolt` pattern).

#![cfg_attr(feature = "guest", no_std)]

use core::hint::black_box;

use fn_dsa_vrfy::{VerifyingKey, VerifyingKey512, DOMAIN_NONE, HASH_ID_RAW};
use jolt::{end_cycle_tracking, spoil_proof, start_cycle_tracking, UnwrapOrSpoilProof};

// Working set for FN-DSA-512:
//   VerifyingKey512  : h = [u16; 512]   + hashed_key = [u8; 64]    ~ 1.1 KiB
//   verify_inner     : tmp_i16 = [i16; 512] + tmp_u16 = [u16; 1024]  3 KiB stack
// Together with call frames the high-water mark is well under 16 KiB.
// The verifier never allocates on the heap, but the `linked_list_allocator`
// global allocator needs a few bytes during boot to lay down its initial
// hole header (`size_of::<Hole>()` ≈ 24 B). Anything ≥ 4 KiB is plenty;
// we set a conservative 16 KiB to leave headroom for runtime / allocator
// internals without compromising trace length.
#[jolt::provable(
    profile = "guest-profile",
    stack_size = 65_536,
    heap_size = 16_384,
    max_trace_length = 8_388_608,
    max_input_size = 4096
)]
fn fn_dsa_verify(pk: &[u8], sig: &[u8], msg: &[u8]) {
    // `black_box` on inputs prevents the optimizer from constant-folding
    // across phase boundaries or eliding work between cycle markers.
    let pk = black_box(pk);
    let sig = black_box(sig);
    let msg = black_box(msg);

    // Phase 1: decode the verifying key.
    //   * Absorb `pk` through SHAKE256 to obtain the 64-byte BUFF tag
    //     `hashed_key` (one of the algorithm's SHAKE256 absorbs).
    //   * Decode the packed `h` polynomial (14 bits per coefficient).
    //   * Transform `h` to NTT representation in-place.
    start_cycle_tracking("phase1_decode_vk");
    let vk = VerifyingKey512::decode(pk).unwrap_or_spoil_proof();
    let vk = black_box(vk);
    end_cycle_tracking("phase1_decode_vk");

    // Phase 2: verify_internal.
    //   * Decode the signature: header byte, 40-byte nonce, Comp-encoded s2.
    //   * Compute c <- SHAKE256(nonce ‖ hashed_key ‖ 0x00 ‖ len(ctx) ‖ ctx
    //                          ‖ msg) by rejection-sampling into Z[X]/(X^n+1).
    //   * NTT(s2); pointwise s2 * h; iNTT to recover s1 = c - s2*h.
    //   * Accept iff `‖s1‖^2 + ‖s2‖^2 <= ⌊β^2⌋ = SQBETA[logn]`.
    //
    // The single library call below performs all of the above. We treat a
    // `false` return as a malicious / malformed input and spoil the proof
    // rather than emitting a provable "verification rejected" trace.
    start_cycle_tracking("phase2_verify_internal");
    let accepted = vk.verify(sig, &DOMAIN_NONE, &HASH_ID_RAW, msg);
    let accepted = black_box(accepted);
    end_cycle_tracking("phase2_verify_internal");

    if !accepted {
        spoil_proof();
    }
}
