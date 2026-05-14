use std::time::Instant;

// Force-link the Keccak inline crate so its `register_inlines!` static lands
// in `inventory`. Otherwise the linker drops it (the host binary never
// references its symbols directly — only the guest does, via ml-dsa-jolt's
// SHAKE backend).
use jolt_inlines_keccak256 as _;

use guest::precomputed::{polynomial_to_signed_bytes, serialize_a_hat};
use jolt_sdk::serialize_and_print_size;
use ml_dsa::{
    compute_mu_with_context, compute_tr, expand_a, sample_in_ball, MlDsa65, ParameterSet,
    SigningKey, VerifyingKeyParams,
};
use signature::{Keypair, Signer, Verifier};
use tracing::info;

pub fn main() {
    tracing_subscriber::fmt::init();

    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_mldsa_verify(target_dir);

    let shared_preprocessing = guest::preprocess_shared_mldsa_verify(&mut program).unwrap();
    let prover_preprocessing = guest::preprocess_prover_mldsa_verify(shared_preprocessing.clone());
    let verifier_setup = prover_preprocessing.generators.to_verifier_setup();
    let verifier_preprocessing =
        guest::preprocess_verifier_mldsa_verify(shared_preprocessing, verifier_setup, None);

    let prove_mldsa_verify = guest::build_prover_mldsa_verify(program, prover_preprocessing);
    let verify_mldsa_verify = guest::build_verifier_mldsa_verify(verifier_preprocessing);

    // Deterministic ML-DSA-65 keypair derived from a fixed 32-byte seed.
    let seed_bytes: [u8; 32] = core::array::from_fn(|i| (i as u8) ^ 0xA5);
    let sk = SigningKey::<MlDsa65>::from_seed(&seed_bytes.into());
    let vk = sk.verifying_key();

    let msg = b"ML-DSA-65 hello, world!".to_vec();
    let signature = sk.sign(&msg);

    vk.verify(&msg, &signature)
        .expect("upstream verify should accept upstream signature");

    let pk = vk.encode().to_vec();
    let sig = signature.encode().to_vec();

    info!(
        "pk: {} bytes, sig: {} bytes, msg: {} bytes",
        pk.len(),
        sig.len(),
        msg.len()
    );

    // ---- Host-side precomputation -------------------------------------
    //
    // Lift every public hashing step that doesn't depend on the lattice
    // arithmetic out of the proof. The verifier (also us, here) MUST
    // recompute these values from `(pk, msg, sig)` independently before
    // calling `Jolt.verify(...)`, otherwise an attacker could supply
    // mismatched values and forge.
    //
    //   A_hat = ExpandA(rho)            (~165–225 SHAKE128 perms)
    //   tr    = SHAKE256(pk, 64)        (~15 SHAKE256 perms)
    //   μ     = SHAKE256(tr ‖ M, 64)    (~2 SHAKE256 perms)
    //   c     = SampleInBall(c̃)         (~1 SHAKE256 perm)
    //
    // After this split, the only SHAKE call left in the proof is
    // `c̃' = SHAKE256(μ ‖ w1Encode(w₁'))` (~8 SHAKE256 perms), which
    // genuinely depends on the lattice computation and can't be lifted.
    let vk_enc = vk.encode();
    let (rho, _t1_enc) = MlDsa65::split_vk(&vk_enc);
    let a_hat = expand_a::<<MlDsa65 as ParameterSet>::K, <MlDsa65 as ParameterSet>::L>(rho);
    let a_hat_bytes = serialize_a_hat(&a_hat);
    let tr = compute_tr(&pk);
    // The host signs with `sk.sign(msg)`, which is the public-facing
    // `Signer` trait → `MultipartSigner` → `raw_sign_deterministic(msg, &[])`.
    // That uses `MuBuilder::new(tr, ctx=[])` — i.e. μ includes the domain
    // separator (0x00) and context prefix (length || bytes), not just `tr ‖ msg`.
    // We must hash with the same shape so verify accepts.
    let mu = compute_mu_with_context(&tr, &[], &[&msg]);
    let c = sample_in_ball(signature.c_tilde(), MlDsa65::TAU);
    let c_bytes = polynomial_to_signed_bytes(&c);
    let tr_bytes: [u8; 64] = tr.into();
    let mu_bytes: [u8; 64] = mu.into();
    info!(
        "precomputed: A_hat={} B, tr=64 B, mu=64 B, c=256 B",
        a_hat_bytes.len()
    );

    let prove_start = Instant::now();
    let (output, proof, program_io) = prove_mldsa_verify(
        &pk,
        &msg,
        &sig,
        &a_hat_bytes,
        &tr_bytes,
        &mu_bytes,
        &c_bytes,
    );
    let prove_elapsed = prove_start.elapsed();

    serialize_and_print_size("Proof", "/tmp/mldsa_proof.bin", &proof)
        .expect("Could not serialize proof.");

    let verify_start = Instant::now();
    let is_valid = verify_mldsa_verify(
        &pk,
        &msg,
        &sig,
        &a_hat_bytes,
        &tr_bytes,
        &mu_bytes,
        &c_bytes,
        output,
        program_io.panic,
        proof,
    );
    let verify_elapsed = verify_start.elapsed();

    info!("guest panicked: {}", program_io.panic);
    info!("proof valid:    {is_valid}");
    info!("Prove time:     {:.3} s", prove_elapsed.as_secs_f64());
    info!(
        "Verify time:    {:.3} ms",
        verify_elapsed.as_secs_f64() * 1000.0
    );
}
