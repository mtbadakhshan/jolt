use std::time::Instant;

// Force-link the Keccak inline crate so its `register_inlines!` static lands
// in `inventory`. Otherwise the linker drops it (the host binary never
// references its symbols directly — only the guest does, via the patched
// `keccak` crate).
use jolt_inlines_keccak256 as _;

use jolt_sdk::serialize_and_print_size;
use ml_dsa::{MlDsa65, SigningKey};
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

    let prove_start = Instant::now();
    let (output, proof, program_io) = prove_mldsa_verify(&pk, &msg, &sig);
    let prove_elapsed = prove_start.elapsed();

    serialize_and_print_size("Proof", "/tmp/mldsa_proof.bin", &proof)
        .expect("Could not serialize proof.");

    let verify_start = Instant::now();
    let is_valid = verify_mldsa_verify(&pk, &msg, &sig, output, program_io.panic, proof);
    let verify_elapsed = verify_start.elapsed();

    info!("guest panicked: {}", program_io.panic);
    info!("proof valid:    {is_valid}");
    info!("Prove time:     {:.3} s", prove_elapsed.as_secs_f64());
    info!(
        "Verify time:    {:.3} ms",
        verify_elapsed.as_secs_f64() * 1000.0
    );
}
