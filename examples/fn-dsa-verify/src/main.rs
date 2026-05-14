use std::time::Instant;

// Force-link the Keccak inline crate so its `register_inlines!` static lands
// in `inventory`. Otherwise the linker drops it (the host binary never
// references its symbols directly — only the guest does, via fn-dsa-comm-jolt's
// KeccakState::process).
use jolt_inlines_keccak256 as _;

use fn_dsa::{
    sign_key_size, signature_size, vrfy_key_size, KeyPairGenerator, KeyPairGeneratorStandard,
    SigningKey, SigningKeyStandard, VerifyingKey, VerifyingKeyStandard, DOMAIN_NONE,
    FN_DSA_LOGN_512, HASH_ID_RAW,
};
use fn_dsa_vrfy::{compute_hashed_key, hash_to_point};
use guest::precomputed::serialize_c;
use jolt_sdk::serialize_and_print_size;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha20Rng;
use tracing::info;
use tracing_subscriber::EnvFilter;

pub fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_fn_dsa_verify(target_dir);

    let shared_preprocessing = guest::preprocess_shared_fn_dsa_verify(&mut program).unwrap();
    let prover_preprocessing = guest::preprocess_prover_fn_dsa_verify(shared_preprocessing.clone());
    let verifier_setup = prover_preprocessing.generators.to_verifier_setup();
    let verifier_preprocessing =
        guest::preprocess_verifier_fn_dsa_verify(shared_preprocessing, verifier_setup, None);

    let prove_fn_dsa_verify = guest::build_prover_fn_dsa_verify(program, prover_preprocessing);
    let verify_fn_dsa_verify = guest::build_verifier_fn_dsa_verify(verifier_preprocessing);

    // Deterministic FN-DSA-512 keypair derived from a fixed 32-byte seed.
    let seed: [u8; 32] = core::array::from_fn(|i| (i as u8) ^ 0xA5);
    let mut rng = ChaCha20Rng::from_seed(seed);

    let mut sign_key = vec![0u8; sign_key_size(FN_DSA_LOGN_512)];
    let mut vrfy_key = vec![0u8; vrfy_key_size(FN_DSA_LOGN_512)];
    let mut kg = KeyPairGeneratorStandard::default();
    kg.keygen(FN_DSA_LOGN_512, &mut rng, &mut sign_key, &mut vrfy_key);

    let mut sk = SigningKeyStandard::decode(&sign_key).expect("decoded our own signing key");
    let msg = b"FN-DSA-512 hello, world!".to_vec();
    let mut sig = vec![0u8; signature_size(sk.get_logn())];

    let mut sig_rng = ChaCha20Rng::from_seed([0x5Au8; 32]);
    sk.sign(&mut sig_rng, &DOMAIN_NONE, &HASH_ID_RAW, &msg, &mut sig);

    let host_vk = VerifyingKeyStandard::decode(&vrfy_key).expect("decoded our own verifying key");
    assert!(
        host_vk.verify(&sig, &DOMAIN_NONE, &HASH_ID_RAW, &msg),
        "upstream verify should accept upstream signature"
    );

    info!(
        "pk: {} bytes, sig: {} bytes, msg: {} bytes",
        vrfy_key.len(),
        sig.len(),
        msg.len()
    );

    // ---- Host-side precomputation ------------------------------------
    //
    // hashed_key = SHAKE256(pk, 64)         ~7 SHAKE256 perms
    // c = hash_to_point(nonce ‖ hashed_key ‖ 0x00 ‖ len(ctx) ‖ ctx ‖ msg)
    //                                       ~8 SHAKE256 perms (reject-sample
    //                                        512 coefficients < q=12289)
    //
    // After this split the only Keccak work left inside the proof is NTT
    // and lattice arithmetic — no SHAKE at all.
    let hashed_key = compute_hashed_key(&vrfy_key);
    let n = 1usize << FN_DSA_LOGN_512;
    let nonce = &sig[1..41];
    let mut c = vec![0u16; n];
    hash_to_point(nonce, &hashed_key, &DOMAIN_NONE, &HASH_ID_RAW, &msg, &mut c);
    let c_bytes = serialize_c(&c);
    info!("precomputed: hashed_key=64 B, c={} B", c_bytes.len());

    let prove_start = Instant::now();
    let (output, proof, program_io) =
        prove_fn_dsa_verify(&vrfy_key, &sig, &msg, &hashed_key, &c_bytes);
    let prove_elapsed = prove_start.elapsed();

    serialize_and_print_size("Proof", "/tmp/fn_dsa_proof.bin", &proof)
        .expect("Could not serialize proof.");

    let verify_start = Instant::now();
    let is_valid = verify_fn_dsa_verify(
        &vrfy_key,
        &sig,
        &msg,
        &hashed_key,
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
