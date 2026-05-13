use std::time::Instant;

use fn_dsa::{
    sign_key_size, signature_size, vrfy_key_size, KeyPairGenerator, KeyPairGeneratorStandard,
    SigningKey, SigningKeyStandard, VerifyingKey, VerifyingKeyStandard, DOMAIN_NONE,
    FN_DSA_LOGN_512, HASH_ID_RAW,
};
use jolt_sdk::serialize_and_print_size;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha20Rng;
use tracing::info;
use tracing_subscriber::EnvFilter;

pub fn main() {
    // Default to `info` so the timing/cycle log lines show without the user
    // having to remember `RUST_LOG=info`. Users can still override via the
    // env var (e.g. `RUST_LOG=debug` for prover-internal traces).
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
    // ChaCha20 is a CSPRNG, so this satisfies fn-dsa's
    // `rand_core::CryptoRng + RngCore` bound while still being reproducible.
    let seed: [u8; 32] = core::array::from_fn(|i| (i as u8) ^ 0xA5);
    let mut rng = ChaCha20Rng::from_seed(seed);

    let mut sign_key = vec![0u8; sign_key_size(FN_DSA_LOGN_512)];
    let mut vrfy_key = vec![0u8; vrfy_key_size(FN_DSA_LOGN_512)];
    let mut kg = KeyPairGeneratorStandard::default();
    kg.keygen(FN_DSA_LOGN_512, &mut rng, &mut sign_key, &mut vrfy_key);

    let mut sk = SigningKeyStandard::decode(&sign_key).expect("decoded our own signing key");
    let msg = b"FN-DSA-512 hello, world!".to_vec();
    let mut sig = vec![0u8; signature_size(sk.get_logn())];

    // FN-DSA signing is randomized (the nonce is freshly sampled). Use a
    // separate RNG stream so the keypair derivation is independent of the
    // signing randomness.
    let mut sig_rng = ChaCha20Rng::from_seed([0x5Au8; 32]);
    sk.sign(&mut sig_rng, &DOMAIN_NONE, &HASH_ID_RAW, &msg, &mut sig);

    // Sanity check: the unmodified upstream verifier must accept our
    // signature before we ask Jolt to prove that acceptance.
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

    let prove_start = Instant::now();
    let (output, proof, program_io) = prove_fn_dsa_verify(&vrfy_key, &sig, &msg);
    let prove_elapsed = prove_start.elapsed();

    serialize_and_print_size("Proof", "/tmp/fn_dsa_proof.bin", &proof)
        .expect("Could not serialize proof.");

    let verify_start = Instant::now();
    let is_valid = verify_fn_dsa_verify(&vrfy_key, &sig, &msg, output, program_io.panic, proof);
    let verify_elapsed = verify_start.elapsed();

    info!("guest panicked: {}", program_io.panic);
    info!("proof valid:    {is_valid}");
    info!("Prove time:     {:.3} s", prove_elapsed.as_secs_f64());
    info!(
        "Verify time:    {:.3} ms",
        verify_elapsed.as_secs_f64() * 1000.0
    );
}
