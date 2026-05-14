//! ML-DSA signature verification.

use crate::{
    B32, B64, EncodedVerifyingKey, MlDsaParams, MuBuilder, Polynomial, Signature,
    algebra::{Elem, NttMatrix, NttVector, Vector},
    crypto::H,
    ntt::{Ntt, NttInverse},
    param::ParameterSet,
    param::VerifyingKeySize,
    sampling::{expand_a, sample_in_ball},
};
use common::{Key, KeyExport, KeyInit, KeySizeUser};
use module_lattice::MaybeBox;
use signature::{Error, MultipartVerifier};

/// An ML-DSA verification key.
#[derive(Clone, Debug, PartialEq)]
pub struct VerifyingKey<P: ParameterSet> {
    /// Public seed used to deterministically re-expand `A_hat`.
    rho: B32,

    /// High bits of the public key polynomial `t`.
    t1: MaybeBox<Vector<P::K>>,

    /// Precomputed expanded values.
    precomputed_values: MaybeBox<PrecomputedValues<P>>,
}

/// Cached values derived from `rho` and `t1` at key construction time to avoid re-expanding them
/// when verifying signatures.
#[derive(Clone, Debug, PartialEq)]
struct PrecomputedValues<P: ParameterSet> {
    /// Expanded public matrix in NTT domain.
    A_hat: NttMatrix<P::K, P::L>,

    /// `2ᵈ ⋅ t1` which can be reused in signature verification.
    t1_2d_hat: NttVector<P::K>,

    /// Hash of the encoded public key, used to bind messages to the key this was precomputed from.
    tr: B64,
}

impl<P: MlDsaParams> PrecomputedValues<P> {
    fn new(t1: &Vector<P::K>, enc: &EncodedVerifyingKey<P>, A_hat: NttMatrix<P::K, P::L>) -> Self {
        let t1_2d_hat = (Elem::new(1 << 13) * t1).ntt();
        let tr = H::default().absorb(enc).squeeze_new();

        Self {
            A_hat,
            t1_2d_hat,
            tr,
        }
    }

    /// Build a `PrecomputedValues` from already-derived inputs.
    ///
    /// Use this when the caller has computed `A_hat = ExpandA(rho)` and
    /// `tr = SHAKE256(pk_encoded)` outside the proof boundary (e.g. so
    /// those SHAKE permutations don't run inside a Jolt guest). The
    /// caller is responsible for the correctness of the inputs; passing
    /// values that don't actually correspond to `rho` / `pk` will produce
    /// a `VerifyingKey` that accepts wrong signatures.
    fn from_parts(t1: &Vector<P::K>, A_hat: NttMatrix<P::K, P::L>, tr: B64) -> Self {
        let t1_2d_hat = (Elem::new(1 << 13) * t1).ntt();
        Self {
            A_hat,
            t1_2d_hat,
            tr,
        }
    }
}

impl<P: MlDsaParams> VerifyingKey<P> {
    /// Construct a `VerifyingKey` from already-decoded `(rho, t1)` plus an
    /// already-expanded `A_hat`.
    ///
    /// This is the building block for "lift the public hashing out of the
    /// proof": the caller computes `A_hat = expand_a(rho)` once on the host,
    /// then passes the result here so the SHAKE128 permutations of `ExpandA`
    /// don't run inside a Jolt guest.
    ///
    /// `enc` may be `None`, in which case the encoded form is reconstructed
    /// from `(rho, t1)`. Reconstruction is required because `tr = H(pk_enc)`
    /// is still computed inside; for the "skip H(pk) too" variant use
    /// [`VerifyingKey::new_with_precomputed`] instead.
    pub fn new(
        rho: B32,
        t1: Vector<P::K>,
        A_hat: NttMatrix<P::K, P::L>,
        enc: Option<EncodedVerifyingKey<P>>,
    ) -> Self {
        let enc = enc.unwrap_or_else(|| Self::encode_internal(&rho, &t1));
        let precomputed_values = PrecomputedValues::new(&t1, &enc, A_hat);

        Self {
            rho,
            t1: MaybeBox::new(t1),
            precomputed_values: MaybeBox::new(precomputed_values),
        }
    }

    /// Construct a `VerifyingKey` from fully precomputed parts: `(rho, t1)`
    /// from `pkDecode`, the expanded matrix `A_hat = ExpandA(rho)`, and the
    /// BUFF tag `tr = SHAKE256(pk_enc, 64)`.
    ///
    /// Compared to [`VerifyingKey::new`], this skips the `H(pk_enc)` SHAKE256
    /// absorb that would otherwise run inside the proof. Intended for callers
    /// (e.g. Jolt guest programs) that have computed `A_hat` and `tr` on the
    /// untrusted host before invoking the prover. The caller MUST recompute
    /// `A_hat` and `tr` from the public key during their own verification,
    /// otherwise an attacker can supply mismatched values and forge.
    pub fn new_with_precomputed(
        rho: B32,
        t1: Vector<P::K>,
        A_hat: NttMatrix<P::K, P::L>,
        tr: B64,
    ) -> Self {
        let precomputed_values = PrecomputedValues::from_parts(&t1, A_hat, tr);

        Self {
            rho,
            t1: MaybeBox::new(t1),
            precomputed_values: MaybeBox::new(precomputed_values),
        }
    }

    #[inline]
    fn new_expand_a(rho: B32, t1: Vector<P::K>, enc: Option<EncodedVerifyingKey<P>>) -> Self {
        let A_hat = expand_a(&rho);
        Self::new(rho, t1, A_hat, enc)
    }

    /// Implementation of Algorithm 8: `ML-DSA.Verify_internal` algorithm from FIPS 204.
    ///
    /// It does not include the domain separator that distinguishes between the normal and
    /// pre-hashed cases, and it does not separate the context string from the rest of the message.
    pub fn verify_internal(&self, M: &[u8], sigma: &Signature<P>) -> bool
    where
        P: MlDsaParams,
    {
        let mu = MuBuilder::internal(&self.precomputed_values.tr, &[M]);
        self.raw_verify_mu(&mu, sigma)
    }

    pub(crate) fn raw_verify_mu(&self, mu: &B64, sigma: &Signature<P>) -> bool
    where
        P: MlDsaParams,
    {
        // Reconstruct w
        let c = sample_in_ball(&sigma.c_tilde, P::TAU);
        self.raw_verify_with_precomputed(mu, &c, sigma)
    }

    /// Verify with both `μ` and `c = SampleInBall(c̃)` precomputed.
    ///
    /// This is the most "stripped-down" verify path: it performs only the
    /// lattice arithmetic (NTTs, matrix-vector mul, hint reconstruction) and
    /// the final `c̃' = H(μ ‖ w1Encode(w1'))` SHAKE256, which is the only
    /// hash that genuinely depends on the lattice computation and therefore
    /// cannot be lifted out of the proof.
    ///
    /// Combined with [`VerifyingKey::new_with_precomputed`], this lets a
    /// caller move *all* publicly determined SHAKE work — `ExpandA(rho)`,
    /// `SHAKE256(pk)`, `SHAKE256(tr ‖ M)`, and `SampleInBall(c̃)` — to the
    /// host, leaving only ~8 SHAKE256 permutations inside the proof. The
    /// caller MUST recompute `c = SampleInBall(sigma.c_tilde)` during
    /// verification, otherwise the security argument breaks.
    pub fn raw_verify_with_precomputed(
        &self,
        mu: &B64,
        c: &Polynomial,
        sigma: &Signature<P>,
    ) -> bool
    where
        P: MlDsaParams,
    {
        let z_hat = sigma.z.ntt();
        let c_hat = c.ntt();
        let Az_hat = &self.precomputed_values.A_hat * &z_hat;
        let ct1_2d_hat = &c_hat * &self.precomputed_values.t1_2d_hat;

        let wp_approx = (&Az_hat - &ct1_2d_hat).ntt_inverse();
        let w1p = sigma.h.use_hint(&wp_approx);

        let w1p_tilde = P::encode_w1(&w1p);
        let cp_tilde = H::default()
            .absorb(mu)
            .absorb(&w1p_tilde)
            .squeeze_new::<P::Lambda>();

        sigma.c_tilde == cp_tilde
    }

    /// Implementation of Algorithm 3: `ML-DSA.Verify` from FIPS 204.
    pub fn verify_with_context(&self, M: &[u8], ctx: &[u8], sigma: &Signature<P>) -> bool {
        self.raw_verify_with_context(&[M], ctx, sigma)
    }

    /// Implementation of Algorithm 3: `ML-DSA.Verify` from FIPS 204 with a pre-computed μ.
    pub fn verify_mu(&self, mu: &B64, sigma: &Signature<P>) -> bool {
        self.raw_verify_mu(mu, sigma)
    }

    fn raw_verify_with_context(&self, M: &[&[u8]], ctx: &[u8], sigma: &Signature<P>) -> bool {
        if ctx.len() > 255 {
            return false;
        }

        let mu = MuBuilder::new(&self.precomputed_values.tr, ctx).message(M);
        self.verify_mu(&mu, sigma)
    }

    pub(crate) fn encode_internal(rho: &B32, t1: &Vector<P::K>) -> EncodedVerifyingKey<P> {
        let t1_enc = P::encode_t1(t1);
        P::concat_vk(rho.clone(), t1_enc)
    }

    /// Encode the key in a fixed-size byte array.
    ///
    /// Implementation of Algorithm 22: `pkEncode` from FIPS 204.
    #[must_use]
    pub fn encode(&self) -> EncodedVerifyingKey<P> {
        Self::encode_internal(&self.rho, &self.t1)
    }

    /// Decode the key from an appropriately sized byte array.
    ///
    /// Implementation of Algorithm 23: `pkDecode` from FIPS 204.
    pub fn decode(enc: &EncodedVerifyingKey<P>) -> Self {
        let (rho, t1_enc) = P::split_vk(enc);
        let t1 = P::decode_t1(t1_enc);
        Self::new_expand_a(rho.clone(), t1, Some(enc.clone()))
    }
}

impl<P: MlDsaParams> KeySizeUser for VerifyingKey<P> {
    type KeySize = VerifyingKeySize<P>;
}

impl<P: MlDsaParams> KeyInit for VerifyingKey<P> {
    fn new(key: &Key<Self>) -> Self {
        Self::decode(key)
    }
}

impl<P: MlDsaParams> KeyExport for VerifyingKey<P> {
    fn to_bytes(&self) -> Key<Self> {
        self.encode()
    }
}

impl<P: MlDsaParams> core::hash::Hash for VerifyingKey<P> {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.encode().hash(state);
    }
}

impl<P: MlDsaParams> signature::Verifier<Signature<P>> for VerifyingKey<P> {
    fn verify(&self, msg: &[u8], signature: &Signature<P>) -> Result<(), Error> {
        self.multipart_verify(&[msg], signature)
    }
}

impl<P: MlDsaParams> MultipartVerifier<Signature<P>> for VerifyingKey<P> {
    fn multipart_verify(&self, msg: &[&[u8]], signature: &Signature<P>) -> Result<(), Error> {
        self.raw_verify_with_context(msg, &[], signature)
            .then_some(())
            .ok_or(Error::new())
    }
}
