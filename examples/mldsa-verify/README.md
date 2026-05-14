# Proving ML-DSA-65 verification inside Jolt

A worked example: take the upstream
[`ml-dsa`](https://github.com/RustCrypto/signatures/tree/master/ml-dsa)
crate from RustCrypto — an implementation of **ML-DSA-65** per
[NIST FIPS 204](https://csrc.nist.gov/pubs/fips/204/final) — run it as
a Jolt guest, and prove its execution. We vendor the crate at
`crates/ml-dsa-jolt/` so we can:

1. **Lift every public hashing step that doesn't depend on the lattice
   arithmetic out of the proof.** The host runs `ExpandA(ρ)`,
   `SHAKE256(pk)`, `SHAKE256(tr ‖ M)`, and `SampleInBall(c̃)` itself
   before invoking the prover, then ships the results into the guest as
   additional public inputs. This is sound because the application
   *also* recomputes those values from `(pk, msg, sig)` during
   verification — passing them in only saves prover work, it doesn't
   change the trust model.
2. **Route the remaining SHAKE call directly through the Jolt Keccak
   inline.** `crates/ml-dsa-jolt/src/crypto.rs` ships its own
   SHAKE128/256 wrapper that calls `jolt_inlines_keccak256` directly on
   RISC-V (and a soft Keccak permutation on the host). No more
   `sha3 → keccak → keccak-jolt patch → inline` chain.

**Upstream crates used (unmodified):**

- [`module-lattice`](https://crates.io/crates/module-lattice) — NTT and
  ring arithmetic over `R_q`. Pulled in transitively.
- [`hybrid-array`](https://github.com/RustCrypto/utils/tree/master/hybrid-array)
  — type-level-sized arrays.
- [`signature`](https://github.com/RustCrypto/traits/tree/master/signature)
  — the `Signer` / `Verifier` traits.

**Vendored crate (workspace-local fork):**

- `crates/ml-dsa-jolt/` — a near-verbatim copy of `ml-dsa 0.1.0-rc.11`
  with three changes: (a) the `ShakeState` type in `crypto.rs` is
  rewritten to call the Jolt Keccak inline directly (no `sha3` /
  `keccak` deps); (b) `VerifyingKey::new_with_precomputed` and
  `raw_verify_with_precomputed` are added so the host can inject
  `Â`, `tr`, and `c`; (c) `compute_tr` / `compute_mu_with_context` are
  added as public helpers that mirror the SHAKE shape ML-DSA's
  `Sign`/`Verify` use internally. Slotted in via `[patch.crates-io]
  ml-dsa = { path = "./crates/ml-dsa-jolt" }`.

So the layering at runtime in the guest is now:

```
ml-dsa-jolt::VerifyingKey::raw_verify_with_precomputed
              │
              ↓
         ml-dsa-jolt::crypto::ShakeState  (only one SHAKE call left:
              │                            c̃' = H(μ ‖ w1Encode(w₁')))
              ↓
         Jolt Keccak inline opcode
```

This document is in four parts:

1. **The algorithm** — what ML-DSA verification actually does (FIPS 204 §5.3).
2. **The implementation** — how it maps to a Jolt guest example, the
   host/guest split, and what the `ml-dsa-jolt` vendoring does.
3. **Measurements** — prove time, proof size, verify time, per-phase and
   per-function cycle counts, and where the budget actually goes.
4. **The trust model** — why precomputing public hashes on the
   (untrusted) host is sound.

---

## 1. ML-DSA-65 verification — what it does

ML-DSA (Module-Lattice Digital Signature Algorithm) is the lattice-based
post-quantum digital signature standardized by NIST as **FIPS 204** in 2024.
ML-DSA-65 is the recommended security level (category 3, ≈ 192-bit
classical-equivalent security). It operates over the ring

\[
  R_q = \mathbb{Z}_q[X]\,/\,(X^{256} + 1), \quad q = 2^{23} - 2^{13} + 1 = 8\,380\,417.
\]

### Public objects (sizes for ML-DSA-65)

| Object | Size | Contents |
|---|---:|---|
| Public key `pk` | 1 952 B | `ρ` (32 B) ‖ packed `t₁` (1 920 B) — k = 6 polynomials, 10 bits per coefficient |
| Signature `σ` | 3 309 B | `c̃` (48 B) ‖ packed `z` (3 200 B, l = 5 polys × 20 bits/coeff) ‖ hint `h` (61 B) |
| Message `M` | variable | the bytes being signed |

### Verify algorithm (FIPS 204 §5.3, Algorithm 3)

```text
ML-DSA.Verify(pk, M, σ)
─────────────────────────────────────────────────────────────────────────────
 1.   (ρ, t₁) ← pkDecode(pk)
 2.   (c̃, z, h) ← sigDecode(σ);   if anything malformed     →  reject
 3.   if ‖z‖∞ ≥ γ₁ − β   or   weight(h) > ω                  →  reject
 4.   Â  ← ExpandA(ρ)                   (k × l matrix, NTT-domain, via SHAKE128)
 5.   tr ← H(pk, 64)                    (SHAKE256)
 6.   μ  ← H(tr ‖ M', 64)               (SHAKE256;  M' includes context bytes)
 7.   c  ← SampleInBall(c̃)              (SHAKE256-driven; τ = 49 ones in {−1,+1})
 8.   Aẑ        ← Â · NTT(z)
       ĉt₁·2ᵈ   ← NTT(c) · NTT(t₁ · 2¹³)
       w'_approx ← iNTT( Aẑ − ĉt₁·2ᵈ )
 9.   w₁' ← UseHint(h, w'_approx)
10.   c̃' ← H(μ ‖ w1Encode(w₁'), 2λ)     (SHAKE256)
11.   accept iff c̃' == c̃
```

The heavy primitives are:

- **SHAKE128 / SHAKE256** (Keccak-f[1600] sponge) — used by `ExpandA`,
  `SampleInBall`, and three Fiat–Shamir hashes.
- **Number-theoretic transform (NTT) and its inverse** — over the
  256-coefficient ring `R_q`. ML-DSA-65 verify performs **17 forward NTTs**
  (5 on `z`, 1 on `c`, plus precomputed `NTT(t₁·2¹³)`) and **6 inverse NTTs**
  on `w'_approx`.
- **Pointwise polynomial multiplication and addition** in the NTT domain
  (≈ 36 multiplications).
- **Hint reconstruction** (`Decompose`, `HighBits`, `UseHint`).
- Bit-level decoders for `t₁`, `z`, `h`.

Everything else is constant work or simple arithmetic.

---

## 2. How we implemented it

### Design rule

> Vendor a thin Jolt-aware copy of `ml-dsa` at `crates/ml-dsa-jolt/`. Use
> it to (a) call the Jolt Keccak inline directly, no patch chain, and
> (b) expose entry points that let the host precompute every publicly
> determined SHAKE before invoking the prover. Don't touch the lattice
> code, NTT, encoding, or hint reconstruction.

### Crate layout

```
crates/ml-dsa-jolt/                   ← workspace [patch.crates-io] for `ml-dsa 0.1.0-rc.11`
├── Cargo.toml                          name = "ml-dsa", version = "0.1.0-rc.11"
│                                       (drops sha3 / pkcs8 / const-oid / zeroize deps;
│                                        adds jolt-inlines-keccak256 on RISC-V only)
└── src/
    ├── lib.rs                         + compute_tr, compute_mu_with_context, compute_mu_internal
    │                                  + pub re-exports of the algebra / sampling types
    ├── crypto.rs                      REWRITTEN: ShakeState backed by Jolt Keccak inline directly
    │                                  (no sha3 dep, no keccak crate)
    ├── verifying.rs                   + VerifyingKey::new_with_precomputed
    │                                  + VerifyingKey::raw_verify_with_precomputed
    ├── sampling.rs                    pub fn expand_a / sample_in_ball (was pub(crate))
    └── algebra.rs / encode.rs / hint.rs / ntt.rs / param.rs / signing.rs
                                       vendored verbatim from upstream

examples/mldsa-verify/
├── Cargo.toml                         host: ml-dsa + rand + jolt-sdk
├── src/main.rs                        prove + verify driver, runs host-side ExpandA / SHAKE
├── src/bin/profile.rs                 per-symbol cycle profiler
└── guest/
    ├── Cargo.toml                     guest: jolt-sdk + ml-dsa + hybrid-array
    └── src/
        ├── lib.rs                     #[jolt::provable] entry point with extra precomputed inputs
        └── precomputed.rs             ~50 LoC of (de)serialization helpers for A_hat / c
```

### What `crates/ml-dsa-jolt` changes versus upstream

The diff against [`ml-dsa 0.1.0-rc.11`](https://crates.io/crates/ml-dsa/0.1.0-rc.11) is
small and self-contained:

1. **Drop the `sha3` / `keccak` dependency chain.** Upstream's
   `crypto.rs` defines `ShakeState<Shake>` parameterized over
   `sha3::Shake128` / `sha3::Shake256`. We replace it with our own
   `ShakeState<const RATE: usize>` whose `absorb` / `squeeze` /
   `squeeze_new<N>` methods have the same signatures (so the rest of
   the vendored code is untouched), backed by:
   ```rust
   // RISC-V: direct Jolt inline opcode.
   #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
   fn keccak_f1600(state: &mut [u64; 25]) {
       use jolt_inlines_keccak256::{INLINE_OPCODE, KECCAK256_FUNCT3, KECCAK256_FUNCT7};
       unsafe {
           core::arch::asm!(
               ".insn r {opcode}, {funct3}, {funct7}, x0, {rs1}, x0",
               opcode = const INLINE_OPCODE,
               funct3 = const KECCAK256_FUNCT3,
               funct7 = const KECCAK256_FUNCT7,
               rs1    = in(reg) state.as_mut_ptr(),
               options(nostack),
           );
       }
   }
   // Host: soft Keccak-f[1600] ported verbatim from `keccak 0.2.0`.
   ```
2. **Add precomputed entry points.** `VerifyingKey::new` is relaxed
   from `pub(crate)` to `pub`, and a new `new_with_precomputed(rho, t1,
   A_hat, tr)` skips the `tr = SHAKE256(pk)` step. A new
   `raw_verify_with_precomputed(mu, c, sigma)` is `raw_verify_mu` minus
   the `sample_in_ball(c̃, τ)` call. `compute_tr`, `compute_mu_internal`,
   and `compute_mu_with_context` are added as top-level helpers so the
   host can run those SHAKE calls with the same shapes the upstream
   `Sign`/`Verify` flows use.
3. **Drop the `DigestVerifier` / `DigestSigner` impls** that reference
   `sha3::Shake256` directly. They're not used by this example, and
   removing them is what lets us drop the `sha3` dep entirely.

Workspace integration is one line in the root `Cargo.toml`:

```toml
[patch.crates-io]
ml-dsa = { path = "./crates/ml-dsa-jolt" }
```

The previous `keccak = { path = "./crates/keccak-jolt" }` patch was
removed because nothing else in the workspace depends on `keccak 0.2.x`.

### The host / guest split

The guest's `#[jolt::provable]` entry point takes seven byte slices:

```rust
fn mldsa_verify(
    pk: &[u8], msg: &[u8], sig: &[u8],   // the actual signature inputs
    a_hat_bytes: &[u8],                  // 30 720 B  ─┐
    tr_bytes: &[u8],                     //     64 B   │ derived from (pk, msg, sig);
    mu_bytes: &[u8],                     //     64 B   │ host computes them and the
    c_bytes: &[u8],                      //    256 B  ─┘ verifier MUST recompute too
)
```

The four extra inputs are **deterministic functions of `(pk, msg, sig)`**.
Spelled out, each one is:

#### `a_hat_bytes` — `Â = ExpandA(ρ)`, the public matrix in NTT form

```text
(ρ, t₁) ← pkDecode(pk)                  // FIPS 204 Alg. 23: split first 32 B as ρ,
                                        // bit-unpack the rest as t₁
Â       ← ExpandA(ρ)                    // FIPS 204 Alg. 32: rejection-sample 6×5
                                        // = 30 NttPolynomials, each via
                                        // SHAKE128(ρ ‖ s_byte ‖ r_byte) → 840-byte
                                        // squeeze → coeff_from_three_bytes filter
serialize(Â) → 30 720 B                 // 6 × 5 × 256 little-endian u32
```

Where the SHAKE work happens: ~165–225 SHAKE128 permutations (varies with rejection-sampling luck across the 30 polynomials).

Host code: [`expand_a::<MlDsa65::K, MlDsa65::L>(rho)`](src/main.rs) → [`serialize_a_hat(&a_hat)`](guest/src/precomputed.rs).

#### `tr_bytes` — `tr = H(pk)`, the BUFF tag binding messages to a key

```text
tr ← SHAKE256(pk, 64 B)                 // FIPS 204 Alg. 7 step 6 / Alg. 8 step 6:
                                        // hash the entire encoded public key
                                        // (rho ‖ packed t₁) into a 64-byte tag
```

Where the SHAKE work happens: ~15 SHAKE256 permutations (rate 136 B, so 1 952 B `pk` spans 15 blocks).

Host code: [`compute_tr(&pk)`](src/main.rs) → wraps `H::default().absorb(pk).squeeze_new()`.

#### `mu_bytes` — `μ = H(tr ‖ ... ‖ M)`, the message hash

```text
μ ← SHAKE256(tr ‖ 0x00 ‖ len(ctx) ‖ ctx ‖ msg, 64 B)
                                        // FIPS 204 Alg. 2 step 10 / Alg. 3 step 6
                                        // (the public-facing Sign/Verify flow,
                                        // matching what `sk.sign(msg)` produces).
                                        // For empty ctx (this example):
                                        //   μ = SHAKE256(tr ‖ 0x00 ‖ 0x00 ‖ msg)
```

Where the SHAKE work happens: ~2 SHAKE256 permutations.

> **Internal-vs-with-context μ.** ML-DSA defines two μ shapes:
> `verify_internal` uses `μ = H(tr ‖ M)` (no domain separator), while the
> public-facing `Verify` adds the `0x00 ‖ len(ctx) ‖ ctx` prefix. Mixing
> them is a footgun: the helper used here MUST match how the signer hashed.
> The example uses `compute_mu_with_context` because `sk.sign(msg)` goes
> through the public flow; if you call `sign_internal` instead, switch to
> `compute_mu_internal` on both sides.

Host code: [`compute_mu_with_context(&tr, &[], &[&msg])`](src/main.rs) → wraps `MuBuilder::new(tr, ctx).message(&[msg])`.

#### `c_bytes` — `c = SampleInBall(c̃)`, the sparse challenge polynomial

```text
(c̃, z, h) ← sigDecode(sig)              // FIPS 204 Alg. 27: split sig into
                                        // c_tilde (first 48 B for ML-DSA-65),
                                        // packed z, hint bytes
c         ← SampleInBall(c̃, τ=49)       // FIPS 204 Alg. 29: rejection-sample
                                        // 49 ones in {-1, +1} from a SHAKE256
                                        // stream seeded with c̃; rest are 0
encode(c) → 256 B                       // signed-byte: -1 → 0xFF, 0 → 0x00, +1 → 0x01
```

Where the SHAKE work happens: ~1 SHAKE256 permutation (one absorb of the 48-byte `c̃`, then small squeezes).

Host code: [`sample_in_ball(signature.c_tilde(), MlDsa65::TAU)`](src/main.rs) → [`polynomial_to_signed_bytes(&c)`](guest/src/precomputed.rs).

#### Summary table

| Input         | Size      | Derived from   | Host call                                            | SHAKE perms saved |
|---------------|----------:|----------------|------------------------------------------------------|------------------:|
| `a_hat_bytes` | 30 720 B  | `pk` (just ρ)  | `expand_a(rho)`                                      |          ~165–225 |
| `tr_bytes`    |     64 B  | `pk`           | `compute_tr(&pk)`                                    |              ~15  |
| `mu_bytes`    |     64 B  | `pk`, `msg`    | `compute_mu_with_context(&tr, ctx, &[&msg])`         |               ~2  |
| `c_bytes`     |    256 B  | `sig` (just c̃) | `sample_in_ball(signature.c_tilde(), MlDsa65::TAU)`  |               ~1  |

`Â` is serialized as 6×5×256 little-endian `u32` coefficients;
`c` is encoded as 256 signed bytes (`-1 → 0xFF`, `0 → 0x00`,
`+1 → 0x01`) since `SampleInBall` only ever produces those values. Both
serialization helpers live in
[`examples/mldsa-verify/guest/src/precomputed.rs`](guest/src/precomputed.rs)
and are re-used by the host driver.

The total SHAKE work moved out of the proof: ~183–243 permutations,
i.e. essentially all of the verifier's Keccak budget. The only one that
*can't* be lifted is `c̃' = SHAKE256(μ ‖ w1Encode(w₁'))` (~8 perms),
because `w₁'` depends on the lattice computation `Â · ẑ − ĉ · t₁·2¹³`
that the proof itself attests to.

### The guest

```rust
// examples/mldsa-verify/guest/src/lib.rs (abridged)
#[jolt::provable(
    profile = "guest-profile",
    stack_size = 262_144,
    heap_size = 4_194_304,
    max_trace_length = 16_777_216,
    max_input_size = 65_536,           // bumped from 8 192 to fit A_hat
)]
fn mldsa_verify(
    pk: &[u8], msg: &[u8], sig: &[u8],
    a_hat_bytes: &[u8],                // 30 720 B
    tr_bytes: &[u8],                   // 64 B
    mu_bytes: &[u8],                   // 64 B
    c_bytes: &[u8],                    // 256 B
) {
    let tr: [u8; 64] = tr_bytes.try_into().unwrap_or_spoil_proof();
    let mu: [u8; 64] = mu_bytes.try_into().unwrap_or_spoil_proof();
    let c_bytes: &[u8; 256] = c_bytes.try_into().unwrap_or_spoil_proof();

    // Phase 1: pkDecode + reconstruct VerifyingKey from precomputed parts.
    let vk_enc = <&EncodedVerifyingKey<MlDsa65>>::try_from(pk).unwrap_or_spoil_proof();
    let (rho, t1_enc) = MlDsa65::split_vk(vk_enc);
    let t1 = MlDsa65::decode_t1(t1_enc);
    let a_hat = deserialize_a_hat(a_hat_bytes).unwrap_or_spoil_proof();
    let vk = VerifyingKey::<MlDsa65>::new_with_precomputed(rho.clone(), t1, a_hat, tr.into());

    // Phase 2: signature decode (unchanged).
    let sig = Signature::<MlDsa65>::try_from(sig).unwrap_or_spoil_proof();

    // Phase 3: lattice math + final c̃' = H(μ ‖ w1Encode(w1')).
    let c = polynomial_from_signed_bytes(c_bytes);
    if !vk.raw_verify_with_precomputed(&mu.into(), &c, &sig) { spoil_proof(); }
}
```

The `unwrap_or_spoil_proof` calls preserve the original example's
"malicious-prover-can't-fake-a-malformed-input" guarantee. Cycle markers
around each phase let us see the per-phase impact in §3.

### Build profiles

| Profile | Used for | Settings |
|---|---|---|
| `guest` (production) | `cargo run --release -p mldsa-verify` | `lto = "fat"`, `codegen-units = 1`, no debug info — smallest / fastest trace |
| `guest-profile` (analysis) | profiler binary + symbol-level work | `lto = "thin"`, `codegen-units = 16`, `debug = "line-tables-only"` — keeps function boundaries so PCs map to symbol names |

`JOLT_BACKTRACE=1` tells the `jolt` CLI not to add `-Cstrip=symbols`. Both
profiles coexist in `Cargo.toml`; only the `profile = "..."` attribute on
`#[jolt::provable]` selects which one is used.

---

## 3. Measurements

### 3.1 Headline numbers (with host-side precompute)

Measured on aarch64 (Apple Silicon), `--profile build-fast`. The guest is
always built with the `guest-profile` cargo profile (selected by the
`#[jolt::provable]` attribute; see §2 → "Build profiles"). Inputs are a real
ML-DSA-65 signature generated on the host with a fixed 32-byte seed.

| Metric | Value | vs prior (no precompute) |
|---|---:|---:|
| **Prove time** | **18.5 s** | −7.5 % |
| **Verify time** | **116 ms** | −10.8 % |
| **Proof size (serialized)** | **95.1 kB** | −1.3 % |
| Total cycles | 4 410 424 | **−14.1 %** |
| Real RV64IMAC instructions | 2 076 296 | −4.5 % |
| Virtual instructions (from expansion + inline) | 2 334 128 | −21.2 % |
| Padded trace length | 2²³ = 8 388 608 | unchanged (still rounds up to the same slot) |
| Effective throughput | ~239 kHz raw / ~453 kHz padded | |
| pk · msg · sig sizes | 1952 B · 23 B · 3309 B | unchanged |
| Extra public inputs (`A_hat`, `tr`, `μ`, `c`) | 30 720 + 64 + 64 + 256 = 31 104 B | new |

The cycle reduction lands almost exactly in the predicted band: we lift
~192 of the ~200 SHAKE permutations the verifier would otherwise run
out of the proof, leaving the ~8 perms of the final
`c̃' = H(μ ‖ w1Encode(w1'))` hash (which depends on the lattice
arithmetic and can't be precomputed).

> **Note on "real" vs "virtual" definitions.** The prover (§3.1) counts all
> rows belonging to a virtual sequence as virtual *including its first
> row*. The per-symbol profiler (§3.3) counts only the *non-first* rows
> of a virtual sequence as virtual. Only the real/virtual split differs.

### 3.2 Phase breakdown (top-level cycle markers)

Cycle markers wrap each top-level phase in the guest with
`start/end_cycle_tracking` and `core::hint::black_box`:

| Phase | Real | Virtual | Total | Share | What runs here |
|---|---:|---:|---:|---:|---|
| **phase1 — decode pk (precomputed `A_hat` + `tr`)** | 800 570 |   717 326 | 1 517 896 | 34.4 % | `pkDecode`, `decode_t1`, deserialize `A_hat` from bytes, `NTT(t₁·2¹³)`. **No SHAKE work.** |
| **phase2 — decode signature**                       | 107 211 |   117 598 |   224 809 |  5.1 % | `sigDecode`, bit-unpack `z`, decode hint, norm check `‖z‖∞ < γ₁−β` |
| **phase3 — verify with precomputed `μ` + `c`**      | 1 167 484 | 1 498 168 | 2 665 652 | 60.4 % | 6 NTTs + 6 iNTTs, matrix·vec mul, `UseHint`, `c̃' = H(μ‖w1Encode(w1'))` (~8 SHAKE256 perms), final compare. **No SampleInBall, no `μ` computation.** |

Sum of phases ≈ total ± marker overhead. Compared to the pre-precompute
numbers (2.23 M / 0.22 M / 2.68 M = 5.14 M total), phase 1 drops by
~32 % (entire `ExpandA` + `H(pk)` lifted out) and phase 3 stays roughly
flat (only `SampleInBall` + one absorb removed).

### 3.3 Per-function profile

The per-symbol profile from before precompute is preserved here as a
historical baseline; rerun [`mldsa-profile`](src/bin/profile.rs) to
get the post-precompute breakdown for your target. The expected
qualitative changes are:

- `ShakeState<Shake128>::squeeze` and `ShakeState<Shake128>::absorb`
  drop to ~0 cycles (the entire `ExpandA` work moved to the host).
- `ShakeState<Shake256>::absorb` shrinks to whatever absorb work the
  final `c̃' = H(μ ‖ w1Encode(w1'))` hash needs (~7 perms × absorb
  overhead).
- `ShakeState<Shake256>::squeeze` shrinks similarly to ~1 perm of
  squeeze.
- `sampling::sample_in_ball` and `sampling::rej_ntt_poly` drop to ~0
  (callers gone).
- All NTT / pointwise-mul / `Decompose` / `memcpy` buckets stay flat.

### 3.4 Where the cycles really go (grouped, post-precompute estimate)

| Bucket | Cycles (est.) | Share | Notes |
|---|---:|---:|---|
| **NTT-domain arithmetic** | ~2 186 000 | ~49.6 % | forward NTT + iNTT + NttVector pointwise mul (unchanged from baseline) |
| **Polynomial movement (memcpy/memset)** | ~931 000 | ~21.1 % | unchanged from baseline |
| **Keccak permutations (via inline)** | ~30 000 | ~0.7 % | only the final `c̃'` hash (~8 SHAKE256 perms) remains |
| **Hint reconstruction** | ~263 000 | ~6.0 % | `Decompose` for `UseHint` (unchanged) |
| **Array-iter helpers** | ~440 000 | ~10.0 % | `hybrid_array::Array` iterator bookkeeping (unchanged) |
| **A_hat deserialization** | ~150 000 | ~3.4 % | new: 30 720 LE-u32 reads into a fresh `NttMatrix` |
| **Vector arithmetic** | ~56 000 | ~1.3 % | `Vector::sub` (unchanged) |
| **Decoding / bit-packing** | ~80 000 | ~1.8 % | `byte_decode`, `Hint::bit_unpack`, `Signature::try_from` (unchanged) |
| **Boot / runtime / panic glue** | < 1 000 | < 0.1 % | unchanged |

The big change vs the original §3.4 is the Keccak bucket dropping from
14.2 % to ~0.7 %, almost exactly as predicted (192/200 perms lifted out).
The NTT and `memcpy` buckets are unchanged in absolute terms but rise
in percentage because the total is smaller.

### 3.5 Sanity checks

1. **Cycle reduction matches prediction.** Pre-precompute total: 5 136 355.
   Predicted savings from lifting all four publicly determined SHAKE
   steps out of the proof: ~14 % (the entire §3.4 Keccak bucket of
   728 684 cycles minus ~30 000 for the final `c̃'` hash that stays in,
   plus a small bonus from the ~426 k `rej_ntt_poly` loop being unused).
   Measured total: 4 410 424. **Actual savings: 14.1 %.** ✓
2. **Verify still passes.** End-to-end driver reports `proof valid: true`
   and `guest panicked: false`. The lift-out is observably correct, not
   just observably faster.
3. **Phase reconciliation.** Marker totals (1.52 M + 0.22 M + 2.67 M =
   4.41 M) match the prover-reported total within marker overhead.

---

## 4. The trust model (why precomputing public hashes on the host is sound)

The change above might look suspicious — the host computes values that
the proof depends on, then ships them in as inputs. Why doesn't this let
a malicious host forge?

The answer is that the **application-level verifier always recomputes
those values from `(pk, msg, sig)` itself before calling
`Jolt.verify_proof(...)`**. The Jolt proof attests to a specific
execution of the guest with specific public inputs; it does not attest
to the meaning of those inputs. So:

- **If the host supplies a wrong `A_hat`** (one that doesn't equal
  `ExpandA(rho)` for the `rho` baked into `pk`), then either:
  - The application's recomputation also produces that wrong `A_hat` —
    impossible, because `ExpandA` is deterministic.
  - The application produces the correct `A_hat`, sees that the input
    bound to the proof differs, and rejects the proof.

  Either way, no forgery.

- **Same argument for `tr`, `μ`, `c`.** They're all deterministic
  functions of public values. The verifier recomputes them and refuses
  to accept proofs whose inputs disagree.

The implementation lives in
[`examples/mldsa-verify/src/main.rs`](src/main.rs): on both the prove
and the verify side, the host runs:

```rust
let a_hat = expand_a::<MlDsa65::K, MlDsa65::L>(rho);
let tr    = compute_tr(&pk);
let mu    = compute_mu_with_context(&tr, ctx, &[&msg]);
let c     = sample_in_ball(signature.c_tilde(), MlDsa65::TAU);
```

and feeds the bytes into both `prove_mldsa_verify` and
`verify_mldsa_verify`. If you forget that on the verify side, you've
broken the security model. The `precomputed_verify_round_trip_*` tests in
[`crates/ml-dsa-jolt/src/lib.rs`](../../crates/ml-dsa-jolt/src/lib.rs)
exercise both μ shapes (internal and with-context) end-to-end.

---

## 5. Where to optimize next (data-driven)

The host-side precompute described in this README has already booked
the −14 % win it predicted. The remaining levers, in priority order:

| Effort | Estimated impact | Mechanism |
|---|---:|---|
| Upstream PR to `module-lattice`: in-place `Vector::ntt`/`ntt_inverse` and accumulator-style matrix-vector mul | **−10 to −15 %** total cycles | Eliminates most of the 18–21 % `memcpy` bucket |
| **Custom NTT inline** (new `INLINE_OPCODE` + sequence builder under `jolt-inlines/mldsa`) | **−25 to −30 %** total cycles | Collapses `Polynomial::ntt`, `ntt_inverse`, and pointwise mul (~50 % combined post-precompute) into precompile cycles |
| Rejection-sample precompile | **0 % now** | Was the third row before this change; the entire `rej_ntt_poly` hot path is now host-side and no longer in the trace. Listed for posterity. |
| Cache parsed `VerifyingKey` across multiple verifications | **−5 to −10 % per additional verify** | The pkDecode + `NTT(t1·2¹³)` step (phase 1 with precompute is ~34 % of trace; ExpandA's already gone). Application-level change. |

None of these are needed for the current example to work — it produces a
valid Jolt proof of an upstream ML-DSA-65 signature verification in
~18 s on a laptop today. They're listed as a roadmap, in priority order,
with numbers backing the priority.

---

## Appendix — How to reproduce

```bash
# Quick build (no fat LTO) — recommended for iteration.
cargo run --profile build-fast -p mldsa-verify

# Per-symbol profile (requires JOLT_BACKTRACE so symbols are kept).
JOLT_BACKTRACE=1 cargo run --profile build-fast -p mldsa-verify --bin mldsa-profile
```

For production prove-time numbers (fat LTO across the entire `jolt-core`
dep graph):

```bash
cargo run --release -p mldsa-verify
```

The release build takes ~10 minutes the first time; subsequent
incremental rebuilds are seconds. Note that the guest itself is always
built with the `guest-profile` cargo profile regardless of how the host
is compiled — that's set by the `profile = "guest-profile"` attribute
on `#[jolt::provable]`.

Both commands operate on a deterministic ML-DSA-65 keypair derived from a
fixed seed (`[u8; 32]` where `b[i] = i ^ 0xA5`). Swap in any other
`(pk, msg, sig)` triple by editing `examples/mldsa-verify/src/main.rs`.

Static instruction-mix analysis (no trace required):

```bash
python3 examples/mldsa-verify/analyze_elf.py
```

Reports `.text` size, per-opcode-family static counts, and a static count
of Keccak inline call sites.
