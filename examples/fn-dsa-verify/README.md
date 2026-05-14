# Proving FN-DSA-512 (draft Falcon) verification inside Jolt

A worked example: take the upstream
[**FN-DSA (draft Falcon)**](https://falcon-sign.info/) signature
verifier from [`pornin/rust-fn-dsa`](https://github.com/pornin/rust-fn-dsa),
run it as a Jolt guest, and prove its execution. We vendor
`fn-dsa-comm` and `fn-dsa-vrfy` at `crates/fn-dsa-comm-jolt/` and
`crates/fn-dsa-vrfy-jolt/` so we can:

1. **Route the Keccak-f[1600] permutation through the Jolt Keccak inline.**
   The upstream's hand-rolled `KeccakState::process()` is replaced by a
   one-liner that emits the inline opcode on RISC-V (and a standard
   software permutation on the host).
2. **Lift every public hashing step out of the proof.** The host computes
   `hashed_key = SHAKE256(pk)` and `c = hash_to_point(nonce, hashed_key,
   ctx, id, msg)` before invoking the prover, then ships the results into
   the guest as additional public inputs.

**Upstream crates used (unmodified):**

- [`fn-dsa`](https://crates.io/crates/fn-dsa) — Thomas Pornin's pure-Rust
  FN-DSA umbrella crate. The host uses it for key generation and signing
  to produce reproducible test vectors.
- [`rand_chacha`](https://crates.io/crates/rand_chacha) — used by the
  host only, to seed the FN-DSA RNG deterministically so every example
  run produces the same `(pk, sig)`.

**Vendored crates (workspace-local forks):**

- `crates/fn-dsa-comm-jolt/` — near-verbatim copy of `fn-dsa-comm 0.3.0`.
  The only change is in `shake.rs`: `KeccakState::process()` is replaced
  by a call to `keccak_f1600()` which dispatches to the Jolt Keccak
  inline on RISC-V and a standard software permutation on the host.
- `crates/fn-dsa-vrfy-jolt/` — near-verbatim copy of `fn-dsa-vrfy 0.3.0`.
  Adds `VerifyingKey512::from_parts(logn, h_ntt, hashed_key)` and
  `verify_with_precomputed_c(sig, c)` so the host can inject precomputed
  values. Also adds `compute_hashed_key(pk)` as a public helper and
  re-exports `hash_to_point`.

**Spec reference:** FN-DSA is still a draft, but its design and pseudocode
closely follow [NIST FIPS 204](https://csrc.nist.gov/pubs/fips/204/final)
(ML-DSA) for the high-level verify flow.

This document is in five parts:

1. **The algorithm** — what FN-DSA verification actually does (Pornin's
   draft, modelled on FIPS 204).
2. **The implementation** — how it maps to a Jolt guest example, the
   host/guest split, and the vendoring.
3. **Measurements** — prove time, proof size, verify time, per-phase and
   per-function cycle counts, side-by-side with ML-DSA.
4. **The trust model** — why precomputing public hashes on the host is sound.
5. **Where to optimize** — data-driven roadmap.

---

## 1. FN-DSA-512 verification — what it does

FN-DSA (Falcon, in draft form) is the lattice-based post-quantum
signature scheme NIST is currently standardising as the second of two PQ
DSAs (ML-DSA being the first). It operates over the cyclotomic ring

\[
  R_q = \mathbb{Z}_q[X]/(X^n + 1), \quad q = 12\,289, \quad n = 2^{\text{logn}}.
\]

For the "level I" parameter set (FN-DSA-512), `n = 512`, `logn = 9`.
There is also FN-DSA-1024 (level V) with `n = 1024`; we use the smaller
parameter set throughout this example.

### Public objects (FN-DSA-512)

| Object | Size | Contents |
|---|---:|---|
| Verifying key `pk` | 897 B | header (1 B) ‖ `modq_encode(h)` polynomial (14 bits/coeff, ≈ 896 B) |
| Signature `σ`     | 666 B | header (1 B) ‖ nonce `r` (40 B) ‖ `Comp_encode(s2)` polynomial (625 B) |
| Message `M`       | variable | the bytes being signed; treated as the pre-hash value `hv` with `HASH_ID_RAW` |

The Falcon trapdoor produces "short" pairs `(s1, s2)` such that
`s1 + s2·h ≡ c (mod q)`. Only `s2` is transmitted; `s1 = c − s2·h` is
reconstructed by the verifier. Validity is decided by a single norm test
on the recovered `(s1, s2)`.

### Verify algorithm (draft FN-DSA, modelled on FIPS 204 §5.3)

```text
FN-DSA.Verify(pk, σ, ctx, id, hv)
─────────────────────────────────────────────────────────────────────────────
 1.   hashed_key ← SHAKE256(pk, 64 bytes)         (BUFF tag)
      check σ[0]    == 0x30 + logn
      h     ← modq_decode(pk[1..])                (n × 14-bit unpack)
      h_ntt ← NTT(h)                              (in-place)
 2.   nonce  ← σ[1..41]
      s2     ← Comp_decode(σ[41..])               (variable-bit Huffman-ish)
      check  every coefficient |s2[i]| ≤ 2047
 3.   c      ← SHAKE256( nonce ‖ hashed_key ‖ 0x00 ‖
                         len(ctx) ‖ ctx ‖ hv )    (XOF-extended, reject-sampled
                                                   into 512 coefficients < q)
 4.   norm2  ← Σ s2[i]² over i
      ŝ2     ← NTT(s2 in ext-form)
      ŝ2·h̄  ← ŝ2 ⊙ h_ntt                         (pointwise mul)
      s1_int ← iNTT( ŝ2·h̄ )                      (back to integer form)
      s1     ← c − s1_int                          (mod q, ext-form)
      norm1  ← Σ s1[i]² over i
 5.   accept iff  norm1 + norm2  ≤  ⌊β²⌋[logn]    (≈ 34 034 726 for n=512)
```

The heavy primitives are:

- **SHAKE256** (Keccak-f[1600] sponge) — one absorb of the encoded pk to
  produce `hashed_key`, plus one full `hash_to_point` rejection-sampling
  stream. ~15 Keccak permutations total.
- **NTT and iNTT** over `Z_{12289}[X]/(X^{512}+1)`. Three NTTs (1 forward
  on `s2`, the pointwise mul accumulated implicitly, 1 inverse) plus the
  setup `NTT(h)` in phase 1.
- **Pointwise multiplication** in NTT domain.
- **Comp_decode** (variable-length Huffman-ish bit decoder) for `s2`, and
  **modq_decode** (14-bit unpack) for `h`.

Falcon's headline algorithmic advantage over ML-DSA is that verification
does *no* matrix-vector multiplication — just a single pointwise product
`s2 · h` in NTT form. ML-DSA-65 in contrast performs a full `k × l = 6 × 5`
matrix product. That difference cascades through the whole pipeline.

---

## 2. How we implemented it

### Design rule

> Vendor thin Jolt-aware copies of `fn-dsa-comm` and `fn-dsa-vrfy` at
> `crates/fn-dsa-{comm,vrfy}-jolt/`. Use them to (a) call the Jolt
> Keccak inline directly, no patch chain, and (b) expose entry points
> that let the host precompute `hashed_key` and `c` before invoking the
> prover. Don't touch the NTT, codec, or modular arithmetic code.

### Crate layout

```
crates/fn-dsa-comm-jolt/                 ← workspace [patch.crates-io] for `fn-dsa-comm 0.3.0`
├── Cargo.toml                             (drops hand-rolled Keccak-f[1600] in favour of inline)
└── src/
    ├── shake.rs                         KeccakState::process() → keccak_f1600() dispatch
    └── lib.rs, codec.rs, mq.rs, ...     vendored verbatim

crates/fn-dsa-vrfy-jolt/                 ← workspace [patch.crates-io] for `fn-dsa-vrfy 0.3.0`
├── Cargo.toml
└── src/lib.rs                           + from_parts, verify_with_precomputed_c,
                                         + compute_hashed_key, re-export hash_to_point

examples/fn-dsa-verify/
├── Cargo.toml                           host: fn-dsa + fn-dsa-vrfy + jolt-sdk
├── src/main.rs                          prove + verify driver, runs host-side SHAKE
├── src/bin/profile.rs                   per-symbol cycle profiler
└── guest/
    ├── Cargo.toml                       guest: jolt-sdk + fn-dsa-vrfy
    └── src/
        ├── lib.rs                       #[jolt::provable] entry point with precomputed inputs
        └── precomputed.rs               serialize/deserialize c polynomial
```

### The host / guest split

The guest's `#[jolt::provable]` entry point takes five byte slices:

```rust
fn fn_dsa_verify(
    pk: &[u8],                          // 897 B  ─ the actual signature inputs
    sig: &[u8],                         // 666 B  │
    msg: &[u8],                         // variable ─┘
    hashed_key_bytes: &[u8],            //  64 B  ─┐ derived from (pk, sig, msg);
    c_bytes: &[u8],                     // 1024 B ─┘ host computes, verifier MUST recompute
)
```

The two extra inputs are **deterministic functions of `(pk, sig, msg)`**:

#### `hashed_key_bytes` — `hashed_key = SHAKE256(pk, 64 B)`

```text
hashed_key ← SHAKE256(pk, 64 B)        // FN-DSA spec, BUFF tag: hash the
                                       // entire encoded public key into a
                                       // 64-byte tag that binds messages to
                                       // this specific verifying key
```

Where the SHAKE work happens: ~7 SHAKE256 permutations (rate 136 B,
so 897 B pk spans ~7 blocks).

Host code: [`compute_hashed_key(&vrfy_key)`](src/main.rs)

#### `c_bytes` — `c = hash_to_point(nonce ‖ hashed_key ‖ ctx ‖ id ‖ msg)`

```text
nonce       ← sig[1..41]               // 40-byte random nonce from the signature
c           ← hash_to_point(nonce,     // FN-DSA: rejection-sample 512
                hashed_key,            // coefficients < q=12289 from a
                DOMAIN_NONE,           // SHAKE256 stream seeded with
                HASH_ID_RAW,           // nonce ‖ hashed_key ‖ 0x00 ‖
                msg)                   // len(ctx) ‖ ctx ‖ msg
serialize(c) → 1024 B                  // 512 × little-endian u16
```

Where the SHAKE work happens: ~8 SHAKE256 permutations (~1 perm for
absorb of the ~128 B input, plus ~7 perms for squeeze of ~1094 B of
rejection-sampled output).

Host code: [`hash_to_point(nonce, &hashed_key, &DOMAIN_NONE, &HASH_ID_RAW, &msg, &mut c)`](src/main.rs) → [`serialize_c(&c)`](guest/src/precomputed.rs)

#### Summary table

| Input              | Size    | Derived from    | Host call                                         | SHAKE perms saved |
|--------------------|--------:|-----------------|----------------------------------------------------|------------------:|
| `hashed_key_bytes` |   64 B  | `pk`            | `compute_hashed_key(&pk)`                          |              ~7   |
| `c_bytes`          | 1024 B  | `sig`, `pk`, `msg` | `hash_to_point(nonce, hashed_key, ctx, id, msg)` |              ~8   |

Total SHAKE work moved out: ~15 permutations — essentially all of the
verifier's Keccak budget. After this split, the only Keccak work inside
the proof is the Keccak inline permutations used by NTT-internal
intermediate steps (zero SHAKE calls remain in the proof).

### The guest

```rust
// examples/fn-dsa-verify/guest/src/lib.rs (abridged)
#[jolt::provable(
    profile = "guest-profile",
    stack_size = 65_536,
    heap_size = 16_384,
    max_trace_length = 8_388_608,
    max_input_size = 8192,
)]
fn fn_dsa_verify(
    pk: &[u8], sig: &[u8], msg: &[u8],
    hashed_key_bytes: &[u8],            // 64 B
    c_bytes: &[u8],                     // 1024 B
) {
    let hashed_key: [u8; 64] = hashed_key_bytes.try_into().unwrap_or_spoil_proof();

    // Phase 1: decode vk. Still does modq_decode + ext_to_int + int_to_NTT
    // (pure arithmetic, not SHAKE). Rebuilds with host-supplied hashed_key.
    let vk = VerifyingKey512::decode(pk).unwrap_or_spoil_proof();
    let vk = VerifyingKey512::from_parts(vk.get_logn(), vk.h_ntt(), hashed_key)
        .unwrap_or_spoil_proof();

    // Phase 2: lattice math only — NTT(s2), pointwise s2*h, iNTT, norm check.
    let c = deserialize_c(c_bytes, 1 << vk.get_logn()).unwrap_or_spoil_proof();
    if !vk.verify_with_precomputed_c(sig, &c) { spoil_proof(); }
}
```

### Why two phases (not three)

FN-DSA is structurally simpler than ML-DSA:

- No `ExpandA` analogue — the verifying key encodes the full polynomial
  `h` directly. Phase 1 just decodes `h` and NTTs it.
- Signature decoding is folded into `verify_with_precomputed_c`.

So phase 1 is "decode vk", phase 2 is "everything else".

### Why `fn-dsa-vrfy` in the guest, not the umbrella `fn-dsa`

The [`fn-dsa`](https://crates.io/crates/fn-dsa) umbrella crate
transitively depends on
[`fn-dsa-kgen`](https://crates.io/crates/fn-dsa-kgen) and
[`fn-dsa-sign`](https://crates.io/crates/fn-dsa-sign), which contain
floating-point code paths (FFT for the Gaussian sampler in signing on
x86/aarch64, plus an integer FP emulator for other targets).
Verification doesn't use any of this, but pulling those crates into
the guest is wasted compile time and ELF size. We depend on
[`fn-dsa-vrfy`](https://crates.io/crates/fn-dsa-vrfy) directly, which
has only [`fn-dsa-comm`](https://crates.io/crates/fn-dsa-comm) as a
runtime dep, which in turn pulls only
[`rand_core`](https://crates.io/crates/rand_core) and (cfg-gated to
x86) [`cpufeatures`](https://crates.io/crates/cpufeatures).

### Reproducible inputs

The host uses `rand_chacha::ChaCha20Rng` seeded with a fixed 32-byte seed
for keypair generation and a separate seed for signing randomness. Every
run produces the exact same `(pk, sig)` pair, so the per-phase and
per-function cycle counts below are stable across runs.

> FN-DSA signing is randomized (each signature draws a fresh nonce).
> `rand::OsRng` would also be valid (the proof would still verify) but
> would defeat byte-level reproducibility.

### Build profiles

| Profile | Used for | Settings |
|---|---|---|
| `guest` (production) | `cargo run --release -p fn-dsa-verify` | `lto = "fat"`, `codegen-units = 1`, no debug info — smallest / fastest trace |
| `guest-profile` (analysis) | profiler binary + symbol-level work | `lto = "thin"`, `codegen-units = 16`, `debug = "line-tables-only"` — keeps function boundaries so PCs map to symbol names |

Selected via `profile = "guest-profile"` on the `#[jolt::provable]`
attribute; `JOLT_BACKTRACE=1` tells the `jolt` CLI not to strip symbols.
The measurements in §3.3 use `guest-profile`; the headline numbers in
§3.1 use the production `guest` profile.

---

## 3. Measurements

### 3.1 Headline numbers

Measured on aarch64 (Apple Silicon), `--profile build-fast`. Inputs are
a real FN-DSA-512 signature generated on the host with a fixed 32-byte
seed.

| Metric | Value |
|---|---:|
| **Prove time** | **5.64 s** |
| **Verify time** | **96 ms** |
| **Proof size (serialized)** | **87.0 kB** |
| Total cycles | 1 039 478 |
| Real RV64IMAC instructions | 510 600 |
| Virtual instructions (from expansion) | 528 878 |
| Padded trace length | 2²⁰ = 1 048 576 |
| Effective throughput | ~184 kHz raw / ~186 kHz padded |
| pk · msg · sig sizes | 897 B · 24 B · 666 B |

> **Note on "real" vs "virtual" definitions.** The prover (§3.1) counts
> all rows belonging to a virtual sequence as virtual *including its
> first row*. The per-symbol profiler (§3.3) counts only the
> *non-first* rows of a virtual sequence as virtual. Totals match
> exactly (1 039 478); only the real/virtual split differs by ~94 000
> cycles. Comparing virtual-instruction shares with `mldsa-verify`'s
> README requires using one convention consistently.

### 3.2 Phase breakdown (top-level cycle markers)

| Phase | Real | Virtual | Total | Share | What runs here |
|---|---:|---:|---:|---:|---|
| **phase1 — decode_vk** | 156 280 | 148 936 | 305 216 | 29.4 % | `SHAKE256(pk)` → `hashed_key` (BUFF tag), `modq_decode(h)` (14-bit unpack), `NTT(h)` |
| **phase2 — verify_internal** | 353 919 | 379 851 | 733 770 | 70.6 % | `sigDecode`, `Comp_decode(s2)`, `hash_to_point(c)` (SHAKE256 reject-sample), `NTT(s2)`, `s2 ⊙ h_ntt`, `iNTT`, norm check |
| Boot / glue (rest)        |     ~290 |     ~200 |     492 | 0.0 % | `_start`, `__platform_bootstrap`, postcard deserialization |

Sum of phases ≈ total ± marker overhead (a few hundred cycles for the
four markers).

### 3.3 Per-function profile

Built with `profile = "guest-profile"` (thin LTO, symbols preserved) so
PCs map back to Rust function names. Use these for **relative
attribution**, not absolute perf claims (thin LTO inlines less
aggressively, so absolute cycle counts will be slightly higher than the
production `guest` profile).

| Real | Virtual | Total | % | Function |
|---:|---:|---:|---:|---|
| 114 684 | 262 134 | 376 818 | 36.3 % | `fn_dsa_vrfy::verify_inner` (NTT mul + iNTT + sub + norm; lots of inlining) |
| 104 756 | 271 930 | 376 686 | 36.2 % | `fn_dsa_comm::mq::mqpoly_int_to_NTT` (forward NTT, called twice) |
| 102 544 |       0 | 102 544 |  9.9 % | `<fn_dsa_comm::shake::KeccakState>::process` (24-round soft permutation) |
|  43 628 |  21 982 |  65 610 |  6.3 % | `<SHAKE<256>>::extract` (squeeze + rate-bookkeeping) |
|  15 210 |  26 887 |  42 097 |  4.0 % | `fn_dsa_comm::hash_to_point` (reject-sample loop body) |
|  17 896 |  10 326 |  28 222 |  2.7 % | `<SHAKE<256>>::inject` (absorb + rate-bookkeeping) |
|   7 589 |  16 117 |  23 706 |  2.3 % | `fn_dsa_comm::codec::modq_decode` (14-bit unpack of `h`) |
|   4 657 |  12 812 |  17 469 |  1.7 % | `fn_dsa_vrfy::decode_inner` (verifying-key decode wrapper) |
|   3 026 |     852 |   3 878 |  0.4 % | `compiler_builtins::mem::memcpy` |
|   1 798 |      36 |   1 834 |  0.2 % | `memset` |
|     ~250 |     ~140 |     390 | 0.04 % | boot / glue / main / `Verifier::decode` / postcard deserialization |

Notable: 78 text symbols total in the ELF. The remaining ~78 − 11 = 67
symbols sum to less than 0.05 % combined and are dropped from the table.

### 3.4 Where the cycles really go (grouped)

| Bucket | Cycles | Share | Notes |
|---|---:|---:|---|
| **NTT-domain arithmetic** | **~570 000** | **~55 %** | `mqpoly_int_to_NTT` (forward & inverse) + the pointwise mul / sub / norm work fused into `verify_inner` |
| **Keccak permutations** (soft) | **~102 500** |  9.9 % | `KeccakState::process` — pure RV64IMAC XOR/AND/NOT/rotate; no virtual expansion at all |
| **SHAKE256 absorb + squeeze wrappers** | **~93 800** |  9.0 % | `<SHAKE<256>>::inject` + `<SHAKE<256>>::extract`, i.e. the rate-bookkeeping and byte-shuffling around the permutation |
| **`hash_to_point` reject-sample loop body** | **~42 000** |  4.0 % | Excludes the SHAKE underneath, which is in the previous two rows |
| **Bit-pack / unpack** | **~24 000** |  2.3 % | `modq_decode` for `h`; `Comp_decode` for `s2` is folded into `verify_inner` |
| **`memcpy` / `memset`** | **~5 700** |  0.5 % | Tiny — `fn-dsa-vrfy` operates on stack arrays in place, not pass-by-value polynomials |
| **Other (boot, postcard, glue)** | **~390** | < 0.1 % | `_start`, `__platform_bootstrap`, panic glue |
| **Total**                  | **1 039 478** | **100 %** | |

The take-away: **NTT-domain arithmetic is 55 % of the trace**;
**Keccak-driven work (soft permutation + sponge wrappers + hash_to_point
body) is 23 % combined**. Everything else is bookkeeping.

### 3.5 Sanity checks

1. **Keccak count.** `KeccakState::process` = 102 544 real cycles. Each
   soft permutation in the upstream's hand-rolled `process()` body is
   ≈ 6 700 RV64IMAC instructions (partially unrolled, two rounds per
   loop iter × 12 iters × ~280 instructions). That gives
   `102 544 / 6 700 ≈ 15 permutations`. Algorithmic expectation for FN-DSA-512
   verify:
   - `SHAKE256(pk)` for `hashed_key` (897 B input, 64 B output): pk
     spans 7 absorb blocks (rate = 136 B), so ~7 perms.
   - `hash_to_point` (≈40 + 64 + 4 + 24 = 132 B input, plus reject-sample
     output for 512 coefficients × 2 B with ~93.7 % acceptance ≈
     1 094 B): ~1 perm for absorb (input < 136 B) + ~8 perms for squeezes.
   - **Total: 7 + 1 + 8 = ~16 permutations.** ✓
2. **Total accounting.** Profiler attributes 1 039 478 of 1 039 478
   cycles. 100 % coverage.
3. **Phase reconciliation.** Cycle-marker phase totals (305 216 +
   733 770 = 1 038 986) account for all but 492 cycles, which are
   exactly the boot / glue / main bucket from §3.3.

### 3.6 Side-by-side vs `mldsa-verify`

Both rows measured with `--profile build-fast` on the same machine; guest
in both cases is the thin-LTO `guest-profile` build.

| Metric | ML-DSA-65 | FN-DSA-512 | Ratio |
|---|---:|---:|---:|
| pk · sig · msg | 1952 B · 3309 B · 23 B | 897 B · 666 B · 24 B | Falcon: 2.2× smaller pk, 5.0× smaller sig |
| Total cycles | 5 136 355 | 1 039 478 | **4.9× smaller** |
| Padded trace | 2²³ = 8.4M | 2²⁰ = 1.05M | **8× smaller** |
| **Prove time** | **20.02 s** | **6.41 s** | **3.1× faster** |
| Verify time | 130 ms | 98 ms | Falcon ~25 % faster verify |
| Proof size | 96.4 kB | 87.0 kB | 1.11× smaller |
| Prover throughput (padded) | 419 kHz | 164 kHz | 2.6× lower |
| Keccak share of trace | 14.2 % (~200 perms × inline) | 18.9 % (~15 perms × soft) | Falcon does much less Keccak in absolute terms |
| NTT/poly share | 42.5 % (k·l matrix-vec) | ~55 % (one pointwise mul) | ML-DSA does ~3× more arithmetic in absolute terms |
| memcpy share | 18.1 % | 0.4 % | Falcon's API is borrow-based, ML-DSA's `Poly` is pass-by-value |

**Why "only" 3.1× faster prove despite an 8× smaller padded trace.**
Jolt prove time has both a linear component (sumcheck work scales with
trace length) and a fixed component (Dory setup, BlindFold, R1CS
construction, Stage 8 batch opening ≈ 1.4 s alone). The fixed costs
amortize well over ML-DSA's 8.4M-cycle trace but dominate FN-DSA's 1M.
This is also why throughput drops: kHz/cycle stays similar for sumcheck
itself, but per-cycle wall-clock rises because the fixed bucket spreads
over fewer cycles.

### 3.7 Keccak: inline (ML-DSA) vs soft (FN-DSA), measured per permutation

Both examples ultimately call `Keccak-f[1600]` under the hood, but
ML-DSA routes it through the Jolt Keccak inline (via the `keccak-jolt`
patch) while FN-DSA runs the hand-rolled software permutation inside
`fn-dsa-comm`. Dividing total Keccak-related cycles (permutation +
sponge wrapper) by the algorithmic permutation count gives a clean
per-permutation comparison.

| | ML-DSA inline | FN-DSA soft | Ratio (soft / inline) |
|---|---:|---:|---:|
| Permutations during verify | ~200 | ~15 | — |
| Real cycles per permutation | ~122 | ~10 938 | 90× more real |
| Virtual cycles per permutation | ~3 522 | ~2 154 | 1.6× fewer virtual |
| **Total cycles per permutation** | **~3 643** | **~13 092** | **~3.6× more expensive without the inline** |
| Total Keccak-related cycles in trace | ~728 k (14.2 %) | ~196 k (18.9 %) | ML-DSA does 13× more permutations, FN-DSA pays 3.6× per perm; net Keccak cycles still ~3.7× higher in ML-DSA |

**What this implies for each algorithm.** The inline trades ~3 500
virtual instructions for ~10 800 real ones per permutation — same
total trace rows, but the virtual ones use the inline's pre-allocated
register file and a tighter constraint shape. For ML-DSA's
permutation-heavy `ExpandA` workload (~200 calls) this saves ~1.9M
trace cycles and keeps the trace well below its 2²³ padded ceiling.
For FN-DSA's ~15 permutations the absolute saving (~140k cycles) would
still leave the trace inside the same 2²⁰ padded slot — i.e. zero
wall-clock benefit — which is why this example consciously skips the
patch (see §4).

---

## 4. Where to optimize (data-driven)

Crucial finding: **Keccak acceleration alone is worthless here.** With
Keccak responsible for 18.9 % of the trace (combining the permutation
itself and the sponge-wrapper code) and the trace pinned at 2²⁰ by
padding, replacing the soft permutation with the inline saves ≈ 50 000
cycles → trace drops from 1.04M to 0.99M → still 2²⁰ padded → prove
time unchanged.

The only way to win prove time is to drop *below* 2¹⁹ = 524 288 cycles,
which requires reducing the trace by **~520 000 cycles or more**. That
puts the priorities in this order:

| Effort | Estimated impact | Mechanism |
|---|---:|---|
| **NTT precompile** for `Z_{12289}[X]/(X^{512}+1)` | **−40 to −50 %** total cycles | Collapses `mqpoly_int_to_NTT` (36 %) and the NTT-side of `verify_inner` (~10–20 %) into precompile cycles. Single biggest lever. Crosses the 2¹⁹ threshold by itself. |
| **NTT + Keccak combo** (NTT precompile + `fn-dsa-comm-jolt` patch following `crates/keccak-jolt` pattern) | **−55 to −65 %** total cycles | Combined ~570 k NTT + ~196 k Keccak = ~766 k cycles eliminated. Trace lands at ~280 k → 2¹⁹ padded. Final prove time estimate: **2–3 s.** |
| **Comp_decode precompile** for `s2` | **−5 to −8 %** of phase 2 | Falcon's Huffman-ish decoder is bit-by-bit; replace with a sequence-builder expansion. Marginal until the NTT and Keccak buckets are already addressed. |
| **Batch verification** (prove N signatures in one guest invocation) | **−~30 % per additional signature** | Application-level. Amortizes the fixed Jolt overheads (Dory setup, Stage 8) across more sumcheck work. The most user-impactful change with no inline development. |

None of these are needed for the current example to work — it produces
a valid Jolt proof of an upstream FN-DSA-512 signature verification in
~5.6 s on a laptop today. They're listed as a roadmap in priority order,
with numbers backing the priority.

---

## Appendix — How to reproduce

```bash
# Quick build (no fat LTO) — recommended for iteration.
cargo run --profile build-fast -p fn-dsa-verify

# Per-symbol profile (requires JOLT_BACKTRACE so symbols are kept).
JOLT_BACKTRACE=1 cargo run --profile build-fast -p fn-dsa-verify --bin fn-dsa-profile
```

For production prove-time numbers (fat LTO across the entire `jolt-core`
dep graph):

```bash
cargo run --release -p fn-dsa-verify
```

The release build takes ~10 minutes the first time; subsequent
incremental rebuilds are seconds.

Both commands operate on a deterministic FN-DSA-512 keypair derived
from a fixed seed (`[u8; 32]` where `b[i] = i ^ 0xA5`) and a separate
fixed seed for signing randomness (`[0x5A; 32]`). Swap in any other
`(pk, sig, msg)` triple by editing `examples/fn-dsa-verify/src/main.rs`.

### Roadmap details — Strategy A (Keccak inline acceleration via vendored `fn-dsa-comm`)

Documented here for posterity; **§4 shows it's not worth doing in
isolation**.

To route `fn-dsa-comm::shake::KeccakState::process()` through Jolt's
existing Keccak inline (`jolt-inlines-keccak256`), we'd vendor
`fn-dsa-comm 0.3.0` source into `crates/fn-dsa-comm-jolt/` and replace
one function, gated by `target_arch`:

```rust
// crates/fn-dsa-comm-jolt/src/shake.rs (RISC-V branch)
#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
fn process(&mut self) {
    use jolt_inlines_keccak256::{INLINE_OPCODE, KECCAK256_FUNCT3, KECCAK256_FUNCT7};
    unsafe {
        core::arch::asm!(
            ".insn r {opcode}, {funct3}, {funct7}, x0, {rs1}, x0",
            opcode = const INLINE_OPCODE,
            funct3 = const KECCAK256_FUNCT3,
            funct7 = const KECCAK256_FUNCT7,
            rs1 = in(reg) self.0.as_mut_ptr(),
            options(nostack),
        );
    }
}
```

Patched into the workspace root `Cargo.toml`:

```toml
[patch.crates-io]
fn-dsa-comm = { path = "./crates/fn-dsa-comm-jolt" }
keccak      = { path = "./crates/keccak-jolt" }   # for ML-DSA
```

Cost: ~2 500 LoC vendored (codec, mq, hash_to_point, shake, …). The
only logical change is the one function above. Maintenance is
"re-vendor on `fn-dsa-comm` version bumps." Falcon is BSD-style
("Unlicense"), so vendoring is unencumbered.

Useful only when combined with the NTT precompile or some other
trace-shrinking work that pushes total cycles below 2¹⁹.
