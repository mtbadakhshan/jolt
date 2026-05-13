# Proving ML-DSA-65 verification inside Jolt

A worked example: take the unmodified upstream
[`ml-dsa`](https://github.com/RustCrypto/signatures/tree/master/ml-dsa)
crate from RustCrypto — an implementation of **ML-DSA-65** per
[NIST FIPS 204](https://csrc.nist.gov/pubs/fips/204/final) — run it as
a Jolt guest, and prove its execution. The Keccak-heavy bits are
accelerated by Jolt's existing Keccak inline via a tiny local `keccak`
crate patched into the workspace.

**Upstream crates used (unmodified):**

- [`ml-dsa`](https://crates.io/crates/ml-dsa) — the FIPS 204 verifier
  itself, which pulls in
  [`module-lattice`](https://crates.io/crates/module-lattice) (NTT and
  ring arithmetic),
  [`hybrid-array`](https://github.com/RustCrypto/utils/tree/master/hybrid-array)
  (type-level-sized arrays), and
  [`signature`](https://github.com/RustCrypto/traits/tree/master/signature)
  (the `Signer` / `Verifier` traits).
- [`sha3`](https://github.com/RustCrypto/hashes/tree/master/sha3) — the
  **SHAKE128/256 sponge**: rate accounting, padding, domain-separator
  bytes, the absorb/squeeze state machine. This crate's code runs
  unmodified in the guest; it's the layer *above* the Keccak
  permutation.

**Patched crate (workspace-local replacement):**

- [`keccak 0.2.x`](https://github.com/RustCrypto/sponges/tree/master/keccak)
  — only the inner **Keccak-f[1600] permutation** that `sha3`'s sponge
  invokes. Our `crates/keccak-jolt/` slot-in (via `[patch.crates-io]`)
  exposes the same crate-level API as upstream and forwards
  `Keccak::with_f1600` to the Jolt Keccak inline opcode on RISC-V (and
  to a verbatim copy of the upstream software permutation on the host).

So the layering at runtime in the guest is:

```
ml-dsa  →  sha3 (sponge)  →  keccak::Keccak::with_f1600  →  Jolt Keccak inline
   ▲           ▲                       ▲                          ▲
   │           │                       │                          │
unmodified  unmodified            patched here                routes to our
upstream    upstream             (crates/keccak-jolt/)          inline opcode
```

This document is in three parts:

1. **The algorithm** — what ML-DSA verification actually does (FIPS 204 §5.3).
2. **The implementation** — how it maps to a Jolt guest example and what the
   `crates/keccak-jolt` patch does.
3. **Measurements** — prove time, proof size, verify time, per-phase and
   per-function cycle counts, and where the budget actually goes.

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

> Use the audited upstream library; route only the lowest-level primitive
> (Keccak-f[1600]) through Jolt's existing inline. Don't fork crypto.

### Crate layout

```
crates/keccak-jolt/                   ← workspace patch for `keccak 0.2.x`
├── Cargo.toml                          name = "keccak", version = "0.2.0"
└── src/lib.rs                          Keccak::with_f1600 → inline on RISC-V
                                                          → soft Keccak on host

examples/mldsa-verify/
├── Cargo.toml                         host: ml-dsa + rand + jolt-sdk
├── src/main.rs                        prove + verify driver
├── src/bin/profile.rs                 per-symbol cycle profiler
└── guest/
    ├── Cargo.toml                     guest: jolt-sdk + ml-dsa
    └── src/lib.rs                     12-line #[jolt::provable] entry point
```

### The patch — what `crates/keccak-jolt` does

[`ml-dsa 0.1.0-rc.11`](https://crates.io/crates/ml-dsa/0.1.0-rc.11) →
[`sha3 0.11`](https://github.com/RustCrypto/hashes/tree/master/sha3) →
[`keccak 0.2.x`](https://github.com/RustCrypto/sponges/tree/master/keccak)
→ `keccak::f1600(state)`.

We slot a workspace-local crate into the `keccak 0.2.x` version slot via
the root workspace's `[patch.crates-io]`:

```toml
# Cargo.toml (workspace root)
[patch.crates-io]
keccak = { path = "./crates/keccak-jolt" }
```

That crate exposes only the `keccak` API that `sha3 0.11` consumes
(`Keccak::with_f1600` + `State1600`). On RISC-V the `f1600` function emits
the custom Keccak inline opcode directly:

```rust
// crates/keccak-jolt/src/lib.rs (RISC-V branch)
unsafe {
    core::arch::asm!(
        ".insn r {opcode}, {funct3}, {funct7}, x0, {rs1}, x0",
        opcode  = const INLINE_OPCODE,        // 0x0B
        funct3  = const KECCAK256_FUNCT3,     // 0
        funct7  = const KECCAK256_FUNCT7,     // 1
        rs1     = in(reg) state.as_mut_ptr(),
        options(nostack),
    );
}
```

On the host target it falls back to a verbatim copy of the upstream
software permutation. The patch affects only consumers of `keccak 0.2.x`;
older `keccak 0.1.x` (used by other parts of the workspace) is untouched.

**Zero modifications to existing code.** The only edits outside our new
files are seven lines in the workspace `Cargo.toml` (two workspace members
and one `[patch.crates-io]` entry).

### The guest

```rust
// examples/mldsa-verify/guest/src/lib.rs
#[jolt::provable(
    profile = "guest-profile",
    stack_size = 262_144,
    heap_size = 4_194_304,
    max_trace_length = 16_777_216,
    max_input_size = 8192,
)]
fn mldsa_verify(pk: &[u8], msg: &[u8], sig: &[u8]) {
    let vk  = VerifyingKey::<MlDsa65>::new(<&EncodedVerifyingKey<MlDsa65>>::try_from(pk).unwrap_or_spoil_proof());
    let sig = Signature::<MlDsa65>::try_from(sig).unwrap_or_spoil_proof();
    vk.verify(msg, &sig).unwrap_or_spoil_proof();
}
```

That's it. Cycle markers around each of the three phases give us the
top-level breakdown without touching `ml-dsa`'s internals.

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

### 3.1 Headline numbers

Measured on aarch64 (Apple Silicon), `--profile build-fast`. The guest is
always built with the `guest-profile` cargo profile (selected by the
`#[jolt::provable]` attribute; see §2 → "Build profiles"). Inputs are a real
ML-DSA-65 signature generated on the host with a fixed 32-byte seed.

| Metric | Value |
|---|---:|
| **Prove time** | **20.02 s** |
| **Verify time** | **130 ms** |
| **Proof size (serialized)** | **96.4 kB** |
| Total cycles | 5 136 355 |
| Real RV64IMAC instructions | 2 172 988 |
| Virtual instructions (from expansion + inline) | 2 963 367 |
| Padded trace length | 2²³ = 8 388 608 |
| Effective throughput | ~256 kHz raw / ~419 kHz padded |
| pk · msg · sig sizes | 1952 B · 23 B · 3309 B |

> **Note on "real" vs "virtual" definitions.** The prover (§3.1) counts all
> rows belonging to a virtual sequence as virtual *including its first
> row*. The per-symbol profiler (§3.3) counts only the *non-first* rows
> of a virtual sequence as virtual. Totals match exactly (5 136 355);
> only the real/virtual split differs by ~366 k cycles.

### 3.2 Phase breakdown (top-level cycle markers)

Cycle markers wrap each top-level phase in the guest with
`start/end_cycle_tracking` and `core::hint::black_box`:

| Phase | Real | Virtual | Total | Share | What runs here |
|---|---:|---:|---:|---:|---|
| **phase1 — decode pk + ExpandA** |   897 513 | 1 333 461 | 2 230 974 | 43.4 % | `pkDecode`, `ExpandA` (~150–200 SHAKE128 perms), `tr = H(pk)`, `NTT(t₁·2¹³)` |
| **phase2 — decode signature**     |   107 210 |   117 598 |   224 808 |  4.4 % | `sigDecode`, bit-unpack `z`, decode hint, norm check `‖z‖∞ < γ₁−β` |
| **phase3 — verify_internal**      | 1 167 866 | 1 512 217 | 2 680 083 | 52.2 % | `μ = H(tr‖msg)`, `SampleInBall`, 6 NTTs + 6 iNTTs, matrix·vec mul, `UseHint`, `c̃' = H(μ‖w1Encode(w1))`, final compare |

Sum of phases ≈ total ± marker overhead (≈ 490 cycles for the 6 markers).

### 3.3 Per-function profile

Built with `profile = "guest-profile"` (thin LTO, symbols preserved) so
PCs map back to Rust function names. The headline §3.1 numbers come from
the same guest build, so per-symbol cycle counts here add up exactly to
the same 5 136 355 total — see the "real vs virtual" note under §3.1
for why the split looks slightly different from the prover's log line.

Top buckets (cycles = real + virtual):

| Real | Virtual | Total | % | Function |
|---:|---:|---:|---:|---|
| 270 432 | 677 184 | 947 616 | 18.4 % | `Polynomial::ntt` (forward NTT) |
| 742 363 | 145 350 | 887 713 | 17.3 % | `compiler_builtins::mem::memcpy` |
| 170 268 | 522 240 | 692 508 | 13.5 % | `&NttVector * &NttVector` (pointwise mul + accumulation) |
|  7 410 | 588 840 | 596 250 | 11.6 % | `ShakeState<Shake128>::squeeze` |
| 171 816 | 374 304 | 546 120 | 10.6 % | `NttPolynomial::ntt_inverse` |
| 125 387 | 300 858 | 426 245 |  8.3 % | `sampling::rej_ntt_poly` (rejection-sample loop) |
| 112 128 | 150 528 | 262 656 |  5.1 % | `Elem::decompose` (HighBits / LowBits) |
|  40 171 | 115 200 | 155 371 |  3.0 % | `Array::ntt` (vector-of-poly NTT wrapper) |
|   8 519 |  86 061 |  94 580 |  1.8 % | `ShakeState<Shake256>::absorb` |
|  23 404 |  53 760 |  77 164 |  1.5 % | NttPoly · NttPoly helper |
|  14 193 |  50 688 |  64 881 |  1.3 % | NttPoly · NttPoly helper |
|  21 721 |  39 945 |  61 666 |  1.2 % | Array (Poly utility iter) |
|  11 613 |  44 780 |  56 393 |  1.1 % | `Vector::sub` (Aẑ − ĉt₁) |
|   9 075 |  44 800 |  53 875 |  1.0 % | `byte_decode` (bit unpack) |
|  40 052 |   3 676 |  43 728 |  0.9 % | `memset` |
|   3 926 |  27 538 |  31 464 |  0.6 % | `ShakeState<Shake256>::squeeze` |
|   4 500 |   1 890 |   6 390 |  0.1 % | `ShakeState<Shake128>::absorb` |
|   1 119 |   2 810 |   3 929 |  0.1 % | `sampling::sample_in_ball` |

The remaining ~80 symbols (signature parsing, hint bit-unpack, boot/panic
glue, runtime) sum to under 1 % combined.

### 3.4 Where the cycles really go (grouped)

| Bucket | Cycles | Share | Notes |
|---|---:|---:|---|
| **NTT-domain arithmetic** | **2 186 244** | **42.5 %** | forward NTT + iNTT + NttVector pointwise mul |
| **Polynomial movement (memcpy/memset)** | **931 441** | **18.1 %** | `Poly` = 1 KiB; `Vector` = L · 1 KiB; the API returns these by value |
| **Keccak permutations (via inline)** | **728 684** | **14.2 %** | Shake128 absorb (~6 k) + squeeze (~596 k, ~174 perms) + Shake256 absorb (~95 k) + squeeze (~31 k) |
| **Rejection-sample loop** | **426 245** | **8.3 %** | `rej_ntt_poly` body, independent of Keccak |
| **Hint reconstruction** | **262 656** | **5.1 %** | `Decompose` for `UseHint` |
| **Array-iter helpers** | **~440 000** | **~8.6 %** | `hybrid_array::Array` iterator bookkeeping |
| **Vector arithmetic** | **~56 000** | **1.1 %** | `Vector::sub` |
| **Decoding / bit-packing** | **~80 000** | **1.6 %** | `byte_decode`, `Hint::bit_unpack`, `Signature::try_from` |
| **Boot / runtime / panic glue** | **< 1 000** | **< 0.1 %** | `__platform_bootstrap`, `_start`, etc. |

### 3.5 Sanity checks

1. **Keccak count.**
   `ShakeState<Shake128>::squeeze = 596 250 cycles`. Each Keccak-f
   permutation expands to ≈ 3 434 virtual instructions in the trace, giving
   `596 250 / 3 434 ≈ 174 permutations`. The algorithmic estimate for
   `ExpandA` + `tr = H(pk)` is 165 – 225 SHAKE128 permutations. ✓
2. **Total accounting.** Profiler attributes 5 136 352 of 5 136 355 cycles
   (3 leftover in two micro-buckets). Essentially complete coverage.
3. **Phase reconciliation.** The cycle-marker phase totals (2.23 M / 0.22 M /
   2.68 M) sum to 5.14 M; the remaining ~490 cycles are the marker probes
   themselves plus boot/glue captured in sub-1% functions. Allocating
   per-function costs across phases (e.g. half of `Polynomial::ntt` to
   phase 1 for `t1·2ᵈ`, half to phase 3 for `z` and `c`) reproduces the
   marker numbers within ~5 %.

---

## 4. Where to optimize (data-driven)

| Effort | Estimated impact | Mechanism |
|---|---:|---|
| Upstream PR to `module-lattice`: in-place `Vector::ntt`/`ntt_inverse` and accumulator-style matrix-vector mul | **−10 to −15 %** total cycles | Eliminates most of the 18 % `memcpy` bucket |
| **Custom NTT inline** (new `INLINE_OPCODE` + sequence builder under `jolt-inlines/mldsa`) | **−25 to −30 %** total cycles | Collapses `Polynomial::ntt`, `ntt_inverse`, and pointwise mul (42.5 % combined) into precompile cycles |
| Rejection-sample precompile (combine SHAKE squeeze + `< q` check + buffer-write) | **−10 %** total cycles | Targets the 8 % `rej_ntt_poly` plus part of the 12 % `Shake128::squeeze` |
| Cache parsed `VerifyingKey` across multiple verifications | **−25 % per additional verify** | Application-level change; ExpandA cost is amortized away |

None of these are needed for the current example to work — it produces a
valid Jolt proof of an upstream ML-DSA-65 signature verification in ~15 s
on a laptop today. They're listed as a roadmap, in priority order, with
numbers backing the priority.

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
