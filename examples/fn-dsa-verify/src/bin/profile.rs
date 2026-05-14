//! Per-symbol cycle profiler for the FN-DSA-512 verify guest.
//!
//! Mirrors `examples/mldsa-verify/src/bin/profile.rs`. Requires the guest to
//! be built with the `guest-profile` Cargo profile and `JOLT_BACKTRACE=1` so
//! the ELF retains its symbol table.
//!
//! Pipeline:
//!   1. Build the guest (via the macro-generated `compile_fn_dsa_verify`).
//!   2. Read the ELF, parse `.symtab` into a sorted `(start, end, name)` table.
//!   3. Run the tracer end-to-end via `trace_fn_dsa_verify(pk, sig, msg)`.
//!   4. Iterate every `TraceRow`, look up the PC, bump per-function counters
//!      for real vs virtual cycles.
//!   5. Print the top functions by cycle count.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use fn_dsa::{
    sign_key_size, signature_size, vrfy_key_size, KeyPairGenerator, KeyPairGeneratorStandard,
    SigningKey, SigningKeyStandard, DOMAIN_NONE, FN_DSA_LOGN_512, HASH_ID_RAW,
};
use fn_dsa_vrfy::{compute_hashed_key, hash_to_point};
use guest::precomputed::serialize_c;
use jolt_inlines_keccak256 as _;
use object::{Object, ObjectSymbol, SymbolKind};
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rustc_demangle::demangle;
use tracing_subscriber::EnvFilter;

#[derive(Clone, Debug)]
struct Symbol {
    start: u64,
    end: u64,
    name: String,
}

fn load_symbols(elf_path: &PathBuf) -> Vec<Symbol> {
    let bytes =
        fs::read(elf_path).unwrap_or_else(|e| panic!("could not read ELF at {elf_path:?}: {e}"));
    let obj = object::File::parse(&*bytes).expect("not a valid ELF");

    let mut syms: Vec<Symbol> = obj
        .symbols()
        .filter(|s| s.kind() == SymbolKind::Text && s.size() > 0)
        .map(|s| Symbol {
            start: s.address(),
            end: s.address() + s.size(),
            name: s
                .name()
                .map(|n| format!("{:#}", demangle(n)))
                .unwrap_or_else(|_| "<unreadable>".into()),
        })
        .collect();
    syms.sort_by_key(|s| s.start);
    syms
}

fn find_owner(syms: &[Symbol], pc: u64) -> Option<&Symbol> {
    let idx = syms.partition_point(|s| s.start <= pc).checked_sub(1)?;
    let sym = &syms[idx];
    if pc < sym.end {
        Some(sym)
    } else {
        None
    }
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let target_dir = "/tmp/jolt-guest-targets";
    let program = guest::compile_fn_dsa_verify(target_dir);

    let elf_path = locate_elf();
    println!("loading symbols from: {}", elf_path.display());
    let symbols = load_symbols(&elf_path);
    println!("loaded {} text symbols", symbols.len());

    // Same deterministic keygen + signing as `src/main.rs`.
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

    // Host-side precomputation (matches main.rs).
    let hashed_key = compute_hashed_key(&vrfy_key);
    let n = 1usize << FN_DSA_LOGN_512;
    let nonce = &sig[1..41];
    let mut c = vec![0u16; n];
    hash_to_point(nonce, &hashed_key, &DOMAIN_NONE, &HASH_ID_RAW, &msg, &mut c);
    let c_bytes = serialize_c(&c);

    let trace_out = guest::trace_fn_dsa_verify(&vrfy_key, &sig, &msg, &hashed_key, &c_bytes)
        .expect("tracer failed");
    let rows = trace_out.trace.rows();
    println!("trace length: {} rows", rows.len());

    let mut totals: HashMap<String, (u64, u64)> = HashMap::new();
    let mut virtual_count = 0u64;
    let mut real_count = 0u64;

    for row in rows {
        let pc = row.instruction.address as u64;
        let is_virtual = row
            .instruction
            .virtual_sequence_remaining
            .map(|n| !row.instruction.is_first_in_sequence || n > 0)
            .unwrap_or(false);

        let bucket = match find_owner(&symbols, pc) {
            Some(sym) => sym.name.as_str(),
            None => "<unattributed>",
        };

        let entry = totals.entry(bucket.to_string()).or_insert((0, 0));
        if is_virtual {
            entry.1 += 1;
            virtual_count += 1;
        } else {
            entry.0 += 1;
            real_count += 1;
        }
    }

    let total = real_count + virtual_count;
    println!();
    println!(" Total cycles : {total:>10}  (real {real_count} + virtual {virtual_count})");
    println!();

    let mut items: Vec<_> = totals.iter().collect();
    items.sort_by(|a, b| (b.1 .0 + b.1 .1).cmp(&(a.1 .0 + a.1 .1)));

    println!(
        "{:>10}  {:>10}  {:>10}  {:>6}  function",
        "real", "virtual", "total", "%"
    );
    println!("{}", "-".repeat(72));
    let cap = 40usize;
    let mut shown_total: u64 = 0;
    for (name, (real, virt)) in items.iter().take(cap) {
        let t = real + virt;
        let pct = 100.0 * t as f64 / total as f64;
        let short = shorten_name(name);
        println!("{real:>10}  {virt:>10}  {t:>10}  {pct:>5.1}%  {short}");
        shown_total += t;
    }
    let remainder = total - shown_total;
    let remainder_pct = 100.0 * remainder as f64 / total as f64;
    println!("{}", "-".repeat(72));
    println!(
        "{:>10}  {:>10}  {:>10}  {:>5.1}%  ... {} more buckets",
        "",
        "",
        remainder,
        remainder_pct,
        items.len().saturating_sub(cap)
    );

    let _ = program;
}

fn locate_elf() -> PathBuf {
    let candidates = [
        "/tmp/jolt-guest-targets/fn-dsa-verify-guest-fn_dsa_verify/\
         riscv64imac-unknown-none-elf/guest-profile/fn-dsa-verify-guest",
        "/tmp/jolt-guest-targets/fn-dsa-verify-guest-fn_dsa_verify/\
         riscv64imac-unknown-none-elf/release/fn-dsa-verify-guest",
    ];
    for path in candidates {
        let p = PathBuf::from(path);
        if p.exists() {
            return p;
        }
    }
    panic!(
        "could not find guest ELF. Run with `JOLT_BACKTRACE=1 cargo run --release \
         --bin fn-dsa-profile -p fn-dsa-verify` after `compile_*` has built it."
    )
}

fn shorten_name(s: &str) -> String {
    let mut t: &str = s;
    if let Some(pos) = t.rfind("::") {
        let tail = &t[pos + 2..];
        if tail.len() == 17
            && tail.starts_with('h')
            && tail.chars().all(|c| c.is_ascii_hexdigit() || c == 'h')
        {
            t = &t[..pos];
        }
    }
    if t.len() > 88 {
        format!("{}…", &t[..87])
    } else {
        t.to_string()
    }
}
