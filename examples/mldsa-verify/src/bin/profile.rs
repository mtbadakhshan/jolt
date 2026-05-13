//! Per-symbol cycle profiler for the ML-DSA-65 verify guest.
//!
//! Requires the guest to be built with the `guest-profile` cargo profile and
//! `JOLT_BACKTRACE=1` so the ELF retains its symbol table. Without symbols
//! we can't attribute PCs to function names.
//!
//! Pipeline:
//!   1. Build the guest (via the macro-generated `compile_*`).
//!   2. Read the ELF, parse `.symtab` into a sorted `(start, end, name)` table.
//!   3. Run the tracer end-to-end via `trace_mldsa_verify(pk, msg, sig)`.
//!   4. Iterate every `TraceRow`, look up the PC in the symbol table, bump a
//!      counter for the owning function.
//!   5. Print the top functions by cycle count (real + virtual).

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

// Force-link the Keccak inline crate so its `register_inlines!` static lands
// in `inventory`.
use jolt_inlines_keccak256 as _;

use ml_dsa::{MlDsa65, SigningKey};
use object::{Object, ObjectSymbol, SymbolKind};
use rustc_demangle::demangle;
use signature::{Keypair, Signer};

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

/// Binary-search the symbol owning `pc`.
fn find_owner(syms: &[Symbol], pc: u64) -> Option<&Symbol> {
    // Last symbol whose `start <= pc`.
    let idx = syms.partition_point(|s| s.start <= pc).checked_sub(1)?;
    let sym = &syms[idx];
    if pc < sym.end {
        Some(sym)
    } else {
        None
    }
}

fn main() {
    tracing_subscriber::fmt::init();

    // Build the guest. This invokes `jolt build` under the hood. With
    // `profile = "guest-profile"` set on the macro and `JOLT_BACKTRACE=1` in
    // the env, the resulting ELF retains `.symtab` and DWARF line tables.
    let target_dir = "/tmp/jolt-guest-targets";
    let program = guest::compile_mldsa_verify(target_dir);

    // Read the ELF directly from the cached build path. The compile step
    // already loaded its contents into `program.elf`; we just need a `PathBuf`.
    let elf_path = locate_elf();
    println!("loading symbols from: {}", elf_path.display());
    let symbols = load_symbols(&elf_path);
    println!("loaded {} text symbols", symbols.len());

    // Generate a real ML-DSA-65 signature (same as the prove/verify driver).
    let seed_bytes: [u8; 32] = core::array::from_fn(|i| (i as u8) ^ 0xA5);
    let sk = SigningKey::<MlDsa65>::from_seed(&seed_bytes.into());
    let vk = sk.verifying_key();
    let msg = b"ML-DSA-65 hello, world!".to_vec();
    let signature = sk.sign(&msg);
    let pk_bytes = vk.encode().to_vec();
    let sig_bytes = signature.encode().to_vec();

    // The macro-generated `trace_*` runs the tracer end-to-end and gives back
    // every row, each one carrying a `NormalizedInstruction { address, .. }`.
    let trace_out = guest::trace_mldsa_verify(&pk_bytes, &msg, &sig_bytes).expect("tracer failed");
    let rows = trace_out.trace.rows();
    println!("trace length: {} rows", rows.len());

    // Aggregate cycle counts per symbol. We bucket "rows from the same
    // logical function" by mangled-and-demangled symbol name; this folds
    // codegen-units suffixes together. PCs that fall outside every symbol
    // (e.g. inline-expansion virtual rows whose `address` is the parent
    // call site) are tallied under "<unattributed>".
    let mut totals: HashMap<String, (u64, u64)> = HashMap::new(); // name → (real, virtual)
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

    // Sort by total cycles desc.
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

    let _ = program; // suppress unused warnings; we keep the guard alive
}

fn locate_elf() -> PathBuf {
    let candidates = [
        "/tmp/jolt-guest-targets/mldsa-verify-guest-mldsa_verify/\
         riscv64imac-unknown-none-elf/guest-profile/mldsa-verify-guest",
        "/tmp/jolt-guest-targets/mldsa-verify-guest-mldsa_verify/\
         riscv64imac-unknown-none-elf/release/mldsa-verify-guest",
    ];
    for path in candidates {
        let p = PathBuf::from(path);
        if p.exists() {
            return p;
        }
    }
    panic!(
        "could not find guest ELF. Run with `JOLT_BACKTRACE=1 cargo run --release \
         --bin mldsa-profile -p mldsa-verify` after `compile_*` has built it."
    )
}

/// Tighten very long Rust mangled names so the table stays readable.
fn shorten_name(s: &str) -> String {
    // Drop hash suffix like ::h1a2b3c4d.
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
