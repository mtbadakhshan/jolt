//! Minimal `keccak 0.2.x` reimplementation that routes the 1600-bit
//! permutation through the Jolt Keccak inline when compiled for RISC-V, and
//! falls back to a software permutation on the host.
//!
//! This crate exists solely to be slotted in via `[patch.crates-io]` in the
//! workspace root. It exposes only the API surface that `sha3 0.11` actually
//! consumes (see `sha3-0.11.0/src/block_api.rs`): `Keccak`, `State1600`,
//! `with_f1600`, and a couple of helper aliases/constants.
//!
//! On RISC-V (the Jolt guest target), `Keccak::with_f1600` hands out a
//! function pointer that invokes `jolt_inlines_keccak256::keccak_f`. This is
//! what makes any SHA3 / SHAKE consumer in the guest (notably `ml-dsa`) end
//! up using the existing Jolt Keccak inline instead of the compiled
//! software permutation.

#![no_std]

pub const PLEN: usize = 25;

pub type State1600 = [u64; PLEN];
pub type Fn1600 = fn(&mut State1600);

#[derive(Debug, Copy, Clone, Default)]
pub struct Keccak;

impl Keccak {
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Run `f` with the 1600-bit Keccak-f permutation. Mirrors the upstream
    /// `keccak::Keccak::with_f1600` API.
    #[inline]
    pub fn with_f1600(&self, f: impl FnOnce(Fn1600)) {
        f(f1600);
    }
}

#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
fn f1600(state: &mut State1600) {
    use jolt_inlines_keccak256::{INLINE_OPCODE, KECCAK256_FUNCT3, KECCAK256_FUNCT7};
    // SAFETY: `state` points to a `[u64; 25]`, properly aligned for u64.
    //         The Jolt Keccak inline reads the state pointer from `rs1` and
    //         permutes it in place; no other memory is touched.
    unsafe {
        core::arch::asm!(
            ".insn r {opcode}, {funct3}, {funct7}, x0, {rs1}, x0",
            opcode = const INLINE_OPCODE,
            funct3 = const KECCAK256_FUNCT3,
            funct7 = const KECCAK256_FUNCT7,
            rs1 = in(reg) state.as_mut_ptr(),
            options(nostack),
        );
    }
}

#[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64")))]
fn f1600(state: &mut State1600) {
    soft::keccak_f1600(state);
}

#[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64")))]
mod soft {
    //! Software Keccak-f[1600] permutation, ported verbatim from upstream
    //! `keccak 0.2.0`'s `backends::soft` module (Apache-2.0 / MIT).

    use super::State1600;

    const RHO: [u32; 24] = [
        1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44,
    ];

    const PI: [usize; 24] = [
        10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1,
    ];

    const RC: [u64; 24] = [
        0x0000_0000_0000_0001,
        0x0000_0000_0000_8082,
        0x8000_0000_0000_808a,
        0x8000_0000_8000_8000,
        0x0000_0000_0000_808b,
        0x0000_0000_8000_0001,
        0x8000_0000_8000_8081,
        0x8000_0000_0000_8009,
        0x0000_0000_0000_008a,
        0x0000_0000_0000_0088,
        0x0000_0000_8000_8009,
        0x0000_0000_8000_000a,
        0x0000_0000_8000_808b,
        0x8000_0000_0000_008b,
        0x8000_0000_0000_8089,
        0x8000_0000_0000_8003,
        0x8000_0000_0000_8002,
        0x8000_0000_0000_0080,
        0x0000_0000_0000_800a,
        0x8000_0000_8000_000a,
        0x8000_0000_8000_8081,
        0x8000_0000_0000_8080,
        0x0000_0000_8000_0001,
        0x8000_0000_8000_8008,
    ];

    pub(super) fn keccak_f1600(a: &mut State1600) {
        for rc in RC {
            // θ
            let mut array: [u64; 5] = [0; 5];
            for x in 0..5 {
                for y_count in 0..5 {
                    let y = y_count * 5;
                    array[x] ^= a[x + y];
                }
            }
            for x in 0..5 {
                for y_count in 0..5 {
                    let y = y_count * 5;
                    a[y + x] ^= array[(x + 4) % 5] ^ array[(x + 1) % 5].rotate_left(1);
                }
            }

            // ρ and π
            let mut last = a[1];
            for x in 0..24 {
                array[0] = a[PI[x]];
                a[PI[x]] = last.rotate_left(RHO[x]);
                last = array[0];
            }

            // χ
            for y_step in 0..5 {
                let y = y_step * 5;
                array[..5].copy_from_slice(&a[y..(5 + y)]);
                for x in 0..5 {
                    a[y + x] = array[x] ^ ((!array[(x + 1) % 5]) & array[(x + 2) % 5]);
                }
            }

            // ι
            a[0] ^= rc;
        }
    }
}
