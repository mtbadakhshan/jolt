//! SHAKE128 / SHAKE256 sponge for ML-DSA, backed by the Jolt Keccak inline.
//!
//! This file replaces the upstream `crypto.rs` (which depended on `sha3`).
//! The public API surface used by the rest of `ml-dsa` is preserved:
//!
//! - `pub(crate) struct ShakeState<const RATE: usize>` with `absorb`,
//!   `squeeze`, `squeeze_new`.
//! - `pub(crate) type G = ShakeState<SHAKE128_RATE>` (rate 168 B).
//! - `pub(crate) type H = ShakeState<SHAKE256_RATE>` (rate 136 B).
//!
//! On RISC-V the Keccak-f[1600] permutation is the Jolt Keccak inline opcode.
//! On the host it falls back to a soft permutation (verbatim from the
//! upstream `keccak 0.2.0` Apache-2.0 / MIT crate).

use hybrid_array::Array;
use module_lattice::ArraySize;

pub(crate) const SHAKE128_RATE: usize = 168;
pub(crate) const SHAKE256_RATE: usize = 136;

#[derive(Clone)]
pub(crate) struct ShakeState<const RATE: usize> {
    state: [u64; 25],
    /// While absorbing: number of input bytes XORed into the current block (in `0..RATE`).
    /// While squeezing: number of output bytes already read from the current block (in `0..=RATE`).
    pos: usize,
    squeezing: bool,
}

impl<const RATE: usize> Default for ShakeState<RATE> {
    fn default() -> Self {
        Self {
            state: [0u64; 25],
            pos: 0,
            squeezing: false,
        }
    }
}

impl<const RATE: usize> ShakeState<RATE> {
    pub(crate) fn absorb(mut self, input: &[u8]) -> Self {
        debug_assert!(!self.squeezing, "cannot absorb after squeezing");
        let mut i = 0;
        while i < input.len() {
            let take = core::cmp::min(RATE - self.pos, input.len() - i);
            for j in 0..take {
                let byte_idx = self.pos + j;
                let lane = byte_idx >> 3;
                let shift = (byte_idx & 7) << 3;
                self.state[lane] ^= u64::from(input[i + j]) << shift;
            }
            self.pos += take;
            i += take;
            if self.pos == RATE {
                keccak_f1600(&mut self.state);
                self.pos = 0;
            }
        }
        self
    }

    fn finalize(&mut self) {
        // SHAKE domain separator: 0x1F at the data boundary, 0x80 in the last byte of the block.
        let pos_lane = self.pos >> 3;
        let pos_shift = (self.pos & 7) << 3;
        self.state[pos_lane] ^= 0x1F_u64 << pos_shift;

        let last = RATE - 1;
        let last_lane = last >> 3;
        let last_shift = (last & 7) << 3;
        self.state[last_lane] ^= 0x80_u64 << last_shift;

        keccak_f1600(&mut self.state);
        self.pos = 0;
        self.squeezing = true;
    }

    pub(crate) fn squeeze(&mut self, output: &mut [u8]) -> &mut Self {
        if !self.squeezing {
            self.finalize();
        }
        let mut i = 0;
        while i < output.len() {
            if self.pos == RATE {
                keccak_f1600(&mut self.state);
                self.pos = 0;
            }
            let take = core::cmp::min(RATE - self.pos, output.len() - i);
            for j in 0..take {
                let byte_idx = self.pos + j;
                let lane = byte_idx >> 3;
                let shift = (byte_idx & 7) << 3;
                output[i + j] = (self.state[lane] >> shift) as u8;
            }
            self.pos += take;
            i += take;
        }
        self
    }

    pub(crate) fn squeeze_new<N: ArraySize>(&mut self) -> Array<u8, N> {
        let mut v = Array::default();
        self.squeeze(&mut v);
        v
    }
}

pub(crate) type G = ShakeState<SHAKE128_RATE>;
pub(crate) type H = ShakeState<SHAKE256_RATE>;

#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
fn keccak_f1600(state: &mut [u64; 25]) {
    use jolt_inlines_keccak256::{INLINE_OPCODE, KECCAK256_FUNCT3, KECCAK256_FUNCT7};
    // SAFETY: `state` is a `[u64; 25]`, naturally u64-aligned. The Jolt Keccak
    // inline reads the state pointer from `rs1` and permutes the 200 bytes in
    // place; no other memory is touched.
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
fn keccak_f1600(state: &mut [u64; 25]) {
    soft::keccak_f1600(state);
}

#[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64")))]
mod soft {
    //! Software Keccak-f[1600], ported verbatim from upstream `keccak 0.2.0`'s
    //! `backends::soft` module (Apache-2.0 / MIT).

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

    pub(super) fn keccak_f1600(a: &mut [u64; 25]) {
        for rc in RC {
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

            let mut last = a[1];
            for x in 0..24 {
                array[0] = a[PI[x]];
                a[PI[x]] = last.rotate_left(RHO[x]);
                last = array[0];
            }

            for y_step in 0..5 {
                let y = y_step * 5;
                array[..5].copy_from_slice(&a[y..(5 + y)]);
                for x in 0..5 {
                    a[y + x] = array[x] ^ ((!array[(x + 1) % 5]) & array[(x + 2) % 5]);
                }
            }

            a[0] ^= rc;
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use hybrid_array::typenum::U32;

    // Test vectors lifted from the upstream `crypto.rs` so we know our SHAKE
    // wrapper is bit-compatible with `sha3 0.11`'s `Shake128` / `Shake256`.
    #[test]
    fn shake128_hello_world() {
        let mut g = G::default().absorb(b"hello world");
        let mut got = [0u8; 32];
        g.squeeze(&mut got);
        let expected: [u8; 32] = [
            0x3a, 0x91, 0x59, 0xf0, 0x71, 0xe4, 0xdd, 0x1c, 0x8c, 0x4f, 0x96, 0x86, 0x07, 0xc3,
            0x09, 0x42, 0xe1, 0x20, 0xd8, 0x15, 0x6b, 0x8b, 0x1e, 0x72, 0xe0, 0xd3, 0x76, 0xe8,
            0x87, 0x1c, 0xb8, 0xb8,
        ];
        assert_eq!(got, expected);

        let next: Array<u8, U32> = g.squeeze_new();
        let expected2: [u8; 32] = [
            0x99, 0x07, 0x26, 0x65, 0x67, 0x4f, 0x26, 0xcc, 0x49, 0x4a, 0x4b, 0xcf, 0x02, 0x7c,
            0x58, 0x26, 0x7e, 0x8e, 0xe2, 0xda, 0x60, 0xe9, 0x42, 0x75, 0x9d, 0xe8, 0x6d, 0x26,
            0x70, 0xbb, 0xa1, 0xaa,
        ];
        assert_eq!(<[u8; 32]>::from(next), expected2);
    }

    #[test]
    fn shake256_hello_world() {
        let mut h = H::default().absorb(b"hello world");
        let mut got = [0u8; 32];
        h.squeeze(&mut got);
        let expected: [u8; 32] = [
            0x36, 0x97, 0x71, 0xbb, 0x2c, 0xb9, 0xd2, 0xb0, 0x4c, 0x1d, 0x54, 0xcc, 0xa4, 0x87,
            0xe3, 0x72, 0xd9, 0xf1, 0x87, 0xf7, 0x3f, 0x7b, 0xa3, 0xf6, 0x5b, 0x95, 0xc8, 0xee,
            0x77, 0x98, 0xc5, 0x27,
        ];
        assert_eq!(got, expected);

        let next: Array<u8, U32> = h.squeeze_new();
        let expected2: [u8; 32] = [
            0xf4, 0xf3, 0xc2, 0xd5, 0x5c, 0x2d, 0x46, 0xa2, 0x9f, 0x2e, 0x94, 0x5d, 0x46, 0x9c,
            0x3d, 0xf2, 0x78, 0x53, 0xa8, 0x73, 0x52, 0x71, 0xf5, 0xcc, 0x2d, 0x9e, 0x88, 0x95,
            0x44, 0x35, 0x71, 0x16,
        ];
        assert_eq!(<[u8; 32]>::from(next), expected2);
    }
}
