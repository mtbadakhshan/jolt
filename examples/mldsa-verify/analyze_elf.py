#!/usr/bin/env python3
"""
Static instruction-mix analysis of the ML-DSA verify guest ELF.

The ELF is fully stripped (no symbol table), so per-function attribution is not
possible from the binary alone. Instead we walk `.text` decoding every RV64IMAC
instruction (including compressed RVC variants) and bucket them by opcode
family. We pay special attention to the custom Keccak inline (opcode 0x0B,
funct3 0, funct7 1) because each such instruction expands at trace time into
~3,037 cycles via the sequence builder, and we already know the runtime cycle
totals per phase from the cycle markers.

The output gives a static picture: how many *call sites* exist for each kind of
operation in the compiled binary. Cross-checked against runtime totals at the
bottom.
"""

from __future__ import annotations

import struct
import sys
from dataclasses import dataclass
from pathlib import Path

ELF_PATH = Path(
    "/tmp/jolt-guest-targets/mldsa-verify-guest-mldsa_verify/"
    "riscv64imac-unknown-none-elf/release/mldsa-verify-guest"
)

# Custom Keccak inline encoding:
#   opcode = 0x0B (custom-0), funct3 = 0, funct7 = 1, rd = x0, rs2 = x0
# rs1 holds the state pointer and varies.
KECCAK_MASK = 0xFE00_F07F  # match funct7 + funct3 + rs2 + opcode, leave rs1/rd free
KECCAK_VAL = 0x0200_000B   # funct7=1<<25, funct3=0, opcode=0x0B, rs2=0

# Standard RV64 opcode family map (bits 6:0 → name). Compressed handled below.
OPCODE_NAMES = {
    0x03: "load",                   # LB/LH/LW/LBU/LHU/LWU/LD
    0x07: "load_fp",
    0x0F: "misc_mem",               # FENCE / FENCE.I
    0x13: "op_imm",                 # ADDI/SLLI/...
    0x17: "auipc",
    0x1B: "op_imm_32",              # ADDIW/SLLIW/...
    0x23: "store",                  # SB/SH/SW/SD
    0x27: "store_fp",
    0x2F: "amo",                    # AMO*
    0x33: "op",                     # ADD/SUB/...(funct7=0/0x20) or MUL/DIV (funct7=1)
    0x37: "lui",
    0x3B: "op_32",                  # ADDW/SUBW/MULW/...
    0x53: "op_fp",
    0x63: "branch",
    0x67: "jalr",
    0x6F: "jal",
    0x73: "system",                 # ECALL/EBREAK/CSR*
    0x0B: "custom_0",               # our Keccak inline lives here
    0x2B: "custom_1",
}


@dataclass
class ElfText:
    base: int
    bytes: bytes


def read_elf_text(path: Path) -> ElfText:
    """Locate the .text section in a stripped ELF64-LE RISC-V binary."""
    with path.open("rb") as f:
        d = f.read()

    # ELF64 header — we only need e_shoff, e_shentsize, e_shnum, e_shstrndx.
    # Layout: e_shoff @ 0x28 (Q), e_shentsize @ 0x3A (H), e_shnum @ 0x3C (H),
    # e_shstrndx @ 0x3E (H).
    assert d[:4] == b"\x7fELF" and d[4] == 2, "not ELF64-LE"
    (e_shoff,) = struct.unpack_from("<Q", d, offset=0x28)
    e_shentsize, e_shnum, e_shstrndx = struct.unpack_from("<HHH", d, offset=0x3A)
    # Section header table: each entry is 64 bytes for ELF64.
    # Layout: name(4) type(4) flags(8) addr(8) offset(8) size(8) link(4) info(4) addralign(8) entsize(8)
    shdrs = []
    for i in range(e_shnum):
        off = e_shoff + i * e_shentsize
        sh = struct.unpack_from("<IIQQQQIIQQ", d, off)
        shdrs.append(sh)
    sh_strtab = shdrs[e_shstrndx]
    strtab_off = sh_strtab[4]  # offset
    strtab_size = sh_strtab[5]
    strtab = d[strtab_off : strtab_off + strtab_size]

    for sh in shdrs:
        name_off = sh[0]
        # zero-terminated section name
        end = strtab.index(b"\x00", name_off)
        name = strtab[name_off:end].decode("ascii", errors="replace")
        if name == ".text":
            addr = sh[3]
            off = sh[4]
            size = sh[5]
            return ElfText(base=addr, bytes=d[off : off + size])
    raise RuntimeError(".text section not found")


def decode_text(text: ElfText) -> dict[str, int]:
    """
    Walk .text decoding instruction-by-instruction (RVC-aware) and bucket by
    opcode family.
    """
    counts: dict[str, int] = {}
    keccak_inline_count = 0
    mul_count = 0
    div_rem_count = 0
    branch_count = 0
    cjump_count = 0  # compressed branch/jal counts (folded into "branch"/"jal")

    pc = 0
    end = len(text.bytes)
    while pc < end:
        # Compressed if bits[1:0] != 0b11
        first = text.bytes[pc]
        if (first & 0x3) != 0x3:
            # 16-bit compressed instruction
            if pc + 2 > end:
                break
            (insn16,) = struct.unpack_from("<H", text.bytes, pc)
            family = decode_rvc(insn16)
            counts[family] = counts.get(family, 0) + 1
            if family == "c_branch":
                branch_count += 1
            pc += 2
        else:
            # 32-bit instruction
            if pc + 4 > end:
                break
            (insn,) = struct.unpack_from("<I", text.bytes, pc)
            family, sub = decode_rv32(insn)
            counts[family] = counts.get(family, 0) + 1
            if family == "custom_0" and (insn & KECCAK_MASK) == KECCAK_VAL:
                keccak_inline_count += 1
            if sub == "mul":
                mul_count += 1
            if sub == "div_rem":
                div_rem_count += 1
            if family == "branch":
                branch_count += 1
            pc += 4

    counts["__keccak_inline_calls"] = keccak_inline_count
    counts["__mul_instructions"] = mul_count
    counts["__div_rem_instructions"] = div_rem_count
    counts["__all_branches"] = branch_count
    return counts


def decode_rv32(insn: int) -> tuple[str, str | None]:
    """Return (family, subkind) for a 32-bit RV instruction."""
    opcode = insn & 0x7F
    family = OPCODE_NAMES.get(opcode, f"unknown_{opcode:#x}")

    # Inspect M extension (mul/div/rem) inside OP / OP-32.
    if family in ("op", "op_32"):
        funct7 = (insn >> 25) & 0x7F
        if funct7 == 0x01:
            funct3 = (insn >> 12) & 0x7
            return family, "mul" if funct3 < 4 else "div_rem"

    return family, None


# RVC decoder — partial, just enough to bucket families. Encoding from
# RISC-V "C" extension spec (RV64 quadrants 0, 1, 2).
def decode_rvc(insn: int) -> str:
    op = insn & 0x3
    funct3 = (insn >> 13) & 0x7
    if op == 0b00:
        # C.ADDI4SPN / C.LW / C.LD / C.SW / C.SD / etc.
        return {
            0: "c_addi4spn",
            2: "c_lw",
            3: "c_ld",
            6: "c_sw",
            7: "c_sd",
        }.get(funct3, "c_q0_other")
    if op == 0b01:
        return {
            0: "c_addi",
            1: "c_addiw",
            2: "c_li",
            3: "c_addi16sp_or_lui",
            4: "c_arith_imm",
            5: "c_j",
            6: "c_beqz",
            7: "c_bnez",
        }.get(funct3, "c_q1_other")
    if op == 0b10:
        return {
            0: "c_slli",
            2: "c_lwsp",
            3: "c_ldsp",
            4: "c_jr_or_mv_or_jalr_or_add",
            6: "c_swsp",
            7: "c_sdsp",
        }.get(funct3, "c_q2_other")
    return "c_unknown"


# Pretty-printing.

GROUPS = {
    "Keccak inline (custom_0)":   ["__keccak_inline_calls"],
    "Memory loads":               ["load", "c_lw", "c_ld", "c_lwsp", "c_ldsp"],
    "Memory stores":              ["store", "c_sw", "c_sd", "c_swsp", "c_sdsp"],
    "Integer arithmetic (imm)":   ["op_imm", "op_imm_32", "c_addi", "c_addiw",
                                   "c_li", "c_addi16sp_or_lui", "c_arith_imm",
                                   "c_slli", "c_addi4spn"],
    "Integer arithmetic (reg)":   ["op", "op_32", "c_jr_or_mv_or_jalr_or_add"],
    "  └─ of which mul":          ["__mul_instructions"],
    "  └─ of which div / rem":    ["__div_rem_instructions"],
    "Branches (all)":             ["__all_branches", "c_beqz", "c_bnez"],
    "Jumps (jal / jalr)":         ["jal", "jalr", "c_j"],
    "Upper immediate (lui/auipc)": ["lui", "auipc"],
    "Atomics (AMO)":              ["amo"],
    "Fences":                     ["misc_mem"],
    "System (ECALL/CSR)":         ["system"],
    "Other / unknown":            [],  # filled below
}


def main() -> None:
    text = read_elf_text(ELF_PATH)
    counts = decode_text(text)

    total_static_instrs = sum(
        v for k, v in counts.items() if not k.startswith("__")
    )

    # Capture any keys not already bucketed.
    accounted = {k for vs in GROUPS.values() for k in vs}
    other_keys = sorted(
        k for k in counts if not k.startswith("__") and k not in accounted
    )
    GROUPS["Other / unknown"] = other_keys

    print()
    print("=" * 72)
    print(" Static instruction mix in ML-DSA-verify guest ELF (.text)")
    print(f" ELF base = 0x{text.base:08x},  .text size = {len(text.bytes):,} bytes")
    print(f" Total static instructions (RV+RVC) = {total_static_instrs:,}")
    print("=" * 72)

    for group, keys in GROUPS.items():
        n = sum(counts.get(k, 0) for k in keys)
        if n == 0 and group != "Other / unknown":
            continue
        pct = 100.0 * n / total_static_instrs if total_static_instrs else 0.0
        print(f"  {group:42}  {n:>8,}   {pct:5.1f}%")
        if group == "Other / unknown" and n > 0:
            for k in keys:
                v = counts.get(k, 0)
                if v:
                    print(f"      {k:38}  {v:>8,}")

    print("-" * 72)
    print()
    print(" Cross-checks against the runtime cycle markers")
    print(" =============================================================")

    # Runtime numbers we collected from the cycle markers.
    rt_real   = 2_219_038            # all phases, RV64IMAC instructions only
    rt_virt   = 3_238_538            # virtual instructions (from inline + bytecode expansion)
    rt_total  = rt_real + rt_virt
    phase1_total = 2_562_711
    phase2_total =   219_769
    phase3_total = 2_674_594

    keccak_static = counts["__keccak_inline_calls"]
    text_bytes = len(text.bytes)

    # The Keccak sequence builder emits roughly one virtual instruction per
    # micro-op: 25 LDs + 24 × (theta 65 + rho/pi 25 + chi 50 + iota 1) + 25 SDs
    # ≈ 50 + 24·141 ≈ 3,434 virtual instructions per Keccak-f permutation.
    keccak_expansion = 3434
    keccak_perms_upper_bound = rt_virt // keccak_expansion

    # Other heavy expansions (LWU, LD-with-alignment, MULH, MULHSU, etc.) also
    # add virtual instructions. We don't know the exact mix dynamically, but
    # static counts give a feel for what's expensive.
    print(f"  .text size                                : {text_bytes:>10,} bytes")
    print(f"  Static instructions decoded               : {total_static_instrs:>10,}")
    print(f"  Static Keccak-inline call sites           : {keccak_static:>10}")
    print()
    print(f"  Runtime RV64IMAC instructions (real)      : {rt_real:>10,}")
    print(f"  Runtime virtual instructions (expansion)  : {rt_virt:>10,}")
    print(f"  Runtime total cycles                      : {rt_total:>10,}")
    print(f"  Code-to-trace amplification (real / static): {rt_real / total_static_instrs:>9.1f}x")
    print()
    print("  Per phase (from cycle markers):")
    print(f"    phase1_decode_pk_and_expand_a            : {phase1_total:>10,}  ({100*phase1_total/rt_total:4.1f}%)")
    print(f"    phase2_decode_signature                  : {phase2_total:>10,}  ({100*phase2_total/rt_total:4.1f}%)")
    print(f"    phase3_verify_internal                   : {phase3_total:>10,}  ({100*phase3_total/rt_total:4.1f}%)")
    print()
    print(" Keccak contribution (math, not measurement)")
    print(" -------------------------------------------------------------")
    print(f"  Each Keccak inline emits ~{keccak_expansion:,} virtual instructions")
    print("  (25 LDs + 24 rounds × ~141 ops + 25 SDs in the sequence builder).")
    print()
    print(f"  Upper bound on Keccak permutations during run:")
    print(f"    {rt_virt:,} virtual / {keccak_expansion:,} = {keccak_perms_upper_bound} permutations")
    print("    (true value is lower — non-Keccak instructions also produce virtuals)")
    print()
    print(f"  Algorithmic expectation for ML-DSA-65 Verify:")
    print(f"    ExpandA: K*L=30 SHAKE128 streams; ~5-7 perms per stream    ≈ 150-210")
    print(f"    tr = H(pk):  1 SHAKE256, pk=1952B → 15 absorb perms        ≈  15")
    print(f"    μ = H(tr|msg): 1 SHAKE256, small input                     ≈   1-2")
    print(f"    SampleInBall: 1 SHAKE256 stream                            ≈   1-2")
    print(f"    c̃' = H(μ|w1Encode(w1)): w1 is ~1024 B                      ≈   8-10")
    print(f"    TOTAL                                                       ≈ 175-240 perms")
    print()
    print(f"  Expected Keccak virtual cycles: 175-240 × {keccak_expansion} ≈ 600k-820k")
    print(f"  That's {100*200*keccak_expansion/rt_virt:.0f}% of all virtual cycles (using midpoint 200).")
    print(f"  Implication: the other ~{rt_virt - 200*keccak_expansion:,} virtuals come from")
    print("  bytecode expansion of non-Keccak ops (LWU, MULH, LD alignment etc.).")
    print()


if __name__ == "__main__":
    main()
