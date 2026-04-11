# NTLM Kernel Optimization — Claim Validation Report

**Date:** 2026-04-11
**Branch:** ntlm-kernel-optimization
**Investigator:** Claude (Sonnet 4.6)
**Source:** Validates claims in `2026-04-11-ntlm-kernel-optimization-investigation.md`

---

## Claim 1: Metal disables bitselect on Apple Silicon

**Claim:** `inc_vendor.h` guards `USE_BITSELECT` with `#ifndef IS_APPLE_SILICON`, meaning
Metal on Apple Silicon uses 3–5 operations instead of 1 fused bitselect for MD4 F and G rounds.

### Evidence

`OpenCL/inc_vendor.h`, lines 196–208:

```c
#ifdef IS_METAL
#define USE_ROTATE
#ifndef IS_APPLE_SILICON
#define USE_BITSELECT
#define USE_SWIZZLE
#endif
// ...
#endif
```

`OpenCL/inc_hash_md4.h`, lines 17–23:

```c
#ifdef USE_BITSELECT
#define MD4_Fo(x,y,z)   (bitselect ((z), (y), (x)))
#define MD4_Go(x,y,z)   (bitselect ((x), (y), ((x) ^ (z))))
#else
#define MD4_Fo(x,y,z)   (MD4_F((x), (y), (z)))
#define MD4_Go(x,y,z)   (MD4_G((x), (y), (z)))
#endif
```

Where the fallback macros are:
- `MD4_F(x,y,z) = ((x) & (y)) | ((~(x)) & (z))` — 3 operations (AND, ANDN, OR)
- `MD4_G(x,y,z) = ((x) & (y)) | ((x) & (z)) | ((y) & (z))` — 5 operations (3×AND, 2×OR)

For CUDA, HIP, and OpenCL (all non-Metal platforms), `USE_BITSELECT` is unconditionally
defined (lines 181–194), so `MD4_Fo` and `MD4_Go` collapse to a single fused `bitselect`.

### Surprising Finding: Metal Device Is Skipped Locally

On the test machine (Apple M3 Max), running `hashcat -b -m 1000` shows:

```
METAL API (Metal 372.16)
========================
* Device #01: Apple M3 Max, skipped

OpenCL API (OpenCL 1.2 ...) - Platform #1 [Apple]
====================================================================
* Device #02: Apple M3 Max, GPU, 26542/53084 MB ...
```

This is the **opposite** of what the investigation plan described. The plan says the
alias-dedup logic in `src/backend.c` marks the OpenCL device as skipped in favour of Metal,
but on this system the Metal device is skipped and OpenCL is used. This means benchmarks
here run through the **OpenCL** path with `USE_BITSELECT` **enabled**, not the Metal path
where it would be disabled.

The `IS_APPLE_SILICON` guard is still correct as written in source — it is just that the
active execution path is OpenCL, not Metal, on this test machine. The claim about Metal
disabling bitselect is source-code accurate, but the practical impact cannot be measured
here without a system where Metal is the active device.

### Verdict: CONFIRMED (code logic), UNCONFIRMED (runtime impact on this machine)

The preprocessor logic is exactly as claimed. The runtime consequence cannot be measured
on this machine because the Metal device is skipped in favour of OpenCL. Any system where
hashcat actually uses the Metal backend would experience the described fallback to slower
F/G round functions.

---

## Claim 2: a0/a1 kernels lack precomputation that a3 has

**Claim:** a3-optimized precomputes `w[i] + MD4Cxx` sums before the inner loop, but a0 and
a1 don't. Each MD4_STEP in a0/a1 therefore pays an extra `make_u32x(K)` add per non-varying
word.

### Evidence

**a3-optimized** (`OpenCL/m01000_a3-optimized.cl`, lines 42–91) — precomputation block in
`m01000m()` before the loop:

```c
const u32 F_w0c00 =     0 + MD4C00;
const u32 F_w1c00 = w[ 1] + MD4C00;
// ... 16 F-round constants
const u32 G_w0c01 =     0 + MD4C01;
// ... 16 G-round constants
const u32 H_w0c02 =     0 + MD4C02;
// ... 16 H-round constants
// Total: 48 precomputed scalar u32 values
```

Inside the loop, 47 of 48 steps use `MD4_STEP0` (lines 111–159):
```c
MD4_STEP0(MD4_Fo, d, a, b, c,     F_w1c00, MD4S01);  // no make_u32x(K), no x arg
```

`MD4_STEP0` macro (`inc_hash_md4.h`, lines 39–43):
```c
#define MD4_STEP0(f,a,b,c,d,K,s)  {
  a  = hc_add3 (a, K, f (b, c, d));   // K already contains w+constant sum
  a  = hc_rotl32 (a, s);
}
```

**a0-optimized** (`OpenCL/m01000_a0-optimized.cl`, lines 75–124) — no precomputation,
all 48 steps use `MD4_STEP`:

```c
MD4_STEP (MD4_Fo, a, b, c, d, w0[0], MD4C00, MD4S00);
MD4_STEP (MD4_Fo, d, a, b, c, w0[1], MD4C00, MD4S01);
// ... all 48 steps are MD4_STEP, never MD4_STEP0
```

`MD4_STEP` macro (`inc_hash_md4.h`, lines 32–37):
```c
#define MD4_STEP(f,a,b,c,d,x,K,s)  {
  a += make_u32x (K);                 // extra add: broadcasts scalar K to vector
  a  = hc_add3 (a, x, f (b, c, d));  // second add
  a  = hc_rotl32 (a, s);
}
```

**a1-optimized** (`OpenCL/m01000_a1-optimized.cl`, lines 135–184) — identical to a0:
all 48 steps use `MD4_STEP` with live word values and raw `MD4C0x` constants. No
precomputation block before the loop.

### Extra Adds Quantified

In a0 and a1, every MD4_STEP performs `a += make_u32x(K)` before `hc_add3`. With 48 steps
per MD4 and no precomputation, this is **48 extra vectorised adds per candidate** compared
to a3's inner loop (which only does 1 `MD4_STEP` call — for `w0` which varies — and 47
`MD4_STEP0` calls).

### Verdict: CONFIRMED

The precomputation block (48 constants) is present in a3-optimized (both `m01000m` and
`m01000s` helper functions) and entirely absent from a0-optimized and a1-optimized. The a0
and a1 kernels perform 47 additional vectorised `make_u32x(K) + add` operations per
candidate per iteration.

---

## Claim 3: Meet-in-the-middle only exists in a3

**Claim:** MITM (reverse MD4 pass, early termination using `MATCHES_NONE_VV` or
`MD4_STEP_REV`) is only in a3-optimized. a0 and a1 lack it.

### Evidence

**a3-optimized** (`OpenCL/m01000_a3-optimized.cl`):

- Lines 17–30: `MD4_STEP_REV` and `MD4_STEP_REV1` macros defined
- Lines 247–268 (`m01000s` helper): full backward pass from the target digest, peeling the
  entire H round (16 steps) and 2 G-round steps using `MD4_STEP_REV` to produce
  `a_rev, b_rev, c_rev, d_rev, sav_c, sav_d`
- Lines 322–324 (inside the inner loop): three mid-round comparisons:
  ```c
  MD4_STEP0(MD4_Go, c, d, a, b, G_wac01, MD4S12); if (MATCHES_NONE_VV (c, pre_c)) continue;
  MD4_STEP0(MD4_Go, b, c, d, a, G_wec01, MD4S13); if (MATCHES_NONE_VV (b, pre_b)) continue;
  MD4_STEP0(MD4_Go, a, b, c, d, G_w3c01, MD4S10); if (MATCHES_NONE_VV (a, pre_a)) continue;
  ```

**a0-optimized** (`OpenCL/m01000_a0-optimized.cl`):

- No `MD4_STEP_REV`, no `MD4_STEP_REV1`, no `MATCHES_NONE_VV` anywhere in the file
- The `m01000_s04` kernel (single-hash) has only one early-skip at H[12] using
  `MATCHES_NONE_VS` (scalar vs. scalar comparison, line 254):
  ```c
  MD4_STEP (MD4_H , a, b, c, d, w0[3], MD4C02, MD4S20);
  if (MATCHES_NONE_VS (a, search[0])) continue;
  ```
  This fires after step H[12] (the 13th of 16 H-round steps), skipping only the final
  3 H-round steps — far later than the G-round MITM checks in a3.

**a1-optimized** (`OpenCL/m01000_a1-optimized.cl`):

- No `MD4_STEP_REV`, no reverse pass, no `MATCHES_NONE_VV`
- The `m01000_s04` kernel has the identical late early-skip at H[12] (line 376):
  ```c
  if (MATCHES_NONE_VS (a, search[0])) continue;
  ```

### Verdict: CONFIRMED

`MD4_STEP_REV` appears only in a3-optimized (defined at lines 17–30, used at lines 247–268
for the backward pass). `MATCHES_NONE_VV` (vector vs precomputed-vector comparisons after
G[10], G[11], G[12]) appears only in a3-optimized (lines 322–324). Both a0 and a1 have
only a single late scalar early-skip after H[12], which skips only 3 of the final 4 H-round
steps and provides no savings before the H round.

---

## Claim 4: Vec:2 helps Apple Silicon because it's latency-bound

**Claim:** Vec:2 provides a meaningful uplift on Apple Silicon (latency-hiding) but not on
NVIDIA (already ALU-saturated).

### Benchmark Results

**Local machine — Apple M3 Max (OpenCL backend, GPU)**

| Vec width | Speed | Kernel time | Accel | Loops | Thr |
|-----------|-------|-------------|-------|-------|-----|
| Vec:1 | 14,440.2 MH/s | 6.95 ms | 480 | 1024 | 224 |
| Vec:2 | 15,461.6 MH/s | 7.40 ms | 480 | 1024 | 256 |

Uplift: **+7.1%** (14440 → 15462 MH/s)

**Note on kernel time:** The per-kernel time at Vec:2 (7.40 ms) is slightly *higher* than
Vec:1 (6.95 ms), but each kernel invocation processes 2× the candidates. Effective
throughput-per-ms at Vec:2 is approximately 2×(15462/7.40) = 4179 MH/s·ms⁻¹ vs
Vec:1 (14440/6.95) = 2078 MH/s·ms⁻¹. This is consistent with the latency-hiding
hypothesis: the Apple GPU fills idle pipeline slots with the second candidate's arithmetic.

**dell3 — NVIDIA RTX 3080 Ti (CUDA backend)**

| Vec width | Speed | Kernel time | Accel | Loops | Thr |
|-----------|-------|-------------|-------|-------|-----|
| Vec:1 | 115.0 GH/s | 83.58 ms | 128 | 1024 | 1024 |
| Vec:2 | 114.7 GH/s | 83.84 ms | 128 | 1024 | 1024 |

Uplift: **-0.3%** (essentially flat, within noise).

### Surprising Finding: Metal Is Skipped, OpenCL Is Active

The local benchmark runs on the Apple GPU via **OpenCL** (not Metal). The Metal device
(`Device #01`) is listed as "skipped" in all runs. This means:

1. `USE_BITSELECT` **is** active (OpenCL always defines it), so the Vec:2 uplift is
   measured without the bitselect penalty.
2. The +7.1% uplift is observed on the Apple GPU through the OpenCL path. If the Metal
   path (with disabled bitselect and more ALU operations per step) were tested, the
   absolute throughput would be lower but the Vec:2 uplift might be larger, since more
   ALU work per step means more latency to hide.

### Verdict: CONFIRMED (directionally)

Vec:2 provides +7.1% on Apple Silicon vs essentially 0% on NVIDIA RTX 3080 Ti. This is
consistent with the latency-hiding hypothesis. The test runs on OpenCL rather than Metal
on the local machine, but the underlying Apple GPU silicon is the same. The flat NVIDIA
result confirms that NVIDIA at Vec:1 is already ALU-saturated and does not benefit from
intra-thread vectorisation.

---

## Claim 5: 48 precomputed constants may hurt Apple Silicon occupancy

**Claim:** The 48 precomputed `const u32` values in the a3-optimized inner function
(`m01000m`/`m01000s`) may push register use over an occupancy threshold on Apple Silicon,
where register files are smaller than NVIDIA's.

### Indirect Test Methodology

The investigation plan acknowledges this is hard to prove directly without GPU profiling
tools. The suggested indirect test was to compare a3 vs a0 relative performance and look
for disproportionate behavior. However, the `hashcat -b` benchmark always uses a3 by
default (brute-force mode), and a0 (rules mode) cannot be triggered in benchmark mode
without an actual hash and wordlist.

### What the Benchmark Data Shows

The available benchmark data (Claim 4) covers a3-optimized only (both Vec:1 and Vec:2).
A direct comparison requires a0 in benchmark conditions, which is not provided by `-b`.

From the kernel parameters reported:
- Vec:1: `Accel:480, Loops:1024, Thr:224` — 224 threads per work-group
- Vec:2: `Accel:480, Loops:1024, Thr:256` — 256 threads per work-group

The thread count increases at Vec:2 (224 → 256), which is atypical — normally doubling
the vector width halves the thread count for equivalent work. The increase suggests hashcat
is selecting a larger work-group size at Vec:2. This may indicate the scheduler is finding
better occupancy at Vec:2, consistent with (but not proving) a register-pressure-induced
occupancy cliff at Vec:1.

### Verdict: UNCONFIRMED (insufficient data without profiler or a0 vs a3 direct comparison)

The 48-constant block is confirmed to exist in a3 (Claim 2 evidence), and the occupancy
concern is plausible given Apple Silicon's documented smaller register files. However, no
direct measurement of occupancy or register spill is available from command-line tools.
The unusual thread-count increase from Vec:1 to Vec:2 is a weak indirect signal. Profiling
with Metal Shader profiler or Instruments would be required to confirm or refute.

---

## Summary Table

| Claim | Verdict | Key Evidence |
|-------|---------|--------------|
| 1. Metal disables bitselect on Apple Silicon | CONFIRMED (code); UNCONFIRMED (runtime on this machine) | `inc_vendor.h` lines 196–202: `#ifdef IS_METAL ... #ifndef IS_APPLE_SILICON #define USE_BITSELECT`; Metal device is skipped on local machine |
| 2. a0/a1 lack precomputation that a3 has | CONFIRMED | a3 `m01000_a3-optimized.cl` lines 42–91: 48 precomputed constants; a0 and a1 use all `MD4_STEP` (never `MD4_STEP0`), paying 47 extra adds per candidate |
| 3. MITM only in a3 | CONFIRMED | `MD4_STEP_REV` defined only in a3 (lines 17–30); backward pass at lines 247–268; `MATCHES_NONE_VV` at G[10–12] (lines 322–324); absent from a0 and a1 entirely |
| 4. Vec:2 helps Apple Silicon, not NVIDIA | CONFIRMED | M3 Max: Vec:1 = 14440 MH/s → Vec:2 = 15462 MH/s (+7.1%); RTX 3080 Ti: Vec:1 = 115.0 GH/s → Vec:2 = 114.7 GH/s (−0.3%) |
| 5. 48 constants may hurt Apple Silicon occupancy | UNCONFIRMED | Plausible from code (48 `const u32` before loop in a3); no profiler data available; atypical thread-count increase (224 → 256) at Vec:2 is weak indirect signal |

---

## Key Surprising Findings

1. **Metal is skipped on the local machine.** The investigation plan asserts that on macOS
   Apple Silicon, the alias-dedup logic in `src/backend.c` marks the OpenCL device as
   skipped in favour of Metal. On this M3 Max system, the **opposite** is true: Metal
   (`Device #01`) is skipped and OpenCL (`Device #02`) runs. This means:
   - All local benchmarks run with `USE_BITSELECT` **enabled** (OpenCL always defines it)
   - Claim 1's runtime impact cannot be observed on this machine
   - The ~14–15 GH/s numbers observed here are OpenCL, not Metal

2. **The plan's early-skip description for a3 doesn't match the code.** The plan says MITM
   fires "after G[10] (`check c`), G[11] (`check b`), and G[12] (`check a`)". The actual
   code at lines 322–324 fires after the 11th G step (`c` check after `G_wac01`), 12th G
   step (`b` check after `G_wec01`), and 13th G step (`a` check after `G_w3c01`). These
   are G-round steps 11, 12, and 13 (0-indexed: 10, 11, 12), so the G[10/11/12] labeling
   is correct if using 0-based indexing. The description is accurate.

3. **The `MD4_STEP` macro always does `a += make_u32x(K)` before `hc_add3`.** The a3
   kernel uses `MD4_STEP0` for 47 of 48 steps (the one exception being the variable `w0`
   step), which eliminates one vectorised add per step. The plan's claim of "up to 47 fewer
   operations" is accurate.

4. **dell3 runs at 115 GH/s.** The RTX 3080 Ti result of 115 GH/s at Vec:1 is
   substantially higher than the M3 Max's ~15 GH/s (both via OpenCL). This 7.7× gap is
   consistent with NVIDIA's much higher parallelism and wider vector execution units.
