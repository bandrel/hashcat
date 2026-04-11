# NTLM Kernel Optimization Investigation Plan

**Date:** 2026-04-11
**Branch:** ntlm-kernel-optimization
**Scope:** Research and investigation only — no source code changes

---

## Background

NTLM (mode 1000) internally computes MD4 over a UTF-16LE encoded password. The
optimized kernel path handles single-block inputs (password fits in one 64-byte
MD4 block, i.e. up to 27 UTF-8 bytes before expansion). Three attack-mode
kernels exist:

| Kernel | File | Notes |
|--------|------|-------|
| a0 (rules) | `OpenCL/m01000_a0-optimized.cl` | Full 16-word MD4, no meet-in-middle |
| a1 (combinator) | `OpenCL/m01000_a1-optimized.cl` | Full 16-word MD4, no meet-in-middle |
| a3 (brute-force) | `OpenCL/m01000_a3-optimized.cl` | Meet-in-middle + early-skip in Round 2 |

The shared round-function macros live in `OpenCL/inc_hash_md4.h`. Hardware
`bitselect` is enabled for Round 1 (F) and Round 2 (G) via `USE_BITSELECT`,
which is set for CUDA, HIP, and OpenCL on all platforms, but **not** for Metal
on Apple Silicon (see `IS_APPLE_SILICON` guard in `OpenCL/inc_vendor.h`).

The `hc_add3` three-operand add is expressed as `a + b + c`; on AMD GCN it
could be `V_ADD3_U32` (the inline asm path is currently commented out). On CUDA
it falls through to plain C addition (the compiler typically folds these). On
Apple Silicon it is also plain C addition.

The core MD4 step macro is:

```c
#define MD4_STEP(f,a,b,c,d,x,K,s) { \
  a += make_u32x(K);                 \
  a  = hc_add3(a, x, f(b,c,d));     \
  a  = hc_rotl32(a, s);             \
}
```

Observations grounded in the source:

- `a3-optimized` precomputes all 48 `w[i] + constant` sums before the inner
  loop (see `F_w*c00`, `G_w*c01`, `H_w*c02` constants), reducing additions
  inside the per-candidate hot path.
- `a0-optimized` and `a1-optimized` do **not** precompute these sums — every
  `MD4_STEP` call with a non-zero word pays an extra add inside the loop.
- The meet-in-the-middle early-skip in `a3-optimized` fires after step G[10]
  (`check c`), G[11] (`check b`), and G[12] (`check a`) — three mid-round
  comparisons before the H round even starts.
- The early-skip in `a0-optimized` and `a1-optimized` fires only after H[12]
  (checking only `a`), which is much later, skipping only three of the last
  four H-round steps.
- Metal on Apple Silicon does not define `USE_BITSELECT`, so the F and G round
  functions fall back to the multi-operation forms:
  - `MD4_F`: `(x & y) | (~x & z)` — 3 ops (AND, ANDN, OR)
  - `MD4_Go` via `bitselect(x, y, x^z)` path: unavailable → `MD4_G`: `(x&y)|(x&z)|(y&z)` — 5 ops
  - `bitselect` path: `bitselect(z, y, x)` — 1 op (fused)
- The alias-dedup logic in `src/backend.c` marks the OpenCL device as **skipped**
  in favour of Metal on macOS (`alias_device->is_metal == true` survives). This
  means current benchmark numbers at ~32 GH/s and ~34.4 GH/s are **Metal**, not
  OpenCL.
- SIMD vectorisation (`u32x`) packs multiple candidates per work-item. At
  `Vec:2` on Apple Silicon, each thread processes two candidates simultaneously.
  The +6.6% uplift at Vec:2 vs Vec:1 on the M3 Max suggests the Apple GPU has
  some spare ALU capacity at Vec:1 that can be filled with intra-thread
  vectorisation, while NVIDIA at Vec:1 is already ALU-saturated.

---

## 1. Profiling Methodology

### 1.1 CUDA — Nsight Compute (ncu)

**Goal:** Obtain cycle-accurate instruction-level metrics for `m01000_m04` and
`m01000_s04` kernels on the RTX 3080 Ti.

**Setup:**

```bash
# Isolate a single benchmark pass, use -n to limit hash count
ncu --set full \
    --target-processes application-only \
    --kernel-name m01000_m04 \
    -o ntlm_m04_profile \
    hashcat -m 1000 -a 3 example0.hash ?a?a?a?a --benchmark-all -n 1
```

**Metrics to collect:**

| Category | Metric | Expected insight |
|----------|--------|-----------------|
| ALU | `sm__inst_executed_pipe_alu` | Integer utilisation |
| ILP | `sm__inst_executed` vs `sm__cycles_active` | Instructions per cycle |
| Occupancy | `sm__warps_active` / `sm__warps_eligible` | Register pressure limiting occupancy |
| Register use | `launch__registers_per_thread` | If high (>32), investigate spilling |
| L1 cache | `l1tex__t_sectors_pipe_lsu_mem_local_op_ld` | Register spill reads |
| Warp stalls | `scheduler__stalls_*` | Identify bottleneck type |
| Throughput | `sm__throughput` | Overall ALU pipe utilisation |

**Key questions for CUDA:**
- Is the kernel compute-bound or latency-bound? (`smsp__cycles_active` vs `smsp__cycles_elapsed`)
- What is the achieved vs theoretical occupancy? If register pressure is
  limiting occupancy, reducing the constant-precomputation block in `m01000m`
  (48 scalar `u32` constants on the stack) may be worth trading against re-computation.
- Are there warp divergence stalls at the `if (MATCHES_NONE_VS)` early-skip
  branches in `a0-s04` and `a3-s04`?
- Does `lop3.b32` get emitted for the XOR-based H round? (`sm__inst_executed_pipe_lsu` for logic)
- With Vec:1 being equal to Vec:2 on NVIDIA, verify via `launch__grid_size` that
  the GPU is already fully occupied — vectorisation provides no benefit because
  it merely reduces the thread count by 2x without reclaiming any ALU cycles.

**Tools:** Nsight Compute 2024+, `ncu-ui` for visual waterfall, Nsight Systems
(`nsys`) for timeline overhead characterisation.

---

### 1.2 Metal — Xcode GPU Frame Capture / Metal System Trace

**Goal:** Profile the translated OpenCL → Metal kernel on Apple M3 Max. Because
hashcat uses OpenCL on macOS (the OpenCL device is skipped in favour of Metal
by alias-dedup), the pipeline being profiled is the Metal-compiled version of
the OpenCL `.cl` source, translated via the `clspv` or Apple's own CL-to-MSL
translator.

**Setup:**

1. Launch hashcat under Xcode's "Metal System Trace" template
   (`Instruments → Metal System Trace`).
2. Alternatively use `xcrun xctrace record --template 'Metal System Trace'
   --attach <pid>`.
3. For GPU shader profiling: Instruments → GPU Frame Capture (requires entitlement
   or debug build; hashcat may need to be built with `-g` and codesigned with
   `com.apple.security.get-task-allow`).

**Metrics to collect:**

| Metric | Tool | Insight |
|--------|------|---------|
| GPU Active time | Metal System Trace → GPU timeline | Wall-clock efficiency |
| ALU Active / % | GPU Frame Capture → Shader profiler | Whether compute-bound |
| Occupancy | Shader profiler → Wave occupancy | Threadgroup utilisation |
| Register allocation | Shader compiler IR dump (see below) | Spill to threadgroup mem? |
| Memory bandwidth | Metal System Trace → Memory | Unexpected reads (spill, input buffer) |

**Shader IR dump:**

```bash
# Compile the OpenCL kernel to Metal IR to inspect register use
xcrun metalfe -x cl m01000_a3-optimized.cl -o m01000.air
xcrun dis m01000.air   # or: llvm-dis m01000.air -o m01000.ll
```

Check for `alloca` in the generated LLVM IR as evidence of register spilling.

**Key questions for Metal:**
- Does the Apple GPU AIR compiler lower `hc_add3(a, b, c)` to a fused 3-operand
  add? Apple Silicon has a native 3-source integer add in the GPU ISA
  (`iadd3` in the undocumented ISA); the C compiler path `a + b + c` may or may
  not fuse it.
- The 48-constant precomputation block in `m01000m` allocates 48 scalar `u32`
  registers before the inner loop. Apple GPU register files are smaller than
  NVIDIA's — investigate whether this pushes the shader over the occupancy cliff.
- Since `USE_BITSELECT` is disabled on `IS_APPLE_SILICON` (Metal path), the F
  and G round functions emit more instructions. Measure the actual instruction
  count difference and estimate the ALU cost.
- Is there overhead from the Apple OpenCL-to-Metal translation layer (clspv or
  Apple's proprietary translator)? Compare binary sizes and instruction counts
  between the OpenCL-submitted and a hypothetical hand-written MSL kernel.

---

### 1.3 OpenCL — clprofiler / cl_khr_profiling_events

**Goal:** Profile OpenCL kernel execution time with event-based timing as a
complement to hardware counters.

**Setup:**

```c
// Force CL_QUEUE_PROFILING_ENABLE in backend.c (for investigation only)
// Measure CL_PROFILING_COMMAND_START / COMMAND_END per enqueue
```

Or use:
```bash
# AMD GPU / OpenCL: use ROCm profiler (rocprof)
rocprof --stats hashcat -m 1000 -a 3 ...

# Intel OpenCL: use Intel VTune
vtune -collect gpu-hotspots -- hashcat -m 1000 ...

# NVIDIA OpenCL: Nsight (same as CUDA path above, OpenCL kernels appear similarly)
```

**Key questions for OpenCL:**
- On macOS, is there any measurable overhead from the OpenCL → Metal translation
  path vs. native Metal dispatch? Use `clGetEventProfilingInfo` event timestamps
  vs. Metal timestamp queries.
- For AMD GPUs: is `V_ADD3_U32` being generated? The code has the inline asm
  commented out in `inc_common.cl` — this is a potential 33% ALU reduction for
  the 3-op add in every MD4 step.

---

## 2. Kernel-Level Optimization Candidates

### 2.1 Constant Precomputation in a0 and a1 Kernels

**Current state:** `a0-optimized` and `a1-optimized` use plain `MD4_STEP` with
live word values and `MD4C00`/`MD4C01`/`MD4C02` constants. Every step does:

```c
a += make_u32x(K);          // add constant K into a (extra add per step)
a = hc_add3(a, x, f(b,c,d)); // then 3-operand add
```

**a3-optimized** precomputes `F_wNc00 = w[N] + MD4C00` for all 16 words before
the loop, then uses `MD4_STEP0` for non-varying words:

```c
a = hc_add3(a, K_plus_w_precomputed, f(b,c,d));  // 1 less add per non-varying step
```

**Investigation question:** How much does adding the 48-constant precomputation
block to `a0` and `a1` gain, and does it cost occupancy? In `a0`, the `w[]`
array is only partially constant (w[0] varies due to rule application on the
first word), but w[1]–w[15] are known before the loop. The pattern used in `a3`
should be directly applicable.

**Estimated ALU savings (a3 pattern as baseline):**
- 15 of 16 Round 1 steps can use `MD4_STEP0` (no `make_u32x(K)` call)
- 16 of 16 Round 2 steps (G round uses `w0` and precomputed words)
- 16 of 16 Round 3 steps

That is up to 47 fewer `make_u32x(K)` + add operations per inner-loop
iteration in `a0` and `a1`. On a 48-step MD4 with 3 operations per step, this
is a non-trivial fraction of total work.

---

### 2.2 Meet-in-the-Middle for a0 (Rules) and a1 (Combinator)

**Current state:** Meet-in-the-middle (MITM) is implemented only in `a3`. The
`a0` single-hash (`s04`) kernel does include a **partial early-skip** at H[12]:

```c
MD4_STEP (MD4_H, a, b, c, d, w0[3], MD4C02, MD4S20);  // H[12]
if (MATCHES_NONE_VS (a, search[0])) continue;           // skip if a doesn't match
MD4_STEP (MD4_H, d, a, b, c, w2[3], MD4C02, MD4S21);
MD4_STEP (MD4_H, c, d, a, b, w1[3], MD4C02, MD4S22);
MD4_STEP (MD4_H, b, c, d, a, w3[3], MD4C02, MD4S23);
```

**The full MITM in `a3-s`** adds:
1. A one-time backward pass (outside the loop) from the target digest, peeling
   the entire H round and two G steps to derive `a_rev`, `b_rev`, `c_rev`,
   `d_rev`, `sav_c`, `sav_d`.
2. Inside the loop, 3 partial-state comparisons after G[10], G[11], G[12]:
   ```c
   if (MATCHES_NONE_VV(c, pre_c)) continue;  // after G[10]
   if (MATCHES_NONE_VV(b, pre_b)) continue;  // after G[11]
   if (MATCHES_NONE_VV(a, pre_a)) continue;  // after G[12]
   ```

**Investigation question for a0:** In the rules attack, `w[0]` is the only word
that varies between iterations (the rule modifies the base password, but the
result is stored in `w0[0]`–`w1[3]`, not a single word). The MITM backward pass
still depends on only the target digest (constant), so `a_rev` through `sav_d`
can still be precomputed once. The per-iteration `pre_a/b/c` computation depends
on `w0` (the vectorised varying word). Investigate whether the MITM G[10]–G[12]
check structure is directly portable to `a0-s04`.

**Constraint:** The MITM only benefits the single-hash `_s` kernels
(where `search[]` is fixed). Multi-hash kernels (`_m`) must test against a
hash list and cannot short-circuit the same way.

---

### 2.3 MD4 Round 3 (H) — XOR vs Bitselect

**Current state:** Round 3 uses `MD4_H(x,y,z) = x ^ y ^ z`. This is 2 XOR
operations. There is no `bitselect` optimisation applied (`MD4_H` is used
directly, not via a `MD4_Ho` macro). On CUDA, `lop3.b32` with immediate 0x96
implements `x ^ y ^ z` in one instruction. On Apple Silicon Metal (no
`USE_BITSELECT` guard applies to H), the compiler should still fold two XORs.

**Investigation question:** Is there any architecture where the H-round XOR
chain is generating more than 2 ALU instructions? Check the PTX / AIR / LLVM IR
output. Specifically:
- CUDA: does ptxas emit `lop3.b32` for the chained XOR in `MD4_H`?
- Metal AIR: does the Apple compiler fuse `x ^ y ^ z` into a single ternary op?

This is lower priority than 2.1 and 2.2 but worth verifying in the profiler.

---

### 2.4 Instruction-Level Parallelism (ILP) and Dependency Chains

**Current state:** Each MD4 step produces a new value of `a` (rotated) which
is used as input `b` in the next step (after rotating through `a, b, c, d`).
This creates a 4-deep recurrence: each step depends on the output of the step
4 steps earlier. This is the fundamental serial dependency in MD4 and cannot be
broken without algorithmic change.

**Investigation question:** Are there instruction scheduling opportunities
within a step? The current macro expands to:

```
a += K
a = a + x + f(b,c,d)     # f computed in parallel with a += K possible?
a = rotl(a, s)
```

On architectures with out-of-order issue (CUDA, AMD GCN), the compiler should
handle this. On the in-order Apple GPU, manual reordering could help. Inspect
the compiled shader to verify the scheduler is not stalling on the dependency
between `a += K` and the subsequent `hc_add3`. Specifically, `a += K` and
`f(b,c,d)` are independent and could execute in parallel.

**Profiling signal:** High `scheduler__stalls_long_scoreboard` in ncu, or high
"ALU stall" in Metal Shader profiler, would confirm this as a bottleneck.

---

### 2.5 Register Pressure and the 48-Constant Precomputation Block

**Current state:** The `m01000m` / `m01000s` functions in `a3-optimized`
allocate 48 `const u32` values on the stack before the inner loop:
`F_w*c00` (16 values), `G_w*c01` (16 values), `H_w*c02` (16 values).

These are compile-time-visible as `const` and the compiler should promote them
to registers. On an architecture with limited registers per thread (Apple Silicon
reportedly has 32–64 physical registers per SIMD lane), 48 scalars plus the 4
state registers `a, b, c, d` plus the vectorised `w0` register may push total
register use over the occupancy threshold.

**Investigation steps:**
1. Compile with `--ptxas-options=--verbose` (CUDA) to get register count per
   thread.
2. For Metal, use the `metalfe` + `dis` approach above to count allocations.
3. Compare register count at Vec:1 vs Vec:2 — doubling vector width doubles the
   state registers but not the constant pool.
4. Consider whether a "lazy precomputation" strategy (precompute only G and H
   constants, since F constants are used only once each) reduces peak register
   pressure without sacrificing the ILP benefit.

---

### 2.6 Apple GPU Architecture-Specific Opportunities

**Tile-based deferred rendering (not relevant) / Unified memory:**

Apple Silicon uses a unified memory architecture where the GPU and CPU share
physical DRAM. This means:
- There is no PCIe transfer cost — candidate buffers (`pws`) and digest buffers
  (`digests_buf`) are accessible without explicit DMA.
- Buffer bandwidth is shared with the CPU. During a benchmark, investigate
  whether CPU activity (dictionary I/O, rule parsing) competes for memory
  bandwidth with the GPU hash loop.

**Apple GPU SIMD width:**

Apple GPU SIMD groups (analogous to NVIDIA warps) have a fixed width of 32
threads. The Metal preferred work-group size multiple
(`hc_mtlGetThreadExecutionWidth`, see `src/backend.c:9359`) reports this value.
If the kernel is dispatched with a threadgroup size that is not a multiple of 32,
some SIMD groups will be partially occupied.

**Investigation question:** What threadgroup size does hashcat use for the NTLM
kernel on Metal? Is it 32, 64, or 256? Larger threadgroups improve GPU
utilisation for memory-latency-bound work but may reduce occupancy if register
pressure is high.

**No warp-shuffle equivalent on Apple GPU Metal:**

NVIDIA's `__shfl_*` warp-level primitives enable cross-lane reductions without
shared memory. Metal has `simd_sum`, `simd_broadcast_first`, etc. For the hash
inner loop these are not currently used and are likely not applicable, but if a
future tree-reduction comparison across candidates is considered, this difference
matters.

---

### 2.7 CUDA-Specific Opportunities

**Funnel shift (`SHFL` + `SHR`):**

The `hc_rotl32` macro uses `USE_FUNNELSHIFT` when `HAS_SHFW == 1`. Verify that
the PTX output uses `shf.l.wrap.b32` for all 12 distinct rotate amounts in MD4
(3, 7, 11, 13, 15, 17, 19, 23 — MD4 uses MD4S00–MD4S23). If any rotate is not
using the funnel-shift path, it should be investigated.

**lop3.b32 for F and G rounds:**

The F round is `(b & c) | (~b & d)` which maps to `lop3.b32 %r, %b, %c, %d,
0xCA`. The G round via `bitselect(b, c, b^d)` should map to `lop3.b32 %r, %b,
%c, %d, 0xE8`. The H round `b ^ c ^ d` is `lop3.b32 %r, %b, %c, %d, 0x96`.
All three should be expressible as a single PTX instruction. Verify via PTX
dump:

```bash
hashcat --opencl-info  # (or cuda equivalent) to identify device
# or compile directly:
nvcc -ptx -arch=sm_86 m01000_a3-optimized.cl -o m01000.ptx
grep "lop3\|shf\|setp" m01000.ptx | head -30
```

**Shared memory for digest comparison (multi-hash only):**

In the multi-hash `_m` kernels, `digests_buf` and the bitmap arrays are read
once per candidate. If the inner loop over `IL_CNT` is long, these reads are
amortised. For short wordlists, the `COMPARE_M_SIMD` bitmap lookup may be a
memory bottleneck. Shared memory preloading of the bitmap slices could be
investigated.

**Warp-level early exit coordination:**

In `a3-s04` and `a0-s04`, the `if (MATCHES_NONE_VS(a, search[0])) continue`
creates divergence: threads that skip skip the last 3 H-round steps plus
`COMPARE_S_SIMD`, while others do not. On NVIDIA, divergent warps serialise.
Investigate whether a `__ballot_sync` + `__all_sync` guard can reduce the cost
of the divergent branch when very few threads hit the early-exit condition
(typical case during a benchmark).

---

## 3. Backend-Specific Investigations

### 3.1 Metal vs OpenCL on Apple Silicon: What Is Actually Running?

**Confirmed from source:** The alias-dedup logic in `src/backend.c:406–410`
marks the OpenCL device as skipped when a Metal device is the alias:

```c
#if defined (__APPLE__)
// this lets Metal devices survive over OpenCL
if (alias_device->is_metal == true) continue;
#endif
alias_device->skipped = true;
```

This means on Apple Silicon (macOS), all NTLM hashing runs through the Metal
backend, not OpenCL. The Metal kernel is compiled from the same `.cl` source
files but via Apple's OpenCL-to-Metal translator.

**Investigation questions:**

1. **Translation fidelity:** Does Apple's CL-to-Metal translator preserve the
   `USE_BITSELECT` / `USE_ROTATE` macros, or does it substitute equivalents?
   The `inc_vendor.h` guard `IS_APPLE_SILICON` disables `USE_BITSELECT` for
   Metal — does the translated Metal shader therefore emit 3–5 instructions for
   F/G rounds instead of 1?

2. **Two-stage vs single-stage compilation:** The context references
   "single-stage compilation" for Apple Silicon OpenCL workaround. Locate the
   relevant code path in `src/backend.c` (search for `opencl_driver_version` or
   `single_stage`) to understand whether this affects kernel compilation speed
   (startup latency) or runtime performance.

3. **Preferred work-group size multiple:** For Metal, `hc_mtlGetThreadExecutionWidth`
   returns the SIMD width (32 for Apple Silicon). For OpenCL on Apple Silicon,
   `CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE` may return a different value
   due to the translation layer. Compare these to understand whether threadgroup
   sizes differ between Metal and OpenCL dispatch.

4. **Occupancy ceiling:** The comment at `src/backend.c:10489` reads "apple
   hack, but perhaps also an alternative for other vendors" around
   `kernel_preferred_wgs_multiple` fallback logic. Investigate whether this
   causes suboptimal dispatch sizes for the NTLM kernel.

5. **Direct Metal shader experiment:** Write a minimal Metal Shading Language
   `.metal` file implementing MD4 for NTLM as a control experiment. This
   bypasses the CL translation layer and allows measuring the upper bound of
   Metal performance. Compare against the CL-translated version.

---

### 3.2 Why Vec:2 Helps Apple Silicon (+6.6%) but Not NVIDIA (0%)

**Hypothesis grounded in source:** At Vec:1, each work-item processes one
candidate per invocation of `m01000m/s`. At Vec:2, `u32x` expands to `uint2`
(or similar) and the compiler processes two candidates simultaneously using
SIMD vector lanes.

On **Apple Silicon**, the GPU SIMD width is 32. With Vec:1, each thread is 1
candidate and the GPU needs 32 threads per SIMD group. With Vec:2, the same
work is done with 16 threads, but each thread does twice the arithmetic. If the
arithmetic at Vec:1 does not fully saturate the SIMD lane's ALU (i.e., the
kernel is latency-bound waiting on dependencies), then doubling the arithmetic
per thread can hide more latency and improve throughput.

On **NVIDIA** at Vec:1, the RTX 3080 Ti has enough warps in flight to fully
saturate the ALU through latency hiding. Vec:2 halves the thread count, which
may reduce occupancy (fewer warps for hiding latency), leaving throughput flat.

**Investigation:** Measure achieved occupancy at Vec:1 vs Vec:2 on both
platforms to confirm or refute this hypothesis. Use `ncu` for NVIDIA and Metal
Shader profiler for Apple Silicon.

---

### 3.3 OpenCL Compatibility Layer Overhead

The `src/backend.c:2289` comment notes: "with apple GPU clEnqueueWriteBuffer()
return CL_INVALID_VALUE, workaround" — the data transfer is done in 16-byte
chunks to work around an Apple OpenCL driver bug. This affects data transfer
time, not kernel execution time, but it is worth measuring:

1. Profile `clEnqueueWriteBuffer` call time vs equivalent Metal buffer update.
2. Measure how much of wall-clock time is kernel execution vs. host-to-device
   transfers for a typical benchmark run (`hashcat -b -m 1000`).

---

## 4. Benchmarking Methodology

### 4.1 Controlling for Thermal State

GPU performance on both Apple Silicon and NVIDIA degrades under sustained load
due to thermal throttling. A single-trial measurement (as noted in the baseline
numbers) is not reliable for regression detection.

**Protocol:**
1. Run a 2-minute "warmup" pass to bring the GPU to steady-state temperature
   before recording any numbers: `hashcat -b -m 1000 --benchmark-all`.
2. Allow 5 minutes of idle between comparison runs.
3. On Apple Silicon: monitor GPU temperature via `sudo powermetrics --samplers
   gpu_power -n 5 -i 1000` or `iStatMenus`. Discard runs where GPU temp exceeds
   85°C.
4. On NVIDIA: monitor with `nvidia-smi -q -d TEMPERATURE` and discard runs
   above 83°C.

### 4.2 Statistical Significance

A minimum of 10 repeated benchmark runs per configuration. Compute:
- Mean
- Standard deviation
- 95% confidence interval (use t-distribution, n=10 is small)

A change is only reportable if the CI does not overlap between baseline and
optimised. For expected gains of ~5–10%, with typical standard deviation of 1–2%
of the mean, 10 runs should be sufficient to detect a 5% difference at 95%
confidence.

**Tooling suggestion:** Collect raw H/s output lines from hashcat's `--machine-
readable` flag and process with a simple Python script.

```bash
hashcat -b -m 1000 --machine-readable 2>/dev/null | grep "^1000:" | awk -F: '{print $4}'
```

### 4.3 Controlling for System Load

On Apple Silicon (unified memory), background system activity competes with the
GPU for memory bandwidth.

- Benchmark with no other applications running.
- Disable Spotlight indexing and Time Machine during benchmarks.
- On macOS: use `sudo launchctl unload /System/Library/LaunchDaemons/
  com.apple.metadata.mds.plist` to stop Spotlight temporarily (reload after).

### 4.4 Vec Size Sweep

For each kernel change, benchmark across all supported `VECT_SIZE` values:

| Platform | Vec sizes to test |
|----------|------------------|
| Apple Silicon (Metal) | 1, 2, 4 (Metal caps at 4 per `inc_vendor.h` comment) |
| NVIDIA RTX 3080 Ti | 1, 2, 4, 8 |

Record the optimal Vec size for each kernel variant — it may change when the
instruction mix changes.

### 4.5 Attack Mode Coverage

Benchmark all three attack modes separately, as their kernels differ
significantly:

```bash
hashcat -b -m 1000                          # uses a3-optimized (benchmark default)
hashcat -m 1000 -a 0 hash.txt wordlist.txt  # a0-optimized
hashcat -m 1000 -a 1 hash.txt w1.txt w2.txt # a1-optimized
```

Any optimisation applied to `a0` or `a1` must be validated on those kernels
specifically, not inferred from `a3` benchmark numbers.

### 4.6 Regression Testing

Before and after any kernel change, run the hashcat self-test for mode 1000:

```bash
hashcat -m 1000 b4b9b02e6f09a9bd760f388b67351e2b --self-test-disable --quiet \
  -a 3 ?l?l?l?l?l?l?l
# expected: hashcat (v7.x) -> b4b9b02e6f09a9bd760f388b67351e2b:hashcat
```

The self-test vector is `hashcat:b4b9b02e6f09a9bd760f388b67351e2b` (from
`src/modules/module_01000.c`). Also run the built-in `--self-test` flag:

```bash
hashcat --self-test -m 1000
```

---

## 5. Prioritised Investigation Order

Based on expected impact and code-reading complexity:

| Priority | Investigation | Kernels | Expected Gain |
|----------|---------------|---------|---------------|
| 1 | Constant precomputation in a0/a1 (section 2.1) | a0, a1 | ~5–15% on all backends |
| 2 | MITM for a0-s/a1-s (section 2.2) | a0-s, a1-s | ~10–20% single-hash only |
| 3 | `USE_BITSELECT` on Apple Silicon Metal (section 2.6 / 3.1) | all, Metal | ~3–8% |
| 4 | Vec:2 mechanism analysis (section 3.2) | all | Understand, not change |
| 5 | Register pressure audit for 48-constant block (section 2.5) | a3, a0+, a1+ | Prerequisite for (1) |
| 6 | CUDA lop3 / funnel-shift verification (section 2.7) | a3 CUDA | Confirm existing |
| 7 | OpenCL layer overhead on macOS (section 3.3) | all Apple | Characterise |
| 8 | Direct Metal shader control experiment (section 3.1 item 5) | Metal | Upper bound |

---

## 6. Files to Read for Each Investigation

| Investigation | Primary files |
|---------------|--------------|
| Constant precomputation | `OpenCL/m01000_a0-optimized.cl`, `m01000_a3-optimized.cl`, `OpenCL/inc_hash_md4.h` |
| MITM portability | `OpenCL/m01000_a3-optimized.cl` (lines 241–346), `m01000_a0-optimized.cl` |
| bitselect on Metal | `OpenCL/inc_vendor.h` (lines 196–208), `OpenCL/inc_hash_md4.h` (lines 17–23) |
| Register pressure | `OpenCL/m01000_a3-optimized.cl` (lines 40–90), ptxas/metalfe output |
| CUDA lop3/funnel | `OpenCL/inc_common.cl` (lines 1978–2050), PTX output |
| Backend alias logic | `src/backend.c` (lines 355–432) |
| Metal compilation | `src/ext_metal.m`, `src/backend.c` (Metal section) |
| Threadgroup sizing | `src/backend.c` (lines 9359–9412) |
