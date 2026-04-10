# Apple Silicon NTLM Vec:2 Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add tuning database entries so Apple M-series GPUs use Vec:2 for NTLM, delivering +6.3% throughput.

**Architecture:** Two config file edits — add an Apple M-series device alias group in `Alias.hctune`, then add a Vec:2 entry for NTLM in `Modules_default.hctune`. No code changes.

**Tech Stack:** hashcat tuning database (`.hctune` format)

---

### Task 1: Add Apple M-series alias group to Alias.hctune

**Files:**
- Modify: `tunings/Alias.hctune:462` (append after last Intel entry)

- [ ] **Step 1: Add the Apple M alias section**

Append the following block after line 462 (the last line, `Intel(R)_Data_Center_GPU_Max_1100`) in `tunings/Alias.hctune`:

```
##
## Apple Silicon
##

Apple_M1                                        ALIAS_Apple_M
Apple_M1_Pro                                    ALIAS_Apple_M
Apple_M1_Max                                    ALIAS_Apple_M
Apple_M1_Ultra                                  ALIAS_Apple_M
Apple_M2                                        ALIAS_Apple_M
Apple_M2_Pro                                    ALIAS_Apple_M
Apple_M2_Max                                    ALIAS_Apple_M
Apple_M2_Ultra                                  ALIAS_Apple_M
Apple_M3                                        ALIAS_Apple_M
Apple_M3_Pro                                    ALIAS_Apple_M
Apple_M3_Max                                    ALIAS_Apple_M
Apple_M3_Ultra                                  ALIAS_Apple_M
Apple_M4                                        ALIAS_Apple_M
Apple_M4_Pro                                    ALIAS_Apple_M
Apple_M4_Max                                    ALIAS_Apple_M
Apple_M4_Ultra                                  ALIAS_Apple_M
```

The format matches existing entries: device name (spaces replaced with underscores, left-aligned) followed by the alias name (padded to column 49). hashcat matches device names like "Apple M3 Max" by converting spaces to underscores.

- [ ] **Step 2: Verify formatting**

Run: `tail -20 tunings/Alias.hctune`

Expected: The new Apple Silicon section appears after the Intel section, with consistent column alignment matching the rest of the file.

- [ ] **Step 3: Commit**

```bash
git add tunings/Alias.hctune
git commit -m "tunings: add Apple M-series alias group (ALIAS_Apple_M)"
```

---

### Task 2: Add NTLM Vec:2 tuning entry for Apple Silicon

**Files:**
- Modify: `tunings/Modules_default.hctune:191` (insert after last ALIAS_INTEL entry, before CryptoAPI section)

- [ ] **Step 1: Add the NTLM Vec:2 entry**

Insert the following block after line 191 (`ALIAS_INTEL ... 99999`) and before the `## CryptoAPI` section in `tunings/Modules_default.hctune`:

```
ALIAS_Apple_M                                   *       1000    2       A       A
```

Fields: device alias, attack mode (`*` = all), hash type (`1000` = NTLM), vector width (`2`), kernel accel (`A` = auto), kernel loops (`A` = auto).

- [ ] **Step 2: Verify formatting**

Run: `grep -n "Apple_M\|CryptoAPI" tunings/Modules_default.hctune`

Expected: The `ALIAS_Apple_M` line appears on its own line before the `## CryptoAPI` comment, with column alignment matching surrounding entries.

- [ ] **Step 3: Commit**

```bash
git add tunings/Modules_default.hctune
git commit -m "tunings: set Vec:2 for NTLM (mode 1000) on Apple Silicon

Benchmarked on M3 Max (40 GPU cores): Vec:2 delivers +6.3% throughput
(34.4 GH/s vs 32.4 GH/s avg over 3 runs). Vec:4/8 show no improvement."
```

---

### Task 3: Verify the tuning is picked up

- [ ] **Step 1: Build hashcat**

Run: `make clean && make -j$(sysctl -n hw.ncpu)`

Expected: Clean build with no errors.

- [ ] **Step 2: Run NTLM benchmark**

Run: `./hashcat -b -m 1000`

Expected: Output shows `Vec:2` in the speed line (not `Vec:1`). Speed should be ~34.4 GH/s on M3 Max.

- [ ] **Step 3: Confirm no regression on other modes**

Run: `./hashcat -b -m 0` (MD5 — should still use Vec:1 since we only tuned mode 1000)

Expected: Output shows `Vec:1`. Speed should be unchanged from baseline.
