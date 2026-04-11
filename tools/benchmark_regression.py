#!/usr/bin/env python3
"""
Hashcat benchmark regression test harness.

Runs benchmarks across CUDA/OpenCL/Metal backends for specified hash modes
and Vec widths, collecting statistically meaningful data for comparison.

Usage:
    # Quick smoke test (3 trials)
    python3 tools/benchmark_regression.py --trials 3

    # Full regression test (30 trials)
    python3 tools/benchmark_regression.py --trials 30

    # Compare against a baseline
    python3 tools/benchmark_regression.py --trials 10 --baseline results/baseline.json

    # Test specific modes
    python3 tools/benchmark_regression.py --modes 0,1000,900 --trials 10

    # Test specific device
    python3 tools/benchmark_regression.py --device 1 --trials 10
"""

import argparse
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Default modes to test: all modes known to benefit from Vec:2 on Apple Silicon
DEFAULT_MODES = [0, 10, 11, 20, 22, 30, 40, 60, 70, 900, 1000, 1100, 2600]

# Vec widths to test
VEC_WIDTHS = [1, 2, 4]

def find_hashcat():
    """Find hashcat binary."""
    for path in ["./hashcat", "hashcat", "/usr/local/bin/hashcat"]:
        try:
            r = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=10)
            if r.returncode == 0:
                return path
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None

def get_system_info(hashcat_bin):
    """Collect system information for the report."""
    info = {"timestamp": datetime.now().isoformat(), "hashcat_binary": hashcat_bin}

    r = subprocess.run([hashcat_bin, "--version"], capture_output=True, text=True, timeout=10)
    info["hashcat_version"] = r.stdout.strip()

    r = subprocess.run([hashcat_bin, "-I"], capture_output=True, text=True, timeout=30)
    info["backend_info"] = r.stdout + r.stderr

    # Detect backends
    backends = []
    combined = r.stdout + r.stderr
    if "CUDA" in combined:
        backends.append("CUDA")
    if "Metal" in combined:
        backends.append("Metal")
    if "OpenCL" in combined:
        backends.append("OpenCL")
    info["backends"] = backends

    # Extract device names
    devices = re.findall(r"Name\.+:\s*(.*)", combined)
    info["devices"] = [d.strip() for d in devices]

    return info

def run_benchmark(hashcat_bin, mode, vec_width, device=None):
    """Run a single benchmark and return speed in MH/s (normalized)."""
    cmd = [hashcat_bin, "-b", "-m", str(mode), f"--backend-vector-width={vec_width}"]
    if device is not None:
        cmd.extend(["-d", str(device), "--force"])

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        combined = r.stdout + r.stderr

        speed_match = re.search(r"Speed[^:]*:\s*([\d.]+)\s*([kMGT]?H/s)", combined)
        if not speed_match:
            return None

        value = float(speed_match.group(1))
        unit = speed_match.group(2)

        # Normalize to MH/s
        multipliers = {"H/s": 1e-6, "kH/s": 1e-3, "MH/s": 1, "GH/s": 1e3, "TH/s": 1e6}
        return value * multipliers.get(unit, 1)

    except subprocess.TimeoutExpired:
        return None

def get_mode_name(hashcat_bin, mode):
    """Get the algorithm name for a hash mode."""
    try:
        r = subprocess.run([hashcat_bin, "--hash-info", "-m", str(mode)],
                          capture_output=True, text=True, timeout=30)
        match = re.search(r"Name\.+:\s*(.*)", r.stdout)
        return match.group(1).strip() if match else f"Mode {mode}"
    except subprocess.TimeoutExpired:
        return f"Mode {mode}"

def welch_t_test(a, b):
    """Perform Welch's t-test. Returns (t_stat, df, significant_at_001)."""
    if len(a) < 2 or len(b) < 2:
        return 0, 0, False
    n1, n2 = len(a), len(b)
    m1, m2 = statistics.mean(a), statistics.mean(b)
    s1, s2 = statistics.stdev(a), statistics.stdev(b)
    if s1 == 0 and s2 == 0:
        return float('inf') if m2 != m1 else 0, n1 + n2 - 2, m2 != m1
    se = math.sqrt(s1**2/n1 + s2**2/n2)
    if se == 0:
        return 0, 0, False
    t = (m2 - m1) / se
    df = (s1**2/n1 + s2**2/n2)**2 / ((s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1))
    # t > 2.66 with df > 20 => p < 0.01
    significant = abs(t) > 2.66 and df > 20
    return t, df, significant

def run_regression_test(hashcat_bin, modes, trials, device=None, verbose=True):
    """Run full regression test suite."""
    results = {}

    for mode in modes:
        name = get_mode_name(hashcat_bin, mode)
        if verbose:
            print(f"\n--- Mode {mode}: {name} ---")

        results[str(mode)] = {"name": name, "vec_results": {}}

        for vec in VEC_WIDTHS:
            speeds = []
            for i in range(trials):
                speed = run_benchmark(hashcat_bin, mode, vec, device)
                if speed is not None:
                    speeds.append(speed)
                if verbose:
                    status = f"{speed:.1f} MH/s" if speed else "FAILED"
                    print(f"  Vec:{vec} trial {i+1}/{trials}: {status}")

            if speeds:
                results[str(mode)]["vec_results"][str(vec)] = {
                    "speeds": speeds,
                    "mean": statistics.mean(speeds),
                    "stdev": statistics.stdev(speeds) if len(speeds) > 1 else 0,
                    "min": min(speeds),
                    "max": max(speeds),
                    "n": len(speeds),
                }

    return results

def compare_results(results, baseline):
    """Compare current results against a baseline."""
    report = []
    regressions = []

    for mode, data in results.items():
        if mode not in baseline:
            continue

        for vec, current in data["vec_results"].items():
            if vec not in baseline[mode]["vec_results"]:
                continue

            base = baseline[mode]["vec_results"][vec]
            cur_mean = current["mean"]
            base_mean = base["mean"]
            pct_change = (cur_mean - base_mean) / base_mean * 100

            t, df, sig = welch_t_test(base["speeds"], current["speeds"])

            entry = {
                "mode": mode,
                "name": data["name"],
                "vec": vec,
                "baseline_mean": base_mean,
                "current_mean": cur_mean,
                "pct_change": pct_change,
                "t_stat": t,
                "df": df,
                "significant": sig,
            }
            report.append(entry)

            # Flag regressions: >2% slower AND statistically significant
            if pct_change < -2 and sig:
                regressions.append(entry)

    return report, regressions

def print_summary(results, system_info):
    """Print a summary table of results."""
    print("\n" + "=" * 100)
    print(f"BENCHMARK RESULTS — {system_info.get('hashcat_version', 'unknown')}")
    print(f"Devices: {', '.join(system_info.get('devices', ['unknown']))}")
    print(f"Backends: {', '.join(system_info.get('backends', ['unknown']))}")
    print(f"Timestamp: {system_info.get('timestamp', 'unknown')}")
    print("=" * 100)

    print(f"\n{'Mode':<7} {'Name':<40} {'Vec1 (MH/s)':>14} {'Vec2 (MH/s)':>14} {'V2 vs V1':>9} {'Vec4 (MH/s)':>14} {'V4 vs V1':>9}")
    print("-" * 110)

    for mode, data in sorted(results.items(), key=lambda x: int(x[0])):
        name = data["name"]
        v1 = data["vec_results"].get("1", {})
        v2 = data["vec_results"].get("2", {})
        v4 = data["vec_results"].get("4", {})

        v1_mean = v1.get("mean", 0)
        v2_mean = v2.get("mean", 0)
        v4_mean = v4.get("mean", 0)

        v2_pct = (v2_mean - v1_mean) / v1_mean * 100 if v1_mean else 0
        v4_pct = (v4_mean - v1_mean) / v1_mean * 100 if v1_mean else 0

        v1_str = f"{v1_mean:>10.1f} ±{v1.get('stdev', 0):>4.0f}" if v1 else "         N/A"
        v2_str = f"{v2_mean:>10.1f} ±{v2.get('stdev', 0):>4.0f}" if v2 else "         N/A"
        v4_str = f"{v4_mean:>10.1f} ±{v4.get('stdev', 0):>4.0f}" if v4 else "         N/A"

        print(f"{mode:<7} {name:<40} {v1_str} {v2_str} {v2_pct:>+8.1f}% {v4_str} {v4_pct:>+8.1f}%")

def main():
    parser = argparse.ArgumentParser(description="Hashcat benchmark regression test harness")
    parser.add_argument("--trials", type=int, default=3, help="Number of trials per configuration (default: 3)")
    parser.add_argument("--modes", type=str, default=None, help="Comma-separated hash modes to test (default: all known Vec:2 beneficiaries)")
    parser.add_argument("--device", type=int, default=None, help="Specific device ID to test")
    parser.add_argument("--baseline", type=str, default=None, help="Path to baseline JSON for comparison")
    parser.add_argument("--output", type=str, default=None, help="Path to save results JSON")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-trial output")
    args = parser.parse_args()

    hashcat_bin = find_hashcat()
    if not hashcat_bin:
        print("ERROR: hashcat not found", file=sys.stderr)
        sys.exit(1)

    modes = [int(m) for m in args.modes.split(",")] if args.modes else DEFAULT_MODES

    print(f"Hashcat regression test: {len(modes)} modes × {len(VEC_WIDTHS)} vec widths × {args.trials} trials")
    print(f"Estimated time: ~{len(modes) * len(VEC_WIDTHS) * args.trials * 10 // 60} minutes\n")

    system_info = get_system_info(hashcat_bin)
    print(f"Version: {system_info['hashcat_version']}")
    print(f"Devices: {', '.join(system_info['devices'])}")
    print(f"Backends: {', '.join(system_info['backends'])}")

    results = run_regression_test(hashcat_bin, modes, args.trials, args.device, not args.quiet)
    print_summary(results, system_info)

    # Compare against baseline if provided
    if args.baseline:
        with open(args.baseline) as f:
            baseline_data = json.load(f)
        report, regressions = compare_results(results, baseline_data["results"])

        if regressions:
            print(f"\n{'!'*60}")
            print(f"REGRESSIONS DETECTED ({len(regressions)}):")
            print(f"{'!'*60}")
            for r in regressions:
                print(f"  Mode {r['mode']} ({r['name']}) Vec:{r['vec']}: "
                      f"{r['baseline_mean']:.1f} -> {r['current_mean']:.1f} MH/s "
                      f"({r['pct_change']:+.1f}%, t={r['t_stat']:.2f})")
            sys.exit(1)
        else:
            print("\nNo regressions detected.")

    # Save results
    output_path = args.output
    if not output_path:
        os.makedirs("results", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        device_tag = "_".join(d.replace(" ", "_") for d in system_info["devices"])
        output_path = f"results/benchmark_{device_tag}_{timestamp}.json"

    output_data = {"system_info": system_info, "results": results}
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()
