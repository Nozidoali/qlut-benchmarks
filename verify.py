#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


def cec(tt: Path, v: Path) -> tuple[bool, str]:
    if not tt.exists():
        return False, "missing truth table"
    if not v.exists():
        return False, "missing verilog"
    try:
        res = subprocess.run(
            ["abc", "-c", f"cec -n {tt} {v}"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except FileNotFoundError:
        return False, "abc not found"
    except Exception as exc:
        return False, f"error: {exc}"
    out = res.stdout + res.stderr
    if "Networks are equivalent" in out or "are identical" in out:
        return True, "equivalent"
    if "Networks are NOT EQUIVALENT" in out:
        return False, "not equivalent"
    return False, "unknown"


def verify(patterns: list[str]) -> None:
    root = Path(__file__).resolve().parent
    tt_dir = root / "benchmarks" / "truth_tables"
    if not tt_dir.exists():
        print(f"missing directory: {tt_dir}")
        sys.exit(1)
    total = passed = 0
    for pattern in patterns:
        for tt_file in sorted(tt_dir.glob(f"{pattern}.tt")):
            ok, msg = cec(tt_file, tt_file.with_suffix(".v"))
            mark = "\033[92m✓\033[0m" if ok else "\033[91m✗\033[0m"
            print(f"{mark} {tt_file.stem:40s} {msg}")
            total += 1
            passed += int(ok)
    failed = total - passed
    print("\n" + "=" * 60)
    print(f"total {total}  passed {passed}  failed {failed}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify QoR with ABC cec")
    parser.add_argument("--patterns", nargs="+", default=["gf_mult*", "modexp_*", "random_*"])
    args = parser.parse_args()
    print("Verifying benchmarks with ABC cec...")
    print("=" * 60)
    verify(args.patterns)
