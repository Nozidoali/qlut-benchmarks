from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

from qualtran.resource_counting import QECGatesCost, get_cost_value

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gfmult import generate_truth_table, find_irreducible_poly, poly_to_str  # type: ignore # noqa: E402
from truthTable import TruthTable  # type: ignore # noqa: E402
from selectswap import truth_table_to_verilog


class BenchmarkInstaller:
    def __init__(self, benchmark_dir: str | Path = None):
        if benchmark_dir is None:
            benchmark_dir = BASE_DIR / "benchmarks"
        self.benchmark_dir = Path(benchmark_dir).resolve()
        self.tt_dir = self.benchmark_dir / "truth_tables"
        self.metadata_file = self.benchmark_dir / "benchmark_metadata.json"
        self._ensure_directories()
        self._load_metadata()

    def _ensure_directories(self) -> None:
        self.benchmark_dir.mkdir(parents=True, exist_ok=True)
        self.tt_dir.mkdir(parents=True, exist_ok=True)

    def _load_metadata(self) -> None:
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r") as f:
                    self.metadata = json.load(f)
            except Exception:
                self.metadata = {}
        else:
            self.metadata = {}

    def _save_metadata(self) -> None:
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def _calculate_t_cost(self, tt: TruthTable, name: str) -> Dict[str, int]:
        results = {}
        configs = [
            ("qrom_t_cost", dict()),
            ("qroam_t_cost", dict(use_qroam=True)),
            ("qroam_selectswap_t_cost", dict(use_qroam=True, use_select_swap=True)),
        ]
        for key, opts in configs:
            try:
                bloq = tt.to_bloq(name=name, **opts)
                gc = get_cost_value(bloq, cost_key=QECGatesCost())
                results[key] = gc.total_t_count()
            except Exception as exc:
                print(f"{key} unavailable for {name}: {exc}")
                results[key] = 0
        return results

    def _update_metadata(
        self, filename: str, tt: TruthTable, problem_type: str, **kwargs
    ) -> None:
        t_costs = self._calculate_t_cost(tt, filename)
        self.metadata[filename] = {
            "problem_type": problem_type,
            "num_inputs": tt.num_inputs,
            "num_outputs": tt.num_outputs,
            **t_costs,
            **kwargs,
        }
        self._save_metadata()

    def _write_verilog(self, tt: TruthTable, path: Path) -> Optional[str]:
        try:
            verilog = truth_table_to_verilog(tt)
            target = path.with_suffix(".v")
            target.write_text(verilog)
            return str(target)
        except Exception as exc:
            print(f"Verilog emission failed for {path.name}: {exc}")
            return None

    def install_gf2_multipliers(self, sizes: List[int] = None) -> List[str]:
        if sizes is None:
            sizes = [2, 3, 4, 5, 6, 7, 8, 9]
        created_files = []
        for n in sizes:
            print(f"Generating GF(2^{n}) multiplier truth table...")
            poly = find_irreducible_poly(n)
            if poly is None:
                print(f"Warning: No irreducible polynomial found for degree {n}")
                continue
            print(f"Using polynomial: {poly_to_str(poly)}")
            try:
                tt = generate_truth_table(n, poly)
                if tt is None:
                    print(f"Warning: Failed to generate truth table for GF(2^{n})")
                    continue
                filename = f"gf_mult{n}.tt"
                filepath = self.tt_dir / filename
                tt.to_file(str(filepath))
                self._update_metadata(
                    filename,
                    tt,
                    "gf2_multiplier",
                    polynomial=poly_to_str(poly),
                    field_size=n,
                )
                self._write_verilog(tt, filepath)
                created_files.append(str(filepath))
            except Exception as e:
                print(f"Error generating GF(2^{n}) multiplier: {e}")
                continue
        return created_files

    def install_custom_truth_table(
        self, name: str, patterns: List[str], num_inputs: int
    ) -> str:
        tt = TruthTable.from_patterns(patterns, num_inputs=num_inputs)
        filename = f"{name}.tt"
        filepath = self.tt_dir / filename
        tt.to_file(str(filepath))
        self._update_metadata(filename, tt, "custom")
        self._write_verilog(tt, filepath)
        return str(filepath)

    def install_random_truth_tables(
        self,
        input_range: tuple[int, int] = (3, 8),
        output_counts: List[int] = [1, 2, 4],
        num_seeds: int = 6,
    ) -> List[str]:
        from randtt import install
        return install(self, input_range, output_counts, num_seeds)

    def install_modexp_truth_tables(self) -> List[str]:
        from modexp import install
        return install(self)

    def install_preparethc_benchmarks(self) -> List[str]:
        from preparethc import install
        return install(self)

    def list_installed_benchmarks(self) -> List[str]:
        if not self.tt_dir.exists():
            return []
        return sorted(str(file_path) for file_path in self.tt_dir.glob("*.tt"))

    def get_metadata(self, name: str = None) -> Dict:
        if name:
            return self.metadata.get(name, {})
        return self.metadata

    def load_benchmark(self, name: str) -> Optional[TruthTable]:
        filepath = self.tt_dir / f"{name}.tt"
        if not filepath.exists():
            print(f"Benchmark '{name}' not found at {filepath}")
            return None
        try:
            return TruthTable.from_file(str(filepath))
        except Exception as e:
            print(f"Error loading benchmark '{name}': {e}")
            return None


def install_default_benchmarks(benchmark_dir: str | Path | None = None) -> List[str]:
    installer = BenchmarkInstaller(benchmark_dir)
    return installer.install_gf2_multipliers()


def install_random_benchmarks(
    benchmark_dir: str | Path | None = None,
    input_range: tuple[int, int] = (3, 8),
    output_counts: List[int] = [1, 2, 4],
    num_seeds: int = 6,
) -> List[str]:
    installer = BenchmarkInstaller(benchmark_dir)
    return installer.install_random_truth_tables(input_range, output_counts, num_seeds)


def install_modexp_benchmarks(benchmark_dir: str | Path | None = None) -> List[str]:
    installer = BenchmarkInstaller(benchmark_dir)
    return installer.install_modexp_truth_tables()


def install_preparethc_benchmarks(
    benchmark_dir: str | Path | None = None,
) -> List[str]:
    installer = BenchmarkInstaller(benchmark_dir)
    return installer.install_preparethc_benchmarks()


if __name__ == "__main__":
    installer = BenchmarkInstaller()
    print("Installing GF2 multiplier benchmarks...")
    gf2_files = installer.install_gf2_multipliers([2, 3, 4, 5])
    print(f"\nCreated {len(gf2_files)} GF2 multiplier files")
    print("\nInstalling random truth table benchmarks...")
    random_files = installer.install_random_truth_tables(
        input_range=(3, 8), output_counts=[1, 2, 4], num_seeds=6
    )
    print(f"\nCreated {len(random_files)} random truth table files")

    print("\nInstalling ModExp benchmarks...")
    modexp_files = installer.install_modexp_truth_tables()
    print(f"\nCreated {len(modexp_files)} ModExp truth table files")

    print("\nInstalling prepareTHC benchmarks...")
    preparethc_files = installer.install_preparethc_benchmarks()
    print(f"\nCreated {len(preparethc_files)} prepareTHC benchmark files")

    print("\nAll installed benchmarks:")
    all_benchmarks = installer.list_installed_benchmarks()
    for benchmark in all_benchmarks:
        print(f"  {benchmark}")
    if all_benchmarks:
        test_name = Path(all_benchmarks[0]).stem
        print(f"\nTesting load of '{test_name}':")
        tt = installer.load_benchmark(test_name)
        if tt:
            print(
                f"  Successfully loaded: {tt.num_inputs} inputs, {tt.num_outputs} outputs"
            )
