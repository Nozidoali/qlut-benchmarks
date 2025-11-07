from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from metadata import load_metadata  # type: ignore # noqa: E402
import gfmult  # type: ignore # noqa: E402
import modexp  # type: ignore # noqa: E402
import preparethc  # type: ignore # noqa: E402
import randtt  # type: ignore # noqa: E402

# Available benchmark types
BENCHMARK_TYPES = {
    'gf2': 'GF(2^n) multipliers',
    'random': 'Random truth tables',
    'modexp': 'Modular exponentiation',
    'preparethc': 'PrepareTHC chemistry',
}


def ensure_directories(benchmark_dir: Path) -> tuple[Path, Path, Path]:
    """Ensure benchmark directories exist and return paths."""
    benchmark_dir = benchmark_dir.resolve()
    tt_dir = benchmark_dir / "truth_tables"
    metadata_file = benchmark_dir / "benchmark_metadata.json"
    
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    tt_dir.mkdir(parents=True, exist_ok=True)
    
    return benchmark_dir, tt_dir, metadata_file


def install_gf2_multipliers(
    benchmark_dir: str | Path | None = None, 
    sizes: List[int] | None = None
) -> List[str]:
    """Install GF(2^n) multiplier benchmarks."""
    if benchmark_dir is None:
        benchmark_dir = BASE_DIR / "benchmarks"
    _, tt_dir, metadata_file = ensure_directories(Path(benchmark_dir))
    metadata = load_metadata(metadata_file)
    return gfmult.install(tt_dir, metadata, metadata_file, sizes)


def install_random_truth_tables(
    benchmark_dir: str | Path | None = None,
    input_range: tuple[int, int] = (2, 8),
    output_counts: List[int] = [1, 2, 4],
    num_seeds: int = 6,
) -> List[str]:
    """Install random truth table benchmarks."""
    if benchmark_dir is None:
        benchmark_dir = BASE_DIR / "benchmarks"
    _, tt_dir, metadata_file = ensure_directories(Path(benchmark_dir))
    metadata = load_metadata(metadata_file)
    return randtt.install(tt_dir, metadata, metadata_file, input_range, output_counts, num_seeds)


def install_modexp_benchmarks(benchmark_dir: str | Path | None = None) -> List[str]:
    """Install modular exponentiation benchmarks."""
    if benchmark_dir is None:
        benchmark_dir = BASE_DIR / "benchmarks"
    _, tt_dir, metadata_file = ensure_directories(Path(benchmark_dir))
    metadata = load_metadata(metadata_file)
    return modexp.install(tt_dir, metadata, metadata_file)


def install_preparethc_benchmarks(benchmark_dir: str | Path | None = None) -> List[str]:
    """Install prepareTHC chemistry benchmarks."""
    if benchmark_dir is None:
        benchmark_dir = BASE_DIR / "benchmarks"
    _, tt_dir, metadata_file = ensure_directories(Path(benchmark_dir))
    metadata = load_metadata(metadata_file)
    return preparethc.install(tt_dir, metadata, metadata_file)


def list_benchmarks(benchmark_dir: str | Path | None = None) -> List[str]:
    """List all installed benchmark files."""
    if benchmark_dir is None:
        benchmark_dir = BASE_DIR / "benchmarks"
    tt_dir = Path(benchmark_dir) / "truth_tables"
    if not tt_dir.exists():
        return []
    return sorted(str(file_path) for file_path in tt_dir.glob("*.tt"))


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Install ASPLOS benchmark truth tables',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available benchmark types:
  gf2         - {BENCHMARK_TYPES['gf2']}
  random      - {BENCHMARK_TYPES['random']}
  modexp      - {BENCHMARK_TYPES['modexp']}
  preparethc  - {BENCHMARK_TYPES['preparethc']}
  all         - Generate all benchmark types (default)

Examples:
  python install.py                                    # Install all benchmarks to default location
  python install.py --output-dir ./my_benchmarks       # Install all to custom directory
  python install.py --benchmarks gf2 random            # Install only GF2 and random benchmarks
  python install.py -o ./output -b modexp preparethc   # Install specific benchmarks to custom directory
        """
    )
    
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default=None,
        help='Output directory for benchmarks (default: ./benchmarks)'
    )
    
    parser.add_argument(
        '-b', '--benchmarks',
        nargs='+',
        choices=list(BENCHMARK_TYPES.keys()) + ['all'],
        default=['all'],
        help='Benchmark types to generate (default: all)'
    )
    
    parser.add_argument(
        '--gf2-sizes',
        nargs='+',
        type=int,
        default=[2, 3, 4, 5],
        help='GF2 multiplier sizes (default: 2 3 4 5)'
    )
    
    parser.add_argument(
        '--random-input-min',
        type=int,
        default=2,
        help='Minimum number of inputs for random truth tables (default: 2)'
    )
    
    parser.add_argument(
        '--random-input-max',
        type=int,
        default=8,
        help='Maximum number of inputs for random truth tables (default: 8)'
    )
    
    parser.add_argument(
        '--random-outputs',
        nargs='+',
        type=int,
        default=[1, 2, 4],
        help='Number of outputs for random truth tables (default: 1 2 4)'
    )
    
    parser.add_argument(
        '--random-seeds',
        type=int,
        default=6,
        help='Number of random seeds (default: 6)'
    )
    
    args = parser.parse_args()
    
    # Determine output directory
    output_dir = args.output_dir if args.output_dir else BASE_DIR / "benchmarks"
    
    # Determine which benchmarks to generate
    benchmarks_to_generate = set(args.benchmarks)
    if 'all' in benchmarks_to_generate:
        benchmarks_to_generate = set(BENCHMARK_TYPES.keys())
    
    print(f"Output directory: {output_dir}")
    print(f"Generating benchmarks: {', '.join(sorted(benchmarks_to_generate))}\n")
    
    all_files = []
    
    # Install GF2 multipliers
    if 'gf2' in benchmarks_to_generate:
        print("Installing GF2 multiplier benchmarks...")
        gf2_files = install_gf2_multipliers(
            benchmark_dir=output_dir,
            sizes=args.gf2_sizes
        )
        print(f"Created {len(gf2_files)} GF2 multiplier files\n")
        all_files.extend(gf2_files)
    
    # Install random truth tables
    if 'random' in benchmarks_to_generate:
        print("Installing random truth table benchmarks...")
        random_files = install_random_truth_tables(
            benchmark_dir=output_dir,
            input_range=(args.random_input_min, args.random_input_max),
            output_counts=args.random_outputs,
            num_seeds=args.random_seeds
        )
        print(f"Created {len(random_files)} random truth table files\n")
        all_files.extend(random_files)
    
    # Install modexp benchmarks
    if 'modexp' in benchmarks_to_generate:
        print("Installing ModExp benchmarks...")
        modexp_files = install_modexp_benchmarks(benchmark_dir=output_dir)
        print(f"Created {len(modexp_files)} ModExp truth table files\n")
        all_files.extend(modexp_files)
    
    # Install prepareTHC benchmarks
    if 'preparethc' in benchmarks_to_generate:
        print("Installing prepareTHC benchmarks...")
        preparethc_files = install_preparethc_benchmarks(benchmark_dir=output_dir)
        print(f"Created {len(preparethc_files)} prepareTHC benchmark files\n")
        all_files.extend(preparethc_files)
    
    # List all installed benchmarks
    print("All installed benchmarks:")
    all_benchmarks = list_benchmarks(benchmark_dir=output_dir)
    for benchmark in all_benchmarks:
        print(f"  {benchmark}")
    
    # Test loading one benchmark
    if all_benchmarks:
        from truthTable import TruthTable
        test_name = Path(all_benchmarks[0]).stem
        print(f"\nTesting load of '{test_name}':")
        tt = TruthTable.from_file(all_benchmarks[0])
        if tt:
            print(f"  Successfully loaded: {tt.num_inputs} inputs, {tt.num_outputs} outputs")
    
    print(f"\n✓ Total files created: {len(all_files)}")


if __name__ == "__main__":
    main()
