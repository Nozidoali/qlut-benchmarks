import random
from pathlib import Path
from typing import Dict, List, Tuple

from truthTable import TruthTable
from metadata import update_metadata


def install(
    tt_dir: Path, 
    metadata: Dict, 
    metadata_file: Path,
    input_range: Tuple[int, int], 
    output_counts: List[int], 
    num_seeds: int
) -> List[str]:
    """Install random truth table benchmarks."""
    files: List[str] = []
    min_inputs, max_inputs = input_range
    for n in range(min_inputs, max_inputs + 1):
        for m in output_counts:
            for seed in range(num_seeds):
                random.seed(seed)
                patterns = [
                    "".join(str(random.randint(0, 1)) for _ in range(1 << n))
                    for _ in range(m)
                ]
                path = tt_dir / f"random_{n}in_{m}out_seed{seed}.tt"
                try:
                    tt = TruthTable.from_patterns(patterns, num_inputs=n)
                    tt.to_file(str(path))
                    update_metadata(metadata, metadata_file, path.name, tt, "random", seed=seed)
                    files.append(str(path))
                except Exception as exc:
                    print(f"random benchmark failed (inputs={n}, outputs={m}, seed={seed}): {exc}")
    return files

