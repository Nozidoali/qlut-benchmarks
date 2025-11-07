import math
from pathlib import Path
from typing import Dict, List, Tuple

from truthTable import TruthTable
from metadata import update_metadata

CASES = [
    (2, 15, 4),
    (3, 35, 8),
    (5, 77, 6),
    (3, 77, 8),
    (3, 221, 4),
    (5, 221, 6),
]


def generate_modexp_truth_table(base: int, mod: int, exp_bitsize: int) -> Tuple[List[str], int, int]:
    n_in = exp_bitsize
    n_out = math.ceil(math.log2(mod))
    n_rows = 1 << n_in
    outputs = [[] for _ in range(n_out)]
    for e in range(n_rows):
        val = pow(base, e, mod)
        bits = format(val, f"0{n_out}b")
        for j, b in enumerate(bits[::-1]):
            outputs[j].append(b)
    return ["".join(col) for col in outputs], n_in, n_out


def install(tt_dir: Path, metadata: Dict, metadata_file: Path) -> List[str]:
    """Install modular exponentiation benchmarks."""
    files: List[str] = []
    for base, mod, exp_bitsize in CASES:
        patterns, n_in, n_out = generate_modexp_truth_table(base, mod, exp_bitsize)
        path = tt_dir / f"modexp_{base}_{mod}_{n_in}_{n_out}.tt"
        try:
            tt = TruthTable.from_patterns(patterns, num_inputs=n_in)
            tt.to_file(str(path))
            update_metadata(
                metadata, metadata_file, path.name, tt, "modexp", 
                base=base, mod=mod, exp_bitsize=exp_bitsize
            )
            files.append(str(path))
        except Exception as exc:
            print(f"modexp benchmark failed (base={base}, mod={mod}, bits={exp_bitsize}): {exc}")
    return files

