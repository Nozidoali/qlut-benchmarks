import random
from typing import List, Tuple

from truthTable import TruthTable


def install(installer, input_range: Tuple[int, int], output_counts: List[int], num_seeds: int) -> List[str]:
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
                path = installer.tt_dir / f"random_{n}in_{m}out_seed{seed}.tt"
                try:
                    tt = TruthTable.from_patterns(patterns, num_inputs=n)
                    tt.to_file(str(path))
                    installer._update_metadata(path.name, tt, "random", seed=seed)
                    installer._write_verilog(tt, path)
                    files.append(str(path))
                except Exception as exc:
                    print(f"random benchmark failed (inputs={n}, outputs={m}, seed={seed}): {exc}")
    return files

