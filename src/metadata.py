import json
from pathlib import Path
from typing import Dict

from qualtran.resource_counting import QECGatesCost, get_cost_value

from truthTable import TruthTable


def calculate_t_cost(tt: TruthTable, name: str) -> Dict[str, int]:
    """Calculate T-costs for QROM and QROAM implementations."""
    results = {}
    configs = [
        ("qrom_t_cost", dict()),
        ("qroam_t_cost", dict(use_qroam=True)),
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


def load_metadata(metadata_file: Path) -> Dict:
    """Load metadata from JSON file."""
    if metadata_file.exists():
        try:
            with open(metadata_file, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_metadata(metadata: Dict, metadata_file: Path) -> None:
    """Save metadata to JSON file."""
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)


def update_metadata(
    metadata: Dict,
    metadata_file: Path,
    filename: str,
    tt: TruthTable,
    problem_type: str,
    **kwargs
) -> None:
    """Update metadata for a benchmark and save to file."""
    t_costs = calculate_t_cost(tt, filename)
    metadata[filename] = {
        "problem_type": problem_type,
        "num_inputs": tt.num_inputs,
        "num_outputs": tt.num_outputs,
        **t_costs,
        **kwargs,
    }
    save_metadata(metadata, metadata_file)

