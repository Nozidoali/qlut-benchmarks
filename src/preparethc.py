from typing import Any, Dict, List

import numpy as np
from qualtran.bloqs.chemistry.thc import PrepareTHC
from qualtran.bloqs.data_loading.qrom import QROM
from qualtran.bloqs.data_loading.qroam_clean import QROAMClean, QROAMCleanAdjoint
from qualtran.resource_counting import QECGatesCost, get_cost_value
from qualtran.resource_counting.generalizers import ignore_split_join

from truthTable import TruthTable

SETUPS = [
    (8, 6, 2, 1e-2),
    (8, 12, 2, 1e-2),
    (8, 23, 2, 1e-2),
    (8, 6, 4, 1e-2),
    (8, 12, 4, 1e-2),
    (8, 23, 4, 1e-2),
]


def generate_preparethc_bloq(orb: int, mu: int, sp: int, eps: float) -> PrepareTHC:
    tpq = np.random.normal(0, 1, size=(orb // 2, orb // 2))
    zeta = np.random.normal(0, 1, size=(mu, mu))
    zeta = 0.5 * (zeta + zeta.T)
    eta = np.random.normal(0, 1, size=(mu, orb // 2))
    
    eri_thc = np.einsum("Pp,Pr,Qq,Qs,PQ->prqs", eta, eta, eta, eta, zeta, optimize=True)
    tpq_prime = tpq - 0.5 * np.einsum("illj->ij", eri_thc, optimize=True) + np.einsum("llij->ij", eri_thc, optimize=True)
    t_l = np.linalg.eigvalsh(tpq_prime)
    
    t_l = t_l[np.abs(t_l) > eps]
    eta = eta[:, np.linalg.norm(eta, axis=0) > eps]
    
    return PrepareTHC.from_hamiltonian_coeffs(t_l, eta, zeta, num_bits_state_prep=sp)


def calculate_qrom_node_t_cost(node: Any) -> Dict[str, int]:
    try:
        gc = get_cost_value(node, cost_key=QECGatesCost())
        return {"qrom_node_t_cost": gc.total_t_count()}
    except Exception:
        return {"qrom_node_t_cost": 0}


def install(installer) -> List[str]:
    files: List[str] = []
    like = (QROM, QROAMClean, QROAMCleanAdjoint)
    for orb, mu, sp, eps in SETUPS:
        try:
            prep = generate_preparethc_bloq(orb, mu, sp, eps)
        except Exception as exc:
            print(f"preparethc bloq failed (orb={orb}, mu={mu}, sp={sp}, eps={eps}): {exc}")
            continue

        graph, _ = prep.call_graph(generalizer=ignore_split_join)
        nodes = [n for n in getattr(graph, "nodes", []) if isinstance(n, like)]
        if not nodes:
            print(f"preparethc bloq has no QROM-like nodes (orb={orb}, mu={mu}, sp={sp})")
            continue

        for idx, node in enumerate(nodes):
            cls_name = type(node).__name__
            filename = f"preparethc_orb{orb}_mu{mu}_sp{sp}_eps{eps:.0e}_{cls_name}_{idx}.tt"
            path = installer.tt_dir / filename
            try:
                tt = TruthTable.from_qrom_bloq(node, bitorder="lsb")
                tt.to_file(str(path))
                costs = calculate_qrom_node_t_cost(node)
                installer.metadata[filename] = {
                    "problem_type": "preparethc_qrom",
                    "bloq_type": "prepareTHC_QROM",
                    "num_spin_orb": orb,
                    "num_mu": mu,
                    "num_bits_state_prep": sp,
                    "eps": eps,
                    "node_class": cls_name,
                    "node_index": idx,
                    "node_name": getattr(node, "name", cls_name),
                    "num_inputs": tt.num_inputs,
                    "num_outputs": tt.num_outputs,
                    **costs,
                }
                installer._save_metadata()
                installer._write_verilog(tt, path)
                files.append(str(path))
            except Exception as exc:
                print(f"preparethc node failed (orb={orb}, mu={mu}, sp={sp}, node={cls_name}, idx={idx}): {exc}")
    return files

