from typing import List

from qualtran.bloqs.data_loading.qrom import QROM
from qualtran.bloqs.data_loading.qroam_clean import QROAMClean, QROAMCleanAdjoint
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


def install(installer) -> List[str]:
    files: List[str] = []
    like = (QROM, QROAMClean, QROAMCleanAdjoint)
    for orb, mu, sp, eps in SETUPS:
        try:
            prep = installer._generate_preparethc_bloq(orb, mu, sp, eps)
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
                costs = installer._calculate_qrom_node_t_cost(node, filename)
                installer._update_qrom_node_metadata(
                    filename,
                    node,
                    "preparethc_qrom",
                    num_spin_orb=orb,
                    num_mu=mu,
                    num_bits_state_prep=sp,
                    eps=eps,
                    node_class=cls_name,
                    node_index=idx,
                    node_name=getattr(node, "name", cls_name),
                    num_inputs=tt.num_inputs,
                    num_outputs=tt.num_outputs,
                    **costs,
                )
                installer._write_verilog(tt, path)
                files.append(str(path))
            except Exception as exc:
                print(f"preparethc node failed (orb={orb}, mu={mu}, sp={sp}, node={cls_name}, idx={idx}): {exc}")
    return files

