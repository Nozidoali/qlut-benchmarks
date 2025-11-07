from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, Iterable, List, Optional

from truthTable import TruthTable


@dataclass
class LogicGate:
    gate_type: str
    inputs: List[str]
    output: str
    data: Dict[str, bool]

    def _lit(self, value: str, inverted: bool) -> str:
        return f"~{value}" if inverted else value

    def to_assignment(self) -> str:
        p1 = bool(self.data.get("p1"))
        p2 = bool(self.data.get("p2"))
        p3 = bool(self.data.get("p3"))

        if self.gate_type == "&":
            a, b = self.inputs
            expr = f"({self._lit(a, p1)} & {self._lit(b, p2)})"
            expr = f"~{expr}" if p3 else expr
            return f"  assign {self.output} = {expr};"

        if self.gate_type == "^":
            a, b = self.inputs
            expr = f"({self._lit(a, p1)} ^ {self._lit(b, p2)})"
            expr = f"~{expr}" if p3 else expr
            return f"  assign {self.output} = {expr};"

        if self.gate_type == "=":
            (a,) = self.inputs
            return f"  assign {self.output} = {self._lit(a, p1)};"

        raise ValueError(f"Unsupported gate type: {self.gate_type}")


class LogicNetwork:
    CONST0 = "const0"
    CONST1 = "const1"

    def __init__(self):
        self.inputs: List[str] = []
        self.outputs: List[str] = []
        self.gates: Dict[str, LogicGate] = {}
        self._fanouts: Dict[str, set[str]] = {}
        self._counter: int = 0

    def _unique_name(self) -> str:
        name = f"n{self._counter}"
        while name in self.gates or name in self.inputs or name in self.outputs:
            self._counter += 1
            name = f"n{self._counter}"
        self._counter += 1
        return name

    def create_pi(self, name: str) -> None:
        self.inputs.append(name)

    def create_pis(self, names: Iterable[str]) -> None:
        self.inputs.extend(names)

    def create_po(self, name: str) -> None:
        self.outputs.append(name)

    def create_pos(self, names: Iterable[str]) -> None:
        self.outputs.extend(names)

    def create_gate(self, gate_type: str, inputs: List[str], output: Optional[str] = None, data: Optional[Dict[str, bool]] = None) -> str:
        if output is None:
            output = self._unique_name()
        self.gates[output] = LogicGate(gate_type, inputs, output, data or {})
        return output

    def create_and(self, a: str, b: str, *, output: Optional[str] = None) -> str:
        return self.create_gate("&", [a, b], output, {"p1": False, "p2": False, "p3": False})

    def create_xor(self, a: str, b: str, *, output: Optional[str] = None) -> str:
        return self.create_gate("^", [a, b], output, {"p1": False, "p2": False, "p3": False})

    def create_buf(self, source: str, *, output: Optional[str] = None, inverted: bool = False) -> str:
        return self.create_gate("=", [source], output, {"p1": inverted})

    def create_const(self, value: str, *, output: Optional[str] = None) -> str:
        const_name = self.CONST0 if value == "0" else self.CONST1
        return self.create_gate("=", [const_name], output, {"p1": False})

    def _compute_fanouts(self) -> None:
        self._fanouts.clear()
        for gate in self.gates.values():
            for fin in gate.inputs:
                self._fanouts.setdefault(fin, set()).add(gate.output)

    def topological_order(self) -> List[str]:
        indegree: Dict[str, int] = {node: 0 for node in self.gates}
        for node in self.inputs:
            indegree[node] = 0

        for gate in self.gates.values():
            for fin in gate.inputs:
                if fin not in indegree:
                    indegree[fin] = 0

        for gate in self.gates.values():
            for fin in gate.inputs:
                if gate.output in indegree:
                    indegree[gate.output] += 1

        queue = [node for node, deg in indegree.items() if deg == 0 and (node in self.inputs or node in (self.CONST0, self.CONST1))]
        order: List[str] = []

        while queue:
            node = queue.pop(0)
            order.append(node)
            for fanout in self._fanouts.get(node, []):
                if fanout in indegree:
                    indegree[fanout] -= 1
                    if indegree[fanout] == 0:
                        queue.append(fanout)

        return [n for n in order if n in self.gates]

    def to_verilog(self) -> str:
        self._compute_fanouts()

        def sanitize(name: str) -> str:
            sanitized = name.replace("\\", "_")
            return re.sub(r"\[(\d+)\]", r"_\1", sanitized)

        pis = [sanitize(name) for name in self.inputs]
        pos = [sanitize(name) for name in self.outputs]
        internal = sorted(
            sanitize(n)
            for n in self.gates
            if n not in self.inputs and n not in self.outputs
        )

        uses_const0 = any(self.CONST0 in g.inputs for g in self.gates.values()) or (self.CONST0 in self.inputs)
        uses_const1 = any(self.CONST1 in g.inputs for g in self.gates.values()) or (self.CONST1 in self.inputs)

        lines: List[str] = []
        lines.append("module top(")
        ports = pis + pos
        lines.append("  " + ", ".join(ports))
        lines.append(");")

        if pis:
            lines.append("  input " + ", ".join(pis) + ";")
        if pos:
            lines.append("  output " + ", ".join(pos) + ";")

        if uses_const0 or uses_const1:
            consts: List[str] = []
            if uses_const0:
                consts.append("const0")
            if uses_const1:
                consts.append("const1")
            lines.append("  wire " + ", ".join(consts) + ";")
            if uses_const0:
                lines.append("  assign const0 = 1'b0;")
            if uses_const1:
                lines.append("  assign const1 = 1'b1;")

        if internal:
            lines.append("  wire " + ", ".join(internal) + ";")

        for node in self.topological_order():
            gate = self.gates[node]
            sanitized_gate = LogicGate(
                gate.gate_type,
                [sanitize(inp) for inp in gate.inputs],
                sanitize(gate.output),
                gate.data,
            )
            lines.append(sanitized_gate.to_assignment())

        lines.append("endmodule")
        lines.append("")
        return "\n".join(lines)


def reshape_for_cofactors(tt: TruthTable, k: int) -> TruthTable:
    n = tt.num_inputs
    m = tt.num_outputs
    remaining = n - k
    cofactors = 1 << k
    reshaped = TruthTable(remaining, cofactors * m)
    table = [[TruthTable.DONTCARE] * (1 << remaining) for _ in range(cofactors * m)]

    for cofactor_idx in range(cofactors):
        prefix = cofactor_idx
        for suffix in range(1 << remaining):
            idx = (prefix << remaining) | suffix
            if idx >= (1 << n):
                continue
            for output_idx in range(m):
                reshaped_idx = cofactor_idx * m + output_idx
                table[reshaped_idx][suffix] = tt.table[output_idx][idx]

    reshaped.table = table
    return reshaped


def generate_minterms(tt: TruthTable, inputs: List[str]) -> tuple[LogicNetwork, Dict[int, str]]:
    network = LogicNetwork()
    network.create_pis(inputs)
    minterms: Dict[int, str] = {}
    idx = 0

    for assignment in range(tt.len_patterns):
        if tt.table[0][assignment] != "1":
            continue
        bits = format(assignment, f"0{tt.num_inputs}b")
        signals: List[str] = []
        for pos, bit in enumerate(bits):
            if bit == "1":
                signals.append(inputs[pos])
            else:
                inv_name = f"inv_{inputs[pos]}"
                if inv_name not in network.gates:
                    network.create_buf(inputs[pos], output=inv_name, inverted=True)
                signals.append(inv_name)

        if not signals:
            term = LogicNetwork.CONST1
        elif len(signals) == 1:
            term = signals[0]
        else:
            term = network.create_and(signals[0], signals[1])
            for extra in signals[2:]:
                term = network.create_and(term, extra)

        minterms[idx] = term
        idx += 1

    return network, minterms


def xor_tree(network: LogicNetwork, signals: List[str]) -> str:
    if not signals:
        return LogicNetwork.CONST0
    if len(signals) == 1:
        return signals[0]

    current = list(signals)
    while len(current) > 1:
        next_level: List[str] = []
        for i in range(0, len(current), 2):
            if i + 1 < len(current):
                node = network.create_xor(current[i], current[i + 1])
                next_level.append(node)
            else:
                next_level.append(current[i])
        current = next_level
    return current[0]


def select_network(tt: TruthTable, inputs: List[str]) -> LogicNetwork:
    network = LogicNetwork()
    network.create_pis(inputs)

    care_bits = []
    for idx in range(1 << tt.num_inputs):
        care_bits.append("1" if any(col[idx] != "X" for col in tt.table) else "0")

    care_tt = TruthTable(tt.num_inputs, 1)
    care_tt.table = [care_bits]

    minterm_network, minterm_map = generate_minterms(care_tt, inputs)
    network.gates.update(minterm_network.gates)

    care_positions = [idx for idx, bit in enumerate(care_bits) if bit == "1"]
    outputs: List[str] = []

    for output_idx in range(tt.num_outputs):
        signal_name = f"select_{output_idx}"
        outputs.append(signal_name)

        active = []
        for i, pos in enumerate(care_positions):
            if tt.table[output_idx][pos] == "1":
                active.append(minterm_map[i])

        if not active:
            network.create_const("0", output=signal_name)
        else:
            network.create_buf(xor_tree(network, active), output=signal_name)

    network.create_pos(outputs)
    return network


def mux_tree(network: LogicNetwork, values: List[str], selectors: List[str]) -> str:
    if len(values) == 1:
        return values[0]

    current = list(values)
    level = 0
    while len(current) > 1:
        nxt: List[str] = []
        for i in range(0, len(current), 2):
            if i + 1 < len(current):
                in0, in1 = current[i], current[i + 1]
                sel = selectors[level]
                diff = network.create_xor(in0, in1)
                gated = network.create_and(sel, diff)
                nxt.append(network.create_xor(in0, gated))
            else:
                nxt.append(current[i])
        current = nxt
        level += 1
    return current[0]


def swap_network(k: int, outputs_per_cofactor: int, select_signals: List[str], selectors: List[str]) -> LogicNetwork:
    network = LogicNetwork()
    network.create_pis(selectors + select_signals)

    outputs = [f"o{idx}" for idx in range(outputs_per_cofactor)]
    network.create_pos(outputs)

    cofactors = 1 << k
    for output_idx in range(outputs_per_cofactor):
        inputs = []
        for cofactor_idx in range(cofactors):
            select_idx = cofactor_idx * outputs_per_cofactor + output_idx
            inputs.append(select_signals[select_idx])
        if len(inputs) == 1:
            network.create_buf(inputs[0], output=outputs[output_idx])
        else:
            final_signal = mux_tree(network, inputs, selectors)
            network.create_buf(final_signal, output=outputs[output_idx])

    return network


def merge_networks(select_net: LogicNetwork, swap_net: LogicNetwork) -> LogicNetwork:
    merged = LogicNetwork()
    all_inputs = []
    for inp in swap_net.inputs:
        if inp not in select_net.outputs and inp not in all_inputs:
            all_inputs.append(inp)
    for inp in select_net.inputs:
        if inp not in all_inputs:
            all_inputs.append(inp)
    merged.inputs = all_inputs
    merged.outputs = swap_net.outputs.copy()
    merged.gates.update(select_net.gates)
    merged.gates.update({k: v for k, v in swap_net.gates.items() if k not in merged.gates})
    merged._compute_fanouts()
    return merged


def truth_table_to_verilog(tt: TruthTable, input_names: Optional[List[str]] = None, k: Optional[int] = None) -> str:
    if tt.num_outputs == 0:
        raise ValueError("TruthTable must have at least one output to export Verilog")
    if input_names is None:
        input_names = [f"x{i}" for i in range(tt.num_inputs)]
    if len(input_names) != tt.num_inputs:
        raise ValueError("input_names must match num_inputs")
    if k is None:
        k = min(tt.num_inputs // 2, 3)

    reshaped = reshape_for_cofactors(tt, k)
    select_inputs = [f"x{i}" for i in range(k, tt.num_inputs)]
    select_net = select_network(reshaped, select_inputs)
    select_outputs = select_net.outputs
    selection_inputs = [f"x{i}" for i in range(k)]

    swap_net = swap_network(k, tt.num_outputs, select_outputs, selection_inputs)
    merged = merge_networks(select_net, swap_net)
    return merged.to_verilog()

