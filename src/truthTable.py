from __future__ import annotations

from typing import Any, List, Optional, Tuple


def _calc_leff(pattern_length: int) -> int:
    if pattern_length <= 1:
        return 0
    return (pattern_length - 1).bit_length()


class TruthTable:
    # Constants for truth table values
    CONST0 = "0"
    CONST1 = "1"
    DONTCARE = "X"

    def __init__(
        self, num_inputs: int, num_outputs: int, pattern_length: Optional[int] = None
    ):
        self.num_inputs = int(num_inputs)
        self.num_outputs = int(num_outputs)
        if self.num_inputs < 0 or self.num_outputs < 0:
            raise ValueError("num_inputs must be >= 0 and num_outputs must be >= 0")
        if pattern_length is None:
            pattern_length = 1 << self.num_inputs
        if pattern_length <= 0:
            raise ValueError("pattern_length must be positive")
        # Track pattern length explicitly so zero-output tables still retain domain size
        self._pattern_length = int(pattern_length)

        self.table: List[List[str]] = [
            [self.CONST0] * pattern_length for _ in range(self.num_outputs)
        ]

    @staticmethod
    def _validate_patterns(patterns: List[str]) -> int:
        if not patterns:
            raise ValueError("Patterns list is empty")
        pattern_length = len(patterns[0])
        for p in patterns:
            if len(p) != pattern_length:
                raise ValueError("All patterns must have the same length")
            if any(
                c
                not in (
                    TruthTable.CONST0,
                    TruthTable.CONST1,
                    TruthTable.DONTCARE,
                    TruthTable.DONTCARE.lower(),
                )
                for c in p
            ):
                raise ValueError("Patterns must contain only '0', '1', or 'X'")
        return pattern_length

    @staticmethod
    def _ceil_log2(n: int) -> int:
        return _calc_leff(n)

    @classmethod
    def from_patterns(
        cls, patterns: List[str], num_inputs: Optional[int] = None
    ) -> "TruthTable":
        pattern_length = cls._validate_patterns(patterns)
        inferred_inputs = cls._ceil_log2(pattern_length)
        if num_inputs is None:
            num_inputs = inferred_inputs
        if pattern_length > (1 << num_inputs):
            raise ValueError(
                f"pattern_length ({pattern_length}) exceeds 2**num_inputs ({1 << num_inputs})"
            )
        tt = cls(
            num_inputs=num_inputs,
            num_outputs=len(patterns),
            pattern_length=pattern_length,
        )
        tt.table = [list(p.upper()) for p in patterns]
        return tt

    def to_patterns(self) -> List[str]:
        return ["".join(row) for row in self.table]

    @classmethod
    def from_file(cls, file_path: str, num_inputs: Optional[int] = None) -> "TruthTable":
        with open(file_path, "r") as f:
            lines: List[str] = [line.strip() for line in f if line.strip()]
        if not lines:
            raise ValueError("Truth table file is empty")
        length = len(lines[0])
        if any(len(line) != length for line in lines):
            raise ValueError(
                "All lines in the truth table file must have the same length"
            )
        allowed = {
            cls.CONST0,
            cls.CONST1,
            cls.DONTCARE,
            cls.DONTCARE.lower(),
        }
        for line in lines:
            for ch in line:
                if ch not in allowed:
                    raise ValueError(
                        f"Truth table lines may only contain '{cls.CONST0}', "
                        f"'{cls.CONST1}', or '{cls.DONTCARE}'"
                    )
        if length & (length - 1) != 0:
            raise ValueError("Each line length must be a power of two (2**n)")
        return cls.from_patterns(lines, num_inputs=num_inputs)

    @classmethod
    def from_file_filled(
        cls, file_path: str, x_fill: str = CONST0
    ) -> "TruthTable":
        x_fill = x_fill.strip()
        if x_fill not in (cls.CONST0, cls.CONST1):
            raise ValueError(f"x_fill must be '{cls.CONST0}' or '{cls.CONST1}'")
        with open(file_path, "r") as f:
            lines: List[str] = [line.strip() for line in f if line.strip()]
        if not lines:
            raise ValueError("Truth table file is empty")
        length = len(lines[0])
        if any(len(line) != length for line in lines):
            raise ValueError(
                "All lines in the truth table file must have the same length"
            )
        allowed = {
            cls.CONST0,
            cls.CONST1,
            cls.DONTCARE,
            cls.DONTCARE.lower(),
        }
        for line in lines:
            for ch in line:
                if ch not in allowed:
                    raise ValueError(
                        f"Truth table lines may only contain '{cls.CONST0}', "
                        f"'{cls.CONST1}', or '{cls.DONTCARE}'"
                    )
        if length & (length - 1) != 0:
            raise ValueError("Each line length must be a power of two (2**n)")
        binary_patterns = [
            line.replace(cls.DONTCARE, x_fill).replace(cls.DONTCARE.lower(), x_fill)
            for line in lines
        ]
        n_inputs = (length - 1).bit_length()
        return cls.from_patterns(binary_patterns, num_inputs=n_inputs)

    def to_file(self, file_path: str) -> None:
        patterns = self.to_patterns()
        if not patterns:
            raise ValueError("TruthTable has no patterns to write")
        with open(file_path, "w") as f:
            for row in patterns:
                f.write(row + "\n")

    @classmethod
    def from_qrom_bloq(cls, bloq: Any, *, bitorder: str = "lsb") -> "TruthTable":
        try:
            import numpy as np
        except ImportError as exc:
            raise ImportError("numpy is required for TruthTable.from_qrom_bloq") from exc
        try:
            from qualtran.bloqs.data_loading.qrom import QROM  # noqa: F401
        except ImportError as exc:
            raise ImportError("qualtran is required for TruthTable.from_qrom_bloq") from exc

        selection_bitsizes = getattr(bloq, "selection_bitsizes", ())
        if not selection_bitsizes:
            raise ValueError("Bloq must define selection_bitsizes")
        L_full = 1 << int(selection_bitsizes[0])

        def ints_to_bitcols(
            v: np.ndarray, *, bitorder: str, max_bits: Optional[int] = None
        ) -> Tuple[np.ndarray, int]:
            v = np.asarray(v).reshape(-1)
            if v.size == 0:
                bw = max_bits or 1
                return np.zeros((0, bw), dtype=int), bw
            if not np.issubdtype(v.dtype, np.integer):
                v = v.astype(np.int64, copy=False)
            if (v < 0).any():
                raise ValueError("Negative integers not supported in bloq data.")
            vmax = int(v.max()) if v.size else 0
            bw = max_bits if max_bits is not None else max(1, vmax.bit_length())
            cols = [((v >> k) & 1) for k in range(bw)]
            bm = np.stack(cols, axis=1).astype(int)
            if bitorder == "msb":
                bm = bm[:, ::-1]
            return bm, bw

        def normalize_register_to_bits(
            arr: np.ndarray, *, bitorder: str
        ) -> Tuple[List[np.ndarray], List[int]]:
            arr = np.asarray(arr)
            if arr.ndim == 1 or (arr.ndim == 2 and arr.shape[1] == 1):
                bm, bw = ints_to_bitcols(arr, bitorder=bitorder)
                return [bm], [bw]
            if arr.ndim != 2:
                raise ValueError(
                    f"Unsupported ndarray shape {arr.shape} for bloq data"
                )
            if not np.issubdtype(arr.dtype, np.integer):
                arr = arr.astype(np.int64, copy=False)
            mn, mx = int(arr.min()), int(arr.max())
            if mn < 0:
                raise ValueError("Negative values not supported in bloq data.")
            if mx <= 1:
                return [arr.astype(int)], [arr.shape[1]]
            bit_blocks: List[np.ndarray] = []
            widths: List[int] = []
            for j in range(arr.shape[1]):
                col_bm, col_bw = ints_to_bitcols(
                    arr[:, j], bitorder=bitorder
                )
                bit_blocks.append(col_bm)
                widths.append(col_bw)
            bm = np.hstack(bit_blocks)
            return [bm], [sum(widths)]

        arrays = tuple(np.asarray(a) for a in getattr(bloq, "data", ()))
        patterns: List[str] = []
        for arr in arrays:
            bit_mats, widths = normalize_register_to_bits(arr, bitorder=bitorder)
            for bm, width in zip(bit_mats, widths):
                if bm.size == 0:
                    continue
                N, BW = bm.shape
                for j in range(BW):
                    col = bm[:, j]
                    patt_bits = [
                        cls.CONST1 if int(x) == 1 else cls.CONST0 for x in col.tolist()
                    ]
                    if N < L_full:
                        patt_bits += [cls.DONTCARE] * (L_full - N)
                    patterns.append("".join(patt_bits))

        num_inputs = L_full.bit_length() - 1 if L_full > 0 else 0
        return cls.from_patterns(patterns, num_inputs=num_inputs)

    def to_bloq(
        self,
        name: str = "truth_table",
        use_qroam: bool = False,
        use_select_swap: bool = False,
    ):
        try:
            import numpy as np
        except ImportError as exc:
            raise ImportError("numpy is required for TruthTable.to_bloq") from exc
        try:
            if use_qroam:
                from qualtran.bloqs.data_loading.qroam_clean import QROAMClean
            else:
                from qualtran.bloqs.data_loading.qrom import QROM
        except ImportError as exc:
            raise ImportError("qualtran is required for TruthTable.to_bloq") from exc

        _ = name  # API compatibility; not currently used
        patterns = self.to_patterns()
        n_inputs = self.num_inputs
        n_outputs = self.num_outputs
        num_rows = 1 << n_inputs
        output_values: List[int] = []
        for row in range(num_rows):
            output_value = 0
            for bit_pos in range(n_outputs):
                if bit_pos >= len(patterns):
                    continue
                pattern = patterns[bit_pos]
                if row >= len(pattern):
                    continue
                bit_char = pattern[row]
                bit_value = 1 if bit_char == self.CONST1 else 0
                output_value |= bit_value << bit_pos
            output_values.append(output_value)

        data = (np.array(output_values),)
        if use_qroam:
            return QROAMClean(
                data_or_shape=data,
                selection_bitsizes=(n_inputs,),
                target_bitsizes=(n_outputs,),
            )
        return QROM(
            data_or_shape=data,
            selection_bitsizes=(n_inputs,),
            target_bitsizes=(n_outputs,),
        )

    @property
    def len_patterns(self) -> int:
        if self.table:
            return len(self.table[0])
        return getattr(self, "_pattern_length", 0)

    @property
    def L(self) -> int:
        return self.num_inputs

    @property
    def Leff(self) -> int:
        return _calc_leff(self.len_patterns)

