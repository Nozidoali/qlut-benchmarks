# ASPLOS Benchmarks

Benchmark suite for quantum circuit synthesis experiments. Generates truth tables and corresponding Verilog implementations using select-swap decomposition.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the installer to generate all benchmark families:

```bash
python install.py
```

This creates:
- Truth table files (`.tt`) in `benchmarks/truth_tables/`
- Verilog implementations (`.v`) using select-swap decomposition
- Metadata with T-costs in `benchmarks/benchmark_metadata.json`

## Benchmark Families

### GF(2^n) Multipliers
Galois field multiplication circuits for `n = 2, 3, ..., 9` with irreducible polynomials.

### Random Truth Tables
Random Boolean functions with configurable inputs (3-8), outputs (1, 2, 4), and seeds (0-5).

### Modular Exponentiation (ModExp)
Circuits computing `base^exp mod N` for various parameter combinations.

### PrepareTHC Chemistry
QROM nodes extracted from quantum chemistry prepareTHC bloqs with different orbital/mu/state-prep parameters.

## Verification

Verify that Verilog implementations match their truth tables using ABC:

```bash
python verify.py --patterns "gf_mult*" "modexp_*" "random_*"
```

Requires ABC: `brew install abc` or `apt install abc`
