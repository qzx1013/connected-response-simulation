# Resource-efficient connected-response simulation

Simulation and plotting code for the connected-response calculations in Figures 2–6 of the accompanying work on pulse-derived noise in superconducting quantum processors.

The repository contains:

- exact-state and PEPS simulation backends;
- fixed residual-coupling geometries and input data;
- figure-specific numerical experiments;
- plotting and validation utilities.

Figures 2–5 correspond to the completed sparse-residual calculations. The `fig6` command implements a seven-edge residual extension using the Figure 5 methodology; it is **not** the zero-residual Figure 6 instance reported in the manuscript. See [`docs/scientific_scope.md`](docs/scientific_scope.md) for details.

## Installation

Python 3.12 is recommended.

```bash
git clone https://github.com/qzx1013/connected-response-simulation.git
cd connected-response-simulation

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

For CUDA 12.4:

```bash
python -m pip install torch==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -c requirements-tested.txt '.[test]'
```

For CPU-only validation:

```bash
python -m pip install torch==2.6.0 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -c requirements-tested.txt '.[test]'
```

Verify the installation:

```bash
residual-tn validate
python -m pytest -q
```

Large production calculations require a high-memory GPU. The reference calculations used an NVIDIA A100 80 GB.

## Running the calculations

Inspect a calculation without launching it:

```bash
residual-tn plan fig5
residual-tn plan fig6
```

Run Figures 2–5:

```bash
residual-tn run fig2 --output results/fig2 --inputs inputs
residual-tn run fig3 --output results/fig3 --inputs inputs
residual-tn run fig4 --output results/fig4 --inputs inputs
residual-tn run fig5 --output results/fig5
```

Run the Figure 6 residual extension:

```bash
residual-tn run fig6 --output results/fig6
residual-tn plot fig6 --results results/fig6
```

Completed stages are checkpointed and reused when a calculation is restarted.

## Repository structure

```text
src/residual_tn/   simulation and plotting code
inputs/            fixed input data
tests/             numerical and portability tests
docs/              scientific scope, architecture, and validation
provenance/        source and geometry provenance
```

For implementation details, physical geometries, checkpoint behavior, and validation records, see [`docs/`](docs/).

## License

MIT License.