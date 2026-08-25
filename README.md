<h1 style="text-align:center;">Mycelial Search: A Graph-Structured Metaheuristic for Continuous Optimisation</h1>

---

[![Python](https://img.shields.io/badge/Python-3.10-3776ab?logo=python&logoColor=fff)](https://www.python.org/)
[![Anaconda](https://img.shields.io/badge/Anaconda-44A833?logo=anaconda&logoColor=fff)](https://www.anaconda.com/)
[![PyCharm](https://img.shields.io/badge/PyCharm-2026.x-000?logo=pycharm&logoColor=fff)](https://www.jetbrains.com/pycharm/)

[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

This repository contains the research implementation and benchmark code for **Mycelial Search (Myco)**, a graph-structured metaheuristic for continuous optimisation. Myco represents candidate solutions as active tips in an evolving graph, combines local flow with community structure, adapts edge conductance, and preserves promising positions as anchors.

[![Graphical Abstract](https://i.postimg.cc/vHg5s2H0/Myco.png)](https://postimg.cc/8jG7dHvv)

---

## Requirements

- Windows, macOS, or Linux
- [Miniconda/Anaconda](https://www.anaconda.com/download/success)
- Python 3.10 (provided by `environment.yml`)

The supplied environment includes NumPy, SciPy, pandas, Matplotlib, NetworkX, Mealpy, OPFUNU, CDlib (for the Leiden backend), and the other benchmark dependencies.

---

## Installation

From the repository root, create the environment defined by [`environment.yml`](environment.yml):

```bash
conda env create --prefix ./.conda --file environment.yml
conda activate ./.conda
```

On Windows, the environment can also be used directly with `.conda\python.exe` without activating it.

---

## Testing

Confirm that the CEC 2022 implementation and bundled input data are available:

```bash
python cec_functions.py
```

For a reduced benchmark smoke test, use one function and one run:

```bash
python run_backend_ablation.py --dim 10 --functions F1 --quick-runs 1 --max-fe 1000 --no-diag
```

The `--max-fe` option keeps a smoke test short. Omit it for the standard CEC
evaluation budget.

---

## Reproducing benchmarks

### Full comparison

`run_comparison.py` evaluates Myco variants and the comparison algorithms on CEC 2022 functions F1-F12. Its default configuration is dimension 10, 30 runs, and the standard CEC evaluation budget: 

```bash
python run_comparison.py
```

The script writes Excel summaries, convergence data, seeds, parameters, and per-run histories to `results/Comparison D10/` by default. To run dimension 20, change `DIMENSION = 10` to `DIMENSION = 20` near the top of the script.

### Myco community-detection backends

`run_backend_ablation.py` is the standalone ablation runner for the Myco
plasticity variant. It includes the optimiser implementation, CEC benchmark
runner, command-line parsing, convergence recording, seed logging, and
community diagnostics:

```bash
python run_backend_ablation.py --backend louvain --dim 10 --runs 30 --roi --functions F1 F2 F6 F7 F10
```

> **Input Arguments**
>
> Please pay attention to the following arguments.
>
> - Omit `--functions` to run all registered CEC functions. 
> - Use `--quick-runs 1` and/or `--max-fe 1000` for a fast test. 
> - Add `--output-dir <directory>` to keep results separate. 
> - Results are written to a backend-specific directory under `results/`, including Excel summaries, convergence JSON, and seed JSON.

To test every supported backend, use `--backend all`:

```bash
python run_backend_ablation.py --backend all --quick-runs 1 --max-fe 1000 --no-diag
```

> **Backends**
>
> Supported backends are:
>
> - `leiden`: Community detection with Leiden [DOI](https://doi.org/10.1038/s41598-019-41695-z)
> - `louvain`: Community detection with Louvain [DOI](https://doi.org/10.1088/1742-5468/2008/10/P10008)
> - `greedy_modularity`: Community detection with Greedy Modularity disassembly [DOI](https://doi.org/10.1038/s41598-024-55190-7)
> - `label_propagation`: Community detection with Label Propagation Algorithm [DOI](https://doi.org/10.1016/j.physa.2019.122058)
> - `asyn_lpa`: Community Detection with Near linear time algorithm [DOI](https://doi.org/10.1103/PhysRevE.76.036106)
> - `connected_components`: Connected component partitioning [DOI](https://doi.org/10.25080/TCWV9851)

The `leiden` backend requires CDlib, which is included in `environment.yml`. The other backends use NetworkX.


### Separate benchmark subsets

The smaller scripts run only the Myco variants:

```bash
python benchmark.py        # Core subset: F1, F2, F4, F6, F9
python benchmark_extra.py  # Extra subset: F3, F5, F7, F8, F10, F11, F12
```

`benchmark.py` writes to `results/Core Functions D10/` by default (or the
selected dimension) and
`benchmark_extra.py` writes to `results/extra/`.

---

## Repository layout

| Path | Purpose |
| --- | --- |
| `algorithms/` | Myco core implementations and comparison algorithms |
| `cec_functions.py` | CEC 2022 function registry and metadata |
| `cec2022_functions.py` | CEC 2022 objective-function implementation |
| `input_data/` | CEC shifts, shuffles, and random seeds |
| `run_comparison.py` | Full comparison benchmark |
| `run_backend_ablation.py` | Standalone configurable plasticity/backend ablation benchmark |
| `benchmark.py`, `benchmark_extra.py` | Core and extra Myco-only benchmarks |
| `results/` | Generated summaries, convergence histories, and seeds |

All benchmark scripts use the CEC 2022 bound range `[-100, 100]` and log the seeds used for reproducibility. The benchmark scripts support dimensions 10 and 20 because those are the bundled CEC input-data dimensions. The wrapper smoke test also exercises dimension 2; other dimensions require additional CEC input data.
Benchmark runs can be lengthy; start with the smoke-test command before launching a full experiment.

---

## Citation

If you use this code, model design, experimental protocol, or the Myco evaluation setting, please cite the Myco manuscript. The machine-readable citation metadata is also available in [`CITATION.cff`](CITATION.cff).

```bibtex
@article{dehshibi2026myco,
  author        = {Mohammad Mahdi Dehshibi},
  title         = {Mycelial Search: {A} Graph-Structured Metaheuristic for Continuous Optimisation},
  year          = {2026},
  eprint        = {2608.23323},
  archivePrefix = {arXiv},
  primaryClass  = {cs.NE},
  url           = {https://arxiv.org/abs/2608.23323},
}

@software{dehshibi2026myco_git,
  author        = {Mohammad Mahdi Dehshibi},
  title         = {Mycelial Search: {A} Graph-Structured Metaheuristic for Continuous Optimisation},
  howpublished  = {ver. 1},
  year          = {2026},
  url           = {https://github.com/dehshibi/Mycelia-Search}
}
```

---

## License

This repository is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for the full license text.

The license applies only to the original source code in this repository.
CEC 2022 input data and third-party source code remain subject to their own licenses, terms of use, and access conditions. Users are responsible for complying with those terms.

## Contact

For questions, bug reports, or collaboration enquiries, please [open an issue](https://github.com/dehshibi/Mycelial-Search/issues).


Copyright © 2026 Mohammad Mahdi Dehshibi.
