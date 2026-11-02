# SaNSDE

SaNSDE is a Python package for global optimization of isolated atomic nanoclusters. It combines a self-adaptive differential-evolution search with ASE local relaxation: candidates are produced by mutation, crossover and cluster-specific transformations, relaxed with BFGS, and selected by energy/similarity criteria. Structures are deduplicated by point group and geometric similarity to keep the population diverse.

Any ASE calculator can be used as the potential. The default is a DeePMD-kit model with an optional hiphive ZBL short-range term; Lennard-Jones is available for fast tests.

## Features

- Optimize one element over a size range (`MIN_ATOM` .. `MAX_ATOM`).
- Population-based differential evolution with adaptive control parameters (`fp`, `p`, `crm`), stagnation handling and similarity-based restarts.
- Point-group aware deduplication driven by a geometric similarity function.
- Multi-round optimization (`ROUND`), each round starting from the previous round's best structures.
- Export of best structures (extxyz), population snapshots (JSON) and per-generation statistics (CSV).
- `xyz2vasp` subcommand to turn optimized clusters into VASP task directories, including template files, `#BSUB` job settings and optional POTCAR generation with `vaspkit`.

## Requirements

- Python >= 3.10
- `numpy`, `ase`
- `pymatgen` (optional, for point group calculation)
- `deepmd-kit` (optional, for DP potentials)
- `hiphive` (optional, for the ZBL short-range term)

## Installation

```bash
uv pip install -e .
# or
pip install -e .
```

Build a wheel and a source distribution with:

```bash
uv build
```

## Command line

```bash
sansde run params.json            # background; log -> params.log, pid -> ./pid
sansde run params.json --fg       # foreground
sansde xyz2vasp xyz2vasp.json     # export VASP task directories
python -m sansde run params.json --fg
```

## Optimization

`params.json` contains every run parameter; omitted keys keep their defaults.

```json
{
    "SCRIPTS": [
      "from ase.calculators.lj import LennardJones",
      "from deepmd.calculator import DP"
    ],
    "ELEMENT": "Au",
    "LATTICE_CONST": 2.88,
    "MIN_ATOM": 2,
    "MAX_ATOM": 30,
    "ROUND": 5,
    "INIT_DIR": "init",
    "INIT_POP_DIR": "",
    "BEST_DIR": "best",
    "KEEP_SNAPSHOTS": 1,
    "POP_DIR": "Populations",
    "POP_SIZE": 12,
    "MAX_GEN": 500,
    "CONV_STEPS": 50,
    "EDIFF": 1e-05,
    "S_STRICT": 0.95,
    "S_LOOSE": 0.98,
    "CAND_SIZE": 10,
    "POTENTIAL": "DP(model=Path('/home/linyj/02-linyj/potentials/dp/Au3c.pth'))",
    "USE_ZBL": true,
    "ZBL_CUTOFF": 1.5,
    "ZBL_SKIN": 0,
    "RELAX_FMAX": 0.05,
    "RELAX_STEPS": 300
}
```

Key points:

- `SCRIPTS` is a list of Python snippets executed in order before the other parameters are read; they can import calculators or modify `params`, e.g. `["from ase.calculators.lj import LennardJones"]`.
- `POTENTIAL` must be an ASE `Calculator`. A string expression is also accepted, but it must evaluate to a calculator; it is evaluated in the module namespace where `DP`, `LennardJones` and any name imported by `SCRIPTS` are available. For a Lennard-Jones test use `"POTENTIAL": "LennardJones(epsilon=0.4443, sigma=2.6367)"`.
- `USE_ZBL` adds the hiphive ZBL short-range repulsion on top of `POTENTIAL` with range `ZBL_CUTOFF + 2 * ZBL_SKIN`.
- `ROUND` is the number of optimization rounds. With `ROUND > 1`, custom `INIT_DIR`, `BEST_DIR` and `INIT_POP_DIR` are ignored: round 1 reads `init/` and writes `best1/`, round `r` reads `best{r-1}/` and writes `best{r}/`.
- `INIT_POP_DIR`, when non-empty, loads `{element}{atom}*.json` population snapshots instead of `INIT_DIR`.
- `RELAX_FMAX` and `RELAX_STEPS` control the BFGS relaxation of every candidate.

Outputs are written to the current directory:

- `best*/{element}{atom}_best.xyz`: best structure and candidate list.
- `Populations/`: population snapshots.
- `Trajectories/`: trajectory files.
- `<params>.csv`: per-generation statistics.
- `<params>.log`: log file when running in the background.

## VASP export

`sansde xyz2vasp` takes one JSON file:

```json
{
    "input": ["best1", "best2"],
    "template": "input",
    "frame": 3,
    "similarity": [0.99, 0.98],
    "cell": [15.0, 15.0, 15.0, 90.0, 90.0, 90.0],
    "output": "relax.Au",
    "compress": true
}
```

- `input`: xyz directories or files (`str` or list). Default `best`.
- `template`: VASP input template directory. Default `input`.
- `frame`: frames read from each xyz file. Default `3`.
- `similarity`: `[s, s_c1]` deduplication thresholds. Default `[0.99, 0.98]`.
- `cell`: six cell parameters `(a, b, c, alpha, beta, gamma)`. When omitted, clusters with at least 40 atoms use a 20 A cube, smaller clusters keep their own cell, and clusters without a cell use a 15 A cube.
- `output`: output name. Default `relax.<elements>`.
- `compress`: pack the task directories into `<output>.tgz` and remove the directory.

The command writes the merged structures to the current directory before the VASP export:

- `merged.xyz`: all merged/deduplicated clusters as a multi-frame extxyz.
- `merged_xyz/`: one extxyz file per cluster.
- `merged_json/`: one JSON file per cluster with symbols, positions, cell, pbc and `info` (energy, point group, ...).
- `<output>/<formula>.<index>/`: VASP task directories with `POSCAR`, template files and patched `#BSUB -J` / `#BSUB -n` lines.
