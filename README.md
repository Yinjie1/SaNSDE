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
- `pymatgen` (for point group calculation)
- `matplotlib` (for cluster analysis plots)
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
sansde init                       # write params.json with default values
sansde run params.json            # background; log -> params.log, pid -> ./pid
sansde run params.json --fg       # foreground
sansde xyz2vasp xyz2vasp.json     # export VASP task directories
sansde check                      # inspect *.json in the current directory
sansde analyse -i ase_json        # cluster analysis from ASE JSON structures
sansde radial -i ase_json         # radial layer analysis from ASE JSON structures
sansde cn -i ase_json             # coordination number analysis from ASE JSON structures
python -m sansde run params.json --fg
```

The run itself writes the SANSDE banner to its stdout at the start. A background run therefore has the
banner at the top of `<params>.log`; the launcher only prints the `started background process ...` line.

## Check

`sansde check [--no-banner] [directory]` reads all `*.json` run parameter files in the given directory
(default: the current directory) and prints the merged optimization systems, for example:

```text
read 3 run parameter file(s)
optimization systems: Au2-25, Au30-40
```

Overlapping or adjacent ranges are merged. JSON files that are not run parameter files, cannot be
parsed, or contain an invalid `MIN_ATOM`/`MAX_ATOM` are reported and skipped. When multiple files
define overlapping systems, each overlapping pair is reported with the shared range:

```text
warning: overlapping optimization systems exist
  a.json <-> c.json: Au18-20
```

It also warns when `BEST_DIR`, `BETTER_DIR` or `POP_DIR` already contain files for the target
sizes, because those files may be modified or overwritten by a run:

```text
warning: params.json (Au2-20) has existing output files:
  best/: best/Au2_best.xyz, best/Au5_best.xyz
  better/: better/Au3_better.xyz
  Populations/: Populations/Au2_gen1.json
  these files may be modified or overwritten by the run
```

Initial inputs are checked per parameter file:

- both `INIT_DIR` and `INIT_POP_DIR` empty: report that the run starts from random clusters (no warning);
- `INIT_POP_DIR` empty: report missing `{element}{atom}*.xyz` structures in `INIT_DIR`;
- `INIT_POP_DIR` non-empty: report missing `{element}{atom}*.json` populations, ignoring `INIT_DIR`.

Initial inputs are optional. Missing structures or populations are reported as yellow warnings rather
than errors. Omitted settings use the run defaults (`INIT_DIR="init"`, `INIT_POP_DIR=""`). Only files
whose chemical formula (or JSON `symbol`) matches the target system count as present. Example:

```text
warning: params.json (Au2-5) is missing initial structures in init:
  missing: Au2, Au4, Au5
```

If `POTENTIAL` is set, `check` executes `SCRIPTS`, evaluates `POTENTIAL`, and computes a single-point
energy and forces for a three-atom `{element}3` cluster at `[1,0,0]`, `[0,1,1]`, `[0,0,1]`.
The probe runs in an isolated child process: its stdout is captured and its stderr is discarded, so
output from `SCRIPTS`, `POTENTIAL` construction, the calculator and library loaders (including
atexit CUDA messages) never reaches the check output. Only the result line is printed. Success reports
the calculator class, and failures are reported as red errors:

```text
potential ok: params.json (Au): LennardJones
error: params.json (Au): potential check failed: name 'LennardJones' is not defined
```

If `USE_ZBL` is true (the default), `check` also verifies that `hiphive.calculators.zbl` can be
imported. Only the first JSON with `USE_ZBL` true is checked:

```text
zbl ok: hiphive.calculators.zbl
```

With `ROUND > 1`, the round directories `best1/`, `best2/`, ... are checked instead of `BEST_DIR`.
Output and initial-structure directories are resolved against the current working directory, matching
`sansde run`, even when the parameter JSON files live in another directory:

```bash
sansde check ../configs
```

A SANSDE banner and a `version: x.y.z` line are printed first; the banner is skipped only on a TTY
narrower than 50 columns. Use `--no-banner` to suppress the banner and version line entirely.

Colors are used when stdout is a terminal: the `optimization systems:` label is blue, output/overlap
`warning:` prefixes are red, missing initial-input `warning:` prefixes and skipped files are yellow,
and "no issues" confirmations, `potential ok:` and `zbl ok:` are green. Colors are disabled automatically when output is piped or
redirected, and can be disabled explicitly with `NO_COLOR=1`.

`check` is read-only: it does not create, modify or delete any file.

## Cluster analysis

Run the analysis from the command line:

```bash
sansde analyse -i ase_json -o cluster_analysis.csv
sansde analyse -i ase_json --s2 > cluster_analysis.csv
sansde analyse -i ase_json --ref reference.json --plot -o cluster_analysis.csv
```

Plot files are disabled by default. `--plot` writes PNG files prefixed by the input directory name:
`<name>_energy.png`, `<name>_shape.png`, `<name>_s.png`, `<name>_energy_ref.png` (when reference data
is supplied), and `<name>_eb.png`. Supported formats are `png`, `svg`, and `eps`. Use
`--plot svg`, `--plot eps`, or combine formats such as `--plot png svg eps`. Use `--shape-start` to
set the first atom count in the shape plot, `--ref` for reference data, and `--e1` for an explicit
isolated-atom energy.

ASE JSON structures can also be analyzed through the `sansde.analysis` module:

```python
from sansde.analysis import analyze_clusters, format_cluster_csv
from sansde.io import load_clusters

clusters = load_clusters("ase_json")
rows = analyze_clusters(clusters, compute_s2=True)
csv_text = format_cluster_csv(rows)
```

`load_clusters` keeps the lowest-energy structure for each chemical formula. The analysis rows contain
`symbol`, `natoms`, `is_ref`, `energy`, `energy_per_atom`, `energy_minus_fit`,
`second_difference`, `PG`, `lambda1/2/3`, `shape`, `S`, optional `S2`, optional
`binding_energy_per_atom`, and `energy_diff_ref_per_atom`.

## Radial analysis

```bash
sansde radial -i ase_json -o radial.csv
sansde radial -i ase_json --k 1 12 30 --plot -o radial.csv
```

The CSV contains `system`, `symbol`, `natoms`, `is_ref`, `PG`, `n_layers`, `layer_sizes`,
`layer_pg`, and `radial_distances`. Layer counts are manual; `--k` supplies counts in ascending
atom-count order. When `--k` is omitted, `n_layers`, `layer_sizes` and `layer_pg` are empty.
`--plot` writes `<input-name>_radial.svg`; `--plot png` writes both SVG and PNG.

The same analysis is available from Python:

```python
from sansde.analysis import analyze_radial, format_radial_csv
from sansde.io import load_clusters

clusters = load_clusters("ase_json")
rows = analyze_radial(clusters, method="gap")
csv_text = format_radial_csv(rows)
```

## Coordination analysis

```bash
sansde cn -i ase_json -o coordination.csv
sansde cn -i ase_json --cutoff 3.0 --plot -o coordination.csv
sansde cn -i ase_json --bulk-cn --plot
```

The bonding cutoff defaults to the mean of the first and second bulk neighbor distances of the most
common element; `--cutoff` supplies an explicit value. The CSV contains `system`, `symbol`, `natoms`,
`is_ref`, `cn_atoms`, `cn_ratios`, `cn_avg`, and `bond_length_avg`. `--bulk-cn` computes the bulk
average coordination number for the plot. `--plot` writes `<input-name>_cn.svg`; `--plot png` writes
both SVG and PNG.

The analysis is also available from Python:

```python
from sansde.analysis import analyze_coordination, format_coordination_csv
from sansde.io import load_clusters

clusters = load_clusters("ase_json")
rows = analyze_coordination(clusters, cutoff=3.0)
csv_text = format_coordination_csv(rows)
```

## Initialize configuration

```bash
sansde init
```

`sansde init` writes `params.json` in the current directory with all default values.

## Optimization

`params.json` contains every run parameter; omitted keys keep their defaults.

```json
{
    "EXPORT": {
      "OMP_NUM_THREADS": "6",
      "DP_INTRA_OP_PARALLELISM_THREADS": "6",
      "DP_INTER_OP_PARALLELISM_THREADS": "6"
    },
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
    "CONTINUE": false,
    "BEST_DIR": "best",
    "BETTER_DIR": "better",
    "KEEP_SNAPSHOTS": 1,
    "POP_DIR": "Populations",
    "POP_SIZE": 12,
    "MAX_GEN": 500,
    "CONV_STEPS": 50,
    "EDIFF": 1e-05,
    "S_STRICT": 0.95,
    "S_LOOSE": 0.98,
    "CAND_SIZE": 10,
    "FP_ADAPTIVE": true,
    "LEARNING_PERIOD": 50,
    "CR_PERIOD": 5,
    "CRM_PERIOD": 25,
    "POTENTIAL": "DP(model=Path('/home/linyj/02-linyj/potentials/dp/Au3c.pth'))",
    "USE_ZBL": true,
    "ZBL_CUTOFF": 1.5,
    "ZBL_SKIN": 0,
    "RELAX_FMAX": 0.05,
    "RELAX_STEPS": 300
}
```

Key points:

- `EXPORT` is a dictionary of environment variables applied to the current process before `SCRIPTS`, for example thread limits for DP. [How to control the parallelism of a job?](https://docs.deepmodeling.com/projects/deepmd/en/stable/troubleshooting/howtoset_num_nodes.html)
- `SCRIPTS` is a list of Python snippets executed in order before the other parameters are read; they can import calculators or modify `params`, e.g. `["from ase.calculators.lj import LennardJones"]`.
- `POTENTIAL` must be an ASE `Calculator`. A string expression is also accepted, but it must evaluate to a calculator; it is evaluated in the module namespace where `DP`, `LennardJones` and any name imported by `SCRIPTS` are available. For a Lennard-Jones test use `"POTENTIAL": "LennardJones(epsilon=0.4443, sigma=2.6367)"`.
- `USE_ZBL` adds the hiphive ZBL short-range repulsion on top of `POTENTIAL` with range `ZBL_CUTOFF + 2 * ZBL_SKIN`.
- `ROUND` is the number of optimization rounds. With `ROUND > 1`, custom `INIT_DIR`, `BEST_DIR` and `INIT_POP_DIR` are ignored: round 1 reads `init/` and writes `best1/`, round `r` reads `best{r-1}/` and writes `best{r}/`.
- `INIT_DIR` is scanned for `{element}{atom}*.xyz`; the last frame is used as the initial structure (candidate frames are written worst-first, so the best is last) and its energy is recomputed. An empty `INIT_DIR` disables initial-structure loading.
- `INIT_POP_DIR`, when non-empty, loads `{element}{atom}*.json` population snapshots instead of `INIT_DIR`.
- `CONTINUE` (default `false`): with `INIT_POP_DIR`, only the lowest-energy structure is imported and the rest of the population is generated randomly; set it to `true` to import the whole saved population and continue the run.
- `LEARNING_PERIOD` is the number of generations over which mutation success counts are accumulated; `p` and `fp` are updated only at the end of each period. With `0` no adaptation is performed and both stay at 0.5. `FP_ADAPTIVE` (default `true`) can keep `p` adaptive while freezing `fp` at 0.5.
- `CR_PERIOD` is the number of generations a set of individual crossover rates is kept; the rates are resampled from `N(CRm, 0.1)` every `CR_PERIOD` generations (5 in SaDE). With `0` the rates are never resampled.
- `CRM_PERIOD` is the number of generations over which successful crossover rates are accumulated; `CRm` is updated as their fitness-improvement-weighted mean every `CRM_PERIOD` generations (25 in SaDE). With `0` no update is performed and `CRm` stays at 0.5.
- `CONV_STEPS` is the number of stagnant generations that triggers early stopping. With `0` no early stopping is performed and the run continues until `MAX_GEN`.
- `RELAX_FMAX` and `RELAX_STEPS` control the BFGS relaxation of every candidate.

Outputs are written to the current directory:

- `best*/{element}{atom}_best.xyz`: candidate frames in candidate-list order (worst first, best last).
- `better/{element}{atom}_better.xyz`: one extxyz frame of the current best structure is appended each
  time the structure-change criterion is met. Imported initial structures are not recorded.
- `Populations/`: population snapshots.
- `Trajectories/`: trajectory files.
- `<params>.csv`: per-generation statistics.
- `<params>_pop.csv`: per-generation `symbol`, `atoms`, `gen` and every entry of the population `info` dictionary.
  The file shares the JSON stem (for example `params.json` gives `params_pop.csv`).
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
    "large_cluster_cell": [20.0, 20.0, 20.0, 90.0, 90.0, 90.0],
    "large_cluster_atom_count": 40,
    "large_cluster_unique": true,
    "output": "relax.Au",
    "compress": true
}
```

- `input`: xyz directories or files (`str` or list). Default `best`.
- `template`: VASP input template directory. Default `input`.
- `frame`: number of frames read from the end of each xyz file; the best candidates are stored last. Default `3`.
- `similarity`: `[s, s_c1]` deduplication thresholds. Default `[0.99, 0.98]`.
- `cell`: six cell parameters `(a, b, c, alpha, beta, gamma)`. It applies to clusters smaller than `large_cluster_atom_count`; larger clusters use `large_cluster_cell` (which takes precedence over `cell`). Smaller clusters without a cell use a 15 A cube.
- `large_cluster_cell`: six cell parameters used for clusters with at least `large_cluster_atom_count` atoms; it takes precedence over `cell`. Default `[20, 20, 20, 90, 90, 90]`.
- `large_cluster_atom_count`: atom-count threshold for `large_cluster_cell` and `large_cluster_unique`. Default `40`.
- `large_cluster_unique`: if `true` (default), large clusters are made unique per chemical formula, keeping the lowest-energy one.
- `output`: output name. Default `relax.<elements>`.
- `compress`: pack the task directories into `<output>.tgz` and remove the directory.

The command writes the merged structures to the current directory before the VASP export:

- `merged.xyz`: all merged/deduplicated clusters as a multi-frame extxyz.
- `merged_xyz/`: one extxyz file per cluster.
- `merged_json/`: one JSON file per cluster with symbols, positions, cell, pbc and `info` (energy, point group, ...).
- `<output>/<formula>.<index>/`: VASP task directories with `POSCAR`, template files and patched `#BSUB -J` / `#BSUB -n` lines.
