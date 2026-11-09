---
name: sansde
description: "Run nanocluster global optimization with the `sansde` CLI: initialize parameters, validate inputs and outputs, run the self-adaptive differential evolution workflow, and optionally export selected structures to VASP tasks. Use when the user wants to optimize nanoclusters with LJ or DeePMD potentials. Treat `sansde` as a black-box CLI."
license: MIT
compatibility: Requires the `sansde` executable on PATH. The potential is configured through `SCRIPTS` and `POTENTIAL`; optional DeePMD and hiphive features require their own installations.
---

# SaNSDE optimization

`sansde` is a black-box command-line tool for nanocluster global optimization. This skill covers the default workflow:

```bash
sansde init       # write params.json
sansde check      # validate configuration and inputs
sansde run        # start optimization
sansde xyz2vasp   # optional VASP task export
```

## Scope

Use `init`, `check`, and `run` for the default optimization workflow. Use `xyz2vasp` only when the user explicitly requests VASP task generation. ~~Do not add unrelated analyses, extra commands, or unrequested output files.~~

## Before running

Confirm these items with the user or from the working context:

- Element and atom-count range.
- Potential: ~~Lennard-Jones or a DeePMD model, including the model path.~~
- Working directory where results should be written.
- Initial structures or initial population, if available.
- Available compute resources and expected runtime.

Run `sansde init` and edit the generated `params.json`. Omitted keys use built-in defaults.

## Initialize parameters

```bash
sansde init
```

This writes `params.json` in the current directory. Use strict JSON.

```json
{
  "SCRIPTS": ["from deepmd.calculator import DP"],
  "ELEMENT": "Au",
  "LATTICE_CONST": 2.88,
  "MIN_ATOM": 2,
  "MAX_ATOM": 20,
  "ROUND": 1,
  "INIT_DIR": "init",
  "INIT_POP_DIR": "",
  "CONTINUE": false,
  "BEST_DIR": "best",
  "BETTER_DIR": "better",
  "POP_DIR": "Populations",
  "POP_SIZE": 12,
  "MAX_GEN": 500,
  "CONV_STEPS": 100,
  "EDIFF": 1e-5,
  "CAND_SIZE": 10,
  "POTENTIAL": "DP(model=Path('/path/to/model.pth'))",
  "USE_ZBL": true,
  "ZBL_CUTOFF": 1.5,
  "ZBL_SKIN": 0,
  "RELAX_FMAX": 0.05,
  "RELAX_STEPS": 300
}
```

Key parameters:

- `SCRIPTS`: Python statements executed in order before the run. Use them to import ASE calculators.
- `POTENTIAL`: expression that evaluates to an ASE calculator. `Path` is available.
- `ELEMENT`: element symbol.
- `LATTICE_CONST`: initial bond-length scale for random clusters.
- `MIN_ATOM`, `MAX_ATOM`: inclusive size range.
- `ROUND`: number of optimization rounds.
- `INIT_DIR`: directory containing initial `{element}{atom}*.xyz` files.
- `INIT_POP_DIR`: directory containing population snapshot `*.json` files.
- `CONTINUE`: with `INIT_POP_DIR`, import the whole population instead of only the best structure.
- `POP_SIZE`: population size.
- `MAX_GEN`: maximum generations.
- `CONV_STEPS`: stagnant generations before early stop; 0 disables early stopping.
- `EDIFF`: energy tolerance for stagnation.
- `CAND_SIZE`: candidate archive size.
- `BEST_DIR`, `BETTER_DIR`, `POP_DIR`: output directories.
- `USE_ZBL`: add the hiphive ZBL short-range term on top of `POTENTIAL`.
- `RELAX_FMAX`, `RELAX_STEPS`: local relaxation settings.

For Lennard-Jones, use:

```jsonc
{
  "SCRIPTS": ["from ase.calculators.lj import LennardJones"],
  "POTENTIAL": "LennardJones(epsilon=0.4443, sigma=2.6367)"
}
```

## Prepare initial structures

Two input modes are available:

- `INIT_DIR`: each matching `{element}{atom}*.xyz` file supplies an initial structure. The last frame is used as the ~~population seed~~.
- `INIT_POP_DIR`: matching `{element}{atom}*.json` files supply saved populations. With `CONTINUE=true`, the whole population is imported; with `false`, only the lowest-energy structure is imported.

If both directories are empty or no matching file exists, the population is generated randomly.
With `ROUND > 1`, round 1 reads `init/`, and later rounds read the previous `best*` directory.

Keep input structures non-periodic and consistent with the element in `params.json`.

## Check the setup

```bash
sansde check
```

`check` is read-only. It reports:

- merged optimization systems from the JSON files in the current directory;
- overlapping systems between different JSON files;
- existing output files that may be overwritten;
- missing initial structures or populations;
- whether the configured potential can be invoked;
- whether the ZBL module can be imported when `USE_ZBL` is true.

Act on these results:

- Potential errors and ZBL errors must be fixed before starting a run.
- Missing initial structures are warnings. Random initialization will be used.
- Existing output files may be modified. Use a new working directory or back up important results.
- Overlapping JSON files in the same directory can produce conflicting output names.

Use `sansde check --no-banner` when a clean log is needed.

## Run optimization

Background run, default:

```bash
sansde run params.json
```

Background runs write:

- `<params>.log`: run log.
- `pid`: one process ID per line.

Monitor the log until `program total_time_s=` appears or the process exits.

The run stops at `MAX_GEN` or after `CONV_STEPS` stagnant generations. Use `CONV_STEPS=0` to run the full `MAX_GEN`.

## Outputs

- `best*/{element}{atom}_best.xyz`: candidate structures for each size. Frames are ordered from higher to lower energy, with the best structure last.
- `better/{element}{atom}_better.xyz`: improved structures appended when the structure-change criterion is met.
- `Populations/{element}{atom}_gen*.json`: population snapshots.
- `Trajectories/`: best-structure trajectories.
- `<params>.csv`: per-generation statistics.
- `<params>_pop.csv`: per-generation population state.
- `<params>.log`: background log.
- `pid`: background process IDs.

With `ROUND > 1`, round directories are `best1/`, `best2/`, and so on.

## Optional: export VASP tasks

Use this step only when the user asks for VASP inputs.

```bash
sansde xyz2vasp xyz2vasp.json
```

Example configuration:

```json
{
  "input": ["best"],
  "template": "input",
  "frame": 3,
  "similarity": [0.99, 0.98],
  "cell": [15.0, 15.0, 15.0, 90.0, 90.0, 90.0],
  "large_cluster_cell": [20.0, 20.0, 20.0, 90.0, 90.0, 90.0],
  "large_cluster_atom_count": 40,
  "large_cluster_unique": true,
  "output": "relax.Au",
  "compress": false
}
```

- `input`: xyz files or directories.
- `template`: directory containing `INCAR`, `KPOINTS`, and optionally `POTCAR`.
- `frame`: number of frames read from the end of each xyz file. The best candidates are last.
- `cell`: cell for clusters below `large_cluster_atom_count`.
- `large_cluster_cell`: cell for clusters at or above the threshold.
- `large_cluster_unique`: keep only the lowest-energy structure per formula for large clusters.
- `output`: output directory for task folders.
- `compress`: pack the output directory into a `.tgz` archive.

The export writes `merged.xyz`, `merged_xyz/`, and `merged_json/` before creating task directories.
This command prepares VASP inputs. It does not run VASP.

## Validation checklist

- [ ] `sansde check` reports no potential or ZBL errors.
- [ ] The working directory is the intended output location.
- [ ] Initial structure files have the expected element and atom count.
- [ ] `MAX_GEN`, `CONV_STEPS`, and `POP_SIZE` match the requested runtime and accuracy.
- [ ] Background log contains a normal finish line.
- [ ] `best*/` and `Populations/` contain the expected sizes.
- [ ] Existing output files were reviewed before starting a run.

## Constraints

- Treat `sansde` as a black-box command-line tool.
- Do not run VASP or relaxation workflows in this skill.
- Do not modify or delete existing input structures without explicit permission.
- Use `sansde check` before long runs.
- Do not add analysis commands or extra output files that the user did not request.
