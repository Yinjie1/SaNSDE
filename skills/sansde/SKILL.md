---
name: sansde
description: "Run nanocluster global optimization with the `sansde` CLI: initialize parameters, validate inputs and outputs, run the self-adaptive differential evolution workflow, and optionally export selected structures to VASP tasks. Use when the user wants to optimize nanoclusters with LJ or DeePMD potentials. Treat `sansde` as a black-box CLI."
license: MIT
compatibility: Requires the `sansde` executable on PATH. The potential is configured through `SCRIPTS` and `POTENTIAL`; optional DeePMD features require their own installation.
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

Use `init`, `check`, and `run` for the default optimization workflow. Use `xyz2vasp` only when the user explicitly requests VASP task generation.

## Before running

Confirm these items with the user or from the working context:

- Element and atom-count range.
- Potential: any ASE calculator configured through `SCRIPTS` and `POTENTIAL`, including any required model path or calculator parameters.
- Working directory where results should be written.
- Initial structures or initial population, if available.
- Available compute resources and expected runtime.

Run `sansde init` and edit the generated `params.json`. Omitted keys use built-in defaults.

## Initialize parameters

```bash
sansde init        # write params.json with all defaults
sansde init -h     # parameter manual: type, default and meaning of every key
```

`params.json` uses strict JSON. Omitted keys keep their defaults. Run `sansde init -h` when you need the type or meaning of a parameter; do not guess. Every parameter is printed on one line, so a few entries can be found by filtering the output, for example `sansde init -h | grep -E '^(CELL|PBC) '`.

The potential is configured with `SCRIPTS` and `POTENTIAL`. For Lennard-Jones, for example:

```jsonc
{
  "SCRIPTS": ["from ase.calculators.lj import LennardJones"],
  "POTENTIAL": "LennardJones(epsilon=0.4443, sigma=2.6367)"
}
```

## Prepare initial structures

Two input modes are available:

- `INIT_DIR`: each matching `{element}{atom}*.xyz` file supplies an initial structure. Only the lowest-energy structure (the last frame) is added to the population; the remaining structures are generated randomly.
- `INIT_POP_DIR`: matching `{element}{atom}*.json` files supply saved populations. With `CONTINUE=true`, the whole population and its adaptive state are imported; with `false`, only the lowest-energy structure is imported and the adaptive state is reset.

If both directories are empty or no matching file exists, the population is generated randomly.
With `ROUND > 1`, round 1 reads `init/`, and later rounds read the previous `best*` directory.

Keep input structures non-periodic and consistent with the element in `params.json`.

## Check the setup

```bash
sansde check
```

`check` is read-only. It reports:

- parameter types in the JSON files, one error per wrong value; prints `parameter types ok` when every value has the expected type;
- merged optimization systems from the JSON files in the current directory;
- overlapping systems between different JSON files;
- existing output files that may be overwritten;
- missing initial structures or populations;
- whether the configured potential can be invoked;
- whether the ZBL settings are valid when `USE_ZBL` is true.

Act on these results:

- Type errors, potential errors and ZBL errors must be fixed before starting a run.
- Missing initial structures are warnings. Random initialization will be used.
- Existing output files may be modified. Use a new working directory or back up important results.
- Overlapping JSON files in the same directory can produce conflicting output names.

Use `sansde check --no-banner` when a clean log is needed.

## Run optimization

Before running, show the prepared `params.json` to the user and wait for explicit confirmation. Do not start a run on your own.

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
sansde xyz2vasp -h            # manual: type, default and meaning of every key
```

Run `sansde xyz2vasp -h` when you need the meaning of a setting; do not guess. Filter the output to look up a few settings, for example `sansde xyz2vasp -h | grep -E '^(cell|frame) '`.

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
- Prepare the run JSON, show it to the user, and wait for explicit confirmation before `sansde run`.
- Use `sansde check` before long runs.
- Do not add analysis commands or extra output files that the user did not request.
