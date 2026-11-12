# SaNSDE

[![PyPI](https://img.shields.io/pypi/v/sansde.svg)](https://pypi.org/project/sansde)
[![Python](https://img.shields.io/pypi/pyversions/sansde.svg)](https://pypi.org/project/sansde)
[![GitHub](https://img.shields.io/github/stars/Yinjie1/SaNSDE.svg)](https://github.com/Yinjie1/SaNSDE)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)

SaNSDE is a self-adaptive differential evolution with neighborhood search for finding low-energy structures of isolated nanoclusters. The whole workflow is built on [ASE](https://ase-lib.org/) and accepts any ASE calculator object as the potential, so the same input works with DeePMD-kit models, asap3 potentials, or something you write yourself.

## Install

```bash
uv tool install sansde
```

## Quick start

```bash
sansde init                      # write default params.json
sansde check                     # sanity-check the setup
sansde run params.json           # background run, log in params.log
sansde run params.json --fg      # stay in the terminal
```

`sansde init` writes `params.json` with all defaults filled in. A minimal configuration looks like this:

```json
{
  "SCRIPTS": ["from deepmd.calculator import DP"],
  "POTENTIAL": "DP(model=Path('/path/to/model.pth'))",
  "ELEMENT": "Au",
  "MIN_ATOM": 2,
  "MAX_ATOM": 20
}
```

Anything you leave out keeps its default. `sansde check` lists the systems that will be optimized and warns about missing initial structures or existing output files before anything is overwritten.

## Potentials

`POTENTIAL` is an ASE calculator, built from a Python expression after `SCRIPTS` has run. Any ASE calculator works; two common choices:

```jsonc
// DeePMD-kit (requires deepmd-kit)
"SCRIPTS": ["from deepmd.calculator import DP"],
"POTENTIAL": "DP(model=Path('/path/to/model.pth'))"

// Gupta (requires asap3)
"SCRIPTS": ["from sansde.calc import GuptaCalculator"],
"POTENTIAL": "GuptaCalculator({'Au': [10.229, 4.036, 0.2061, 1.790, 2.884]})"
```

`USE_ZBL` is on by default. It adds a short-range repulsion wall so atoms cannot collapse onto each other when the potential is evaluated outside the region it was trained for. Turn it off for potentials that already handle close contacts.

## Alloy clusters

Give a second element and the total cluster size to search fixed-composition alloys:

```json
"ELEMENT": "Au",
"ELEMENT2": "Ag",
"TOTAL_ATOM": 24
```

`MIN_ATOM`..`MAX_ATOM` then count only the Au atoms; the rest of the cluster is Ag, so `TOTAL_ATOM` must be larger than `MAX_ATOM`. Output files use the chemical formula, for example `Ag12Au12_best.xyz`.

## Results

- `best/{formula}_best.xyz` — best candidates found, worst first, best last.
- `better/{formula}_better.xyz` — every structure that improved the search.
- `Populations/` — saved populations; point `INIT_POP_DIR` at them to continue.
- `params.csv` — per-generation statistics, `params.log` — the run log.

With `"ROUND": 2` or more, each round writes its own `best1/`, `best2/`, ... and starts from the previous round's structures.

## Analysis

The analysis commands read ASE JSON structures, one structure per file, such as the `merged_json/` folder written by `sansde xyz2vasp`.

```bash
sansde analyse -i merged_json -o analysis.csv     # energies, shapes, similarity
sansde radial -i merged_json -o radial.csv        # radial layer structure
sansde cn -i merged_json -o coordination.csv      # coordination numbers
```

Add `--plot` for PNG figures (also `svg` or `eps`), `--ref` to compare against reference data, and `--s2` for the angle similarity.

## VASP export

`sansde xyz2vasp config.json` turns the best structures into ready-to-submit VASP task directories, using a template folder that holds `INCAR`, `KPOINTS` and optionally `POTCAR`:

```json
{
  "input": ["best"],
  "template": "vasp_template",
  "output": "relax.Au"
}
```

## License

SaNSDE is licensed under the [MIT](./LICENSE)
