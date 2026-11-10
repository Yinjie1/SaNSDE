---
name: sansde-analyse
description: "Post-process nanocluster results with the `sansde` CLI: energy and magic-number candidates, point groups and overall shape, similarity S/S2, radial layers, coordination numbers, and average bond lengths. Inputs are ASE JSON structure directories. Use when the user has cluster optimization results in ASE JSON and wants energy, shape, similarity, radial layer, or coordination analysis. Treat `sansde` as a black-box CLI."
license: MIT
compatibility: Requires the `sansde` executable on PATH. Inputs are ASE JSON files containing non-periodic `Atoms` objects.
---

# SaNSDE cluster analysis

`sansde` is a black-box command-line tool. This skill covers input preparation, the three analysis
commands, CSV and image outputs, and interpretation. Reading the program source is unnecessary.

Available commands:

```bash
sansde analyse -i <ase_json_dir>   # energy, shape, similarity
sansde radial  -i <ase_json_dir>   # radial distances and layers
sansde cn      -i <ase_json_dir>   # coordination numbers and bond lengths
```

## Scope

If the user does not explicitly request specific metrics, perform exactly these analyses: `sansde
analyse`, the two-pass `sansde radial` workflow, and `sansde cn`. Follow the recommended workflow strictly. Do not add other analyses, extra derived quantities, extra plots, or extra output files.

## When to use

- Inspect energy trends and identify magic-number candidates.
- Check point groups, overall shape, and similarity between adjacent sizes.
- Analyze radial layers and point groups of individual layers.
- Analyze coordination-number distributions, average coordination, and average bond length.
- Compare energies against reference data.

## Input data

The input is a directory whose JSON files each contain one ASE `Atoms` object.

Structure requirements:

- Clusters are non-periodic.
- Coordinates are unwrapped and centered. Folded coordinates across periodic boundaries are invalid.
- `atoms.info["energy"]` stores the total energy in eV.
- `atoms.info["is_ref"]` is optional. A value of 1 marks a reference structure; missing values count as 0.
- If several JSON files share a chemical formula, only the lowest-energy structure is kept.

Do not mix VASP inputs or outputs into this directory. Structures from other programs must first be
converted to the same ASE JSON format.

## Common conventions

- `-i, --input` selects the ASE JSON directory.
- `-o, --output` selects the CSV path. Without it, the CSV is written to standard output.
- No images are written by default.
- For `sansde analyse`, `--plot` writes PNG files; `--plot svg` writes SVG files instead.
- For `sansde radial` and `sansde cn`, `--plot` writes SVG files; `--plot png` writes both SVG and
  PNG files.
- Image names use the input directory name as prefix. For `test/ase_json`, images start with
  `ase_json_`.

Use `-o` in automation so the CSV stays separate from other output.

## 1. Energy, shape, and similarity

```bash
sansde analyse -i ase_json -o cluster_analysis.csv --plot
```

Options:

- `--s2` also computes the angle similarity `S2`.
- `--ref <path>` supplies reference data as an ASE JSON directory or a CSV file with `natoms` and
  either `energy` or `energy_per_atom`.
- `--e1 <value>` supplies the isolated-atom energy. Without it, the energy of the N=1 structure is
  used.
- `--shape-start <N>` sets the first atom count shown in the shape plot.
- `--plot` writes PNG files. `--plot svg` writes SVG files instead.

CSV columns:

- `system`: system name.
- `symbol`: chemical formula.
- `natoms`: atom count.
- `is_ref`: 0 for calculated data, 1 for reference data.
- `energy`, `energy_per_atom`: total and per-atom energy.
- `energy_minus_fit`: deviation from the fitted energy curve.
- `second_difference`: second energy difference.
- `PG`: point group.
- `lambda1`, `lambda2`, `lambda3`: overall shape eigenvalues.
- `shape`: `球形`, `饼形`, or `直线形`.
- `S`: distance similarity to the N-1 cluster.
- `S2`: angle similarity, present only with `--s2`.
- `binding_energy_per_atom`: per-atom binding energy; omitted when no isolated-atom energy exists.
- `energy_diff_ref_per_atom`: per-atom energy difference from reference data; empty without `--ref`.

Interpretation:

- Valleys in `energy_minus_fit` and peaks in `second_difference` are magic-number candidates.
- `second_difference` uses true adjacent sizes. Missing sizes stay empty; do not fill them manually.
- `S` close to 1 means the N cluster resembles the N-1 cluster plus one atom. Values above 0.95 are plausible; values above 0.98 are stronger.
- Read `S2` together with `S`. Both should be high for a reliable add-one-atom relation.
- `shape` uses the three normalized eigenvalues. Similar values indicate a compact, sphere-like cluster; one clearly smaller value indicates a disk-like cluster; one clearly larger value indicates a linear cluster.
- Gray regions in plots mark sizes affected by reference structures. Do not base main conclusions only on those sizes.
- `is_ref=1` marks reference structures. Cross-dataset energy offsets appear in `energy_diff_ref_per_atom`; flag them in reports.

Images:

- `<name>_energy.png`: `E - E_fit` and second energy difference.
- `<name>_shape.png`: shape eigenvalues.
- `<name>_s.png`: similarity `S` and optional `S2`.
- `<name>_energy_ref.png`: per-atom energy difference from reference data; written with `--ref`.
- `<name>_eb.png`: per-atom binding energy; written when an isolated-atom energy is available.
- `--plot svg` writes the same names with `.svg`.

## 2. Radial distances and layers

Layer counts are specified manually. The recommended workflow has two passes.

### First pass: inspect radial distances

```bash
sansde radial -i ase_json -o radial_pass1.csv --plot png
```

Without `--k`:

- `radial_distances` is still written.
- `n_layers`, `layer_sizes`, and `layer_pg` are empty.
- `<name>_radial.png` shows radial-distance clustering.

### Second pass: specify layer counts and recompute

Determine the layer count for each system from the radial-distance plot and `radial_distances`, then pass counts in ascending atom-count order:

```bash
sansde radial -i ase_json --k K1 K2 K3 -o radial_final.csv --plot png
```

Rules:

- The number of `--k` values must equal the number of analyzed systems.
- Systems are ordered by atom count.
- One structure per chemical formula is kept, so the system count may be smaller than the JSON file count.
- Use the second-pass CSV for the final report.

CSV columns:

- `system`, `symbol`, `natoms`, `is_ref`: system information.
- `PG`: point group of the whole cluster.
- `n_layers`: layer count.
- `layer_sizes`: atoms per layer, for example `(1 12 30)`.
- `layer_pg`: point group of each layer; single-atom layers use `null`.
- `radial_distances`: ascending distances from the geometric center.

Interpretation:

- Radii clustered near a few values indicate clear layers.
- A continuous distribution usually means one layer.
- `layer_pg` compares local symmetry of individual layers.
- Scatter color in the image shows radial-distance overlap at the same N; red means more overlap.

## 3. Coordination numbers and average bond length

```bash
sansde cn -i ase_json -o coordination.csv --plot png
sansde cn -i ase_json --cutoff 3.0 --plot png -o coordination.csv
sansde cn -i ase_json --bulk-cn --plot png
```

Options:

- `--cutoff <A>` sets the bonding cutoff. Without it, the cutoff is the mean of the first and second
  bulk neighbor distances of the most common element.
- `--bulk-cn` and `--no-bulk-cn` control whether the bulk average coordination number is computed and plotted.
- `--plot png` writes both SVG and PNG.

CSV columns:

- `system`, `symbol`, `natoms`, `is_ref`: system information.
- `cn_atoms`: atom counts for each coordination number. For example, `(0 0 2 0 3)` means two atoms have coordination 3 and three atoms have coordination 5.
- `cn_ratios`: the same distribution as percentages.
- `cn_avg`: average coordination number over all atoms.
- `bond_length_avg`: mean distance over all bonded pairs inside the cutoff; empty when there are no bonds.

Interpretation:

- Compare coordination distributions across sizes with `cn_ratios`.
- Average coordination approaching the bulk value indicates a weaker surface effect.
- Average bond length versus size shows contraction or surface relaxation.
- With `--bulk-cn`, the left panel includes a dashed bulk average line.
- A custom `--cutoff` changes coordination numbers and bond lengths. Report the value.

Image:

- `<name>_cn.svg`: average coordination and bond length on the left, coordination-ratio stacked bars on the right.

## Recommended workflow

1. Confirm the ASE JSON directory, elements, energy units, and reference data.
2. Run `sansde analyse`, save the CSV, and use `--plot` when images are needed.
3. Run `sansde radial` once without `--k`, inspect radial distances and images, and determine layer counts.
4. Run `sansde radial` again with `--k`; use the second-pass CSV as the final result.
5. Run `sansde cn`, setting `--cutoff` or `--bulk-cn` when required.
6. Summarize energy trends, magic-number candidates, shape, similarity, layers, and coordination.

## Validation checklist

- [ ] Empty CSV: confirm that `-i` points to a directory containing `*.json`.
- [ ] Empty `energy` column: check `atoms.info["energy"]`.
- [ ] Empty layer columns: confirm that `--k` was supplied.
- [ ] Empty `energy_diff_ref_per_atom`: confirm that `--ref` was supplied and covers the requested sizes.
- [ ] No images: confirm that `--plot` was supplied. For `sansde analyse`, the default is PNG; `--plot svg` requests SVG.
- [ ] Unexpected similarity, radial distances, or coordination numbers: verify unwrapped, centered, non-periodic coordinates.
- [ ] Fewer systems than JSON files: duplicate chemical formulas keep only the lowest-energy structure.

## Report checklist

Include:

- [ ] Input directory, element, size range, and reference-data usage.
- [ ] Lowest total energy and per-atom energy for each size.
- [ ] Magic-number candidates from `energy_minus_fit` valleys and `second_difference` peaks.
- [ ] Point groups and shape classification.
- [ ] Outlier sizes in `S` and optional `S2`.
- [ ] Radial layers and the `--k` values used in the second pass.
- [ ] Coordination distributions, average coordination, and average bond length; report custom cutoff values.
- [ ] Sizes with `is_ref=1` and the range of conclusions affected by reference data.

## Constraints

- This tool only post-processes existing ASE JSON data. It does not run relaxation or VASP.
- Treat input data as read-only. Do not modify or delete original JSON files.
- Use commands, CSV files, and images as the black-box interface.
- Do not substitute `is_ref=1` structures for independent calculated results.
- Keep missing sizes empty. Do not invent energies or similarities across gaps.
