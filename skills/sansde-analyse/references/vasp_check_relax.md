# VASP relaxation check and ASE JSON export

`scripts/check_relax.py` reads a tree of VASP relaxation directories and does two things:

- writes the ASE JSON files consumed by the analysis commands;
- prints a reliability CSV and an S/S2 plot so the relaxations can be judged.

```bash
python3 scripts/check_relax.py <vasp_root> --json-out ase_json
python3 scripts/check_relax.py <vasp_root>                    # CSV only
python3 scripts/check_relax.py <vasp_root> --plot             # CSV + PNG plot
python3 scripts/check_relax.py <vasp_root> --plot svg eps     # other formats
```

Images are named `<root-directory>_check_s.<format>` in the current directory, following the convention of the analysis commands. Without `--plot` no image is written.

A calculation directory is any directory that contains at least one of `POSCAR`, `CONTCAR`, `OSZICAR` or `OUTCAR`. The script walks the whole tree, so a root holding several generations or datasets works in one call.

## ASE JSON export

For every calculation directory the script writes `<out>/<system>.json`, where `<system>` is the path relative to the root with `/` replaced by `_` (so `gen11/Au9.0` becomes `gen11_Au9.0`, and a directory directly under the root keeps its own name):

- structure from `CONTCAR`, or from `POSCAR` when `CONTCAR` is missing;
- coordinates unwrapped across periodic boundaries and centered, `pbc=False`;
- `atoms.info["energy"]` is the last ionic-step energy from `OSZICAR` (preferring `E0=`, falling back to `F=`);
- `atoms.info["is_ref"]` is 1 when the directory contains an `IS_REF` marker file, otherwise 0.

This is exactly the format expected by `sansde analyse`, `sansde radial` and `sansde cn`.

## CSV columns

| Column | Meaning |
| --- | --- |
| `system` | calculation directory name |
| `natoms` | atom count |
| `PG_POSCAR`, `PG_CONTCAR` | point groups before and after relaxation |
| `S` | distance similarity between POSCAR and CONTCAR; 1 is identical, above 0.95 is plausible, above 0.98 is stronger |
| `S2` | angle similarity; informational only, not used for the pseudo-minimum decision |
| `(E_CONTCAR-E_POSCAR)/N` | energy change per atom |
| `relax_converged` | 1 when `OUTCAR` contains the structural convergence marker |
| `scf_converged` | 1 when the final ionic step in `OSZICAR` converged |
| `nionic_steps` | number of ionic steps |
| `box` | cubic-equivalent cell edge `V**(1/3)` in Angstrom; empty for a periodic-free cell |
| `max_extent` | largest cluster width along the three principal axes, after unwrapping |
| `box_ratio` | `max_extent / box` |

## Interpretation

1. Reliability, in order:
   - `relax_converged=0`: the ionic relaxation stopped before reaching the accuracy marker. Check the job output for a time limit, a failed SCF, or exhausted ionic steps.
   - `scf_converged=0`: the final SCF cycle did not converge. Look at the `dE` trend at the end of `OSZICAR`.
2. Pseudo-minima: a relaxation that changes the structure means the starting geometry was probably not a minimum on the DFT surface.
   - `PG_POSCAR != PG_CONTCAR` is a strong signal.
   - `S` below 0.95 also indicates a structural change.
   - A clearly lower energy per atom than neighbouring sizes supports the same conclusion.
3. Periodic box too small:
   - `box_ratio > 0.80` means the cluster is close to or across the cell boundary and bonds to its own images; the energy per atom is underestimated and the structure must be discarded.
   - Typical symptom: `max_extent` close to `box`, `box_ratio` near 1.
4. Unexpected termination:
   - search the job output for a scheduler time-limit message;
   - check whether the last `OSZICAR` line has the `E0=` form;
   - inspect the standard output and `log` files.

When reporting, list the affected sizes with their point-group change, `S` and energy change, for example `Au10 Td -> Cs S=0.844 dE/N=-0.0786 199 steps`.
