#!/usr/bin/env python3
"""VASP relaxation check with ASE JSON export.

For every VASP calculation directory under the given root (a directory that
contains any of ``POSCAR``, ``CONTCAR``, ``OSZICAR`` or ``OUTCAR``) the script
prints one CSV row with:

    system,natoms,PG_POSCAR,PG_CONTCAR,S,S2,(E_CONTCAR-E_POSCAR)/N,
    relax_converged,scf_converged,nionic_steps,box,max_extent,box_ratio

With ``--json-out`` it also writes one ``<system>.json`` per calculation
directory in the ASE JSON format used by the analysis commands: an unwrapped,
centered, non-periodic ``Atoms`` object with ``atoms.info["energy"]`` (last
ionic step) and ``atoms.info["is_ref"]`` (1 when the directory holds an
``IS_REF`` marker).

Usage:
    python3 check_relax.py [VASP_ROOT] [--json-out ase_json]
                           [--plot check_s.svg | --no-plot]
"""

from __future__ import annotations

import argparse
import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read
from ase.io.jsonio import write_json

try:
    from pymatgen.core import Molecule
    from pymatgen.symmetry.analyzer import PointGroupAnalyzer
except ImportError:  # pragma: no cover
    Molecule = None
    PointGroupAnalyzer = None

PG_TOL = 0.3
UL = 1.0

STRUCT_CONV_MARKER = (
    "reached required accuracy - stopping structural energy minimisation"
)
IONIC_RE = re.compile(r"^\s*\d+\s+F=")

PLOT_FILE = "check_s.svg"
X_STEP = 5
COLOR_S = "#0000ff"
COLOR_S2 = "#ff0000"
PG_BAND = "0.5"
PG_BAND_ALPHA = 0.30

CALC_FILES = ("OUTCAR", "OSZICAR", "POSCAR", "CONTCAR")


def natural_key(name: str) -> list[int | str]:
    """Return a sort key that orders embedded integers numerically."""
    return [
        int(token) if token.isdigit() else token.lower()
        for token in re.split(r"(\d+)", name)
    ]


def system_name(root: Path, directory: Path) -> str:
    """Return a unique system name from the path relative to the root."""
    try:
        relative = directory.relative_to(root)
    except ValueError:
        relative = directory
    if relative == Path("."):
        return directory.name or str(directory)
    return relative.as_posix().replace("/", "_")


def find_calc_dirs(root: Path) -> list[Path]:
    """Return directories that contain VASP input or output files."""
    dirs = [
        path
        for path in root.rglob("*")
        if path.is_dir() and any((path / name).is_file() for name in CALC_FILES)
    ]
    return sorted(dirs, key=lambda path: natural_key(str(path)))


def file_has(path: Path, marker: str) -> bool:
    """Return whether a file exists and contains ``marker``."""
    if not path.is_file():
        return False
    try:
        with path.open("r", errors="ignore") as handle:
            return any(marker in line for line in handle)
    except OSError:
        return False


def read_lines(path: Path) -> list[str]:
    """Return the lines of a text file, or an empty list when unreadable."""
    if not path.is_file():
        return []
    try:
        with path.open("r", errors="ignore") as handle:
            return handle.readlines()
    except OSError:
        return []


def parse_float(text: str) -> float | None:
    """Parse a Fortran-style float, returning ``None`` on failure."""
    try:
        return float(text.replace("D", "E").replace("d", "e"))
    except ValueError:
        return None


def oszicar_energies(lines: list[str]) -> list[float]:
    """Return the energy of every ionic step, preferring ``E0=`` over ``F=``."""
    values: list[float] = []
    for line in lines:
        if not IONIC_RE.match(line):
            continue
        match = re.search(r"E0=\s*([-+0-9.EeDd]+)", line) or re.search(
            r"F=\s*([-+0-9.EeDd]+)", line
        )
        if match:
            value = parse_float(match.group(1))
            if value is not None:
                values.append(value)
    return values


def n_ionic_steps(lines: list[str]) -> int:
    """Return the index of the last ionic step in ``OSZICAR``."""
    last = 0
    for line in lines:
        tokens = line.split()
        if tokens and tokens[0].isdigit():
            last = int(tokens[0])
    return last


def scf_last_ok(lines: list[str]) -> bool:
    """Return whether the final ``OSZICAR`` line is a converged SCF step."""
    last = ""
    for line in lines:
        if line.strip():
            last = line
    return bool(IONIC_RE.match(last) and "E0=" in last)


def unwrap(positions: np.ndarray, cell: np.ndarray, cutoff: float = 3.5) -> np.ndarray:
    """Unwrap coordinates across periodic boundaries along nearest-neighbor bonds."""
    cell = np.asarray(cell, dtype=float)
    if abs(np.linalg.det(cell)) < 1e-8:
        return positions
    inverse = np.linalg.inv(cell)
    diff = positions[:, None, :] - positions[None, :, :]
    fractional = diff @ inverse
    fractional -= np.round(fractional)
    vectors = fractional @ cell
    distances = np.sqrt((vectors**2).sum(-1))

    count = len(positions)
    unwrapped = np.empty_like(positions)
    visited = np.zeros(count, dtype=bool)
    unwrapped[0] = positions[0]
    visited[0] = True
    queue = [0]
    while queue:
        current = queue.pop(0)
        for neighbor in np.argsort(distances[current]):
            if visited[neighbor] or distances[current, neighbor] >= cutoff:
                continue
            unwrapped[neighbor] = unwrapped[current] + vectors[current, neighbor]
            visited[neighbor] = True
            queue.append(neighbor)
    for index in range(count):
        if not visited[index]:
            candidates = np.where(visited)[0]
            anchor = candidates[np.argmin(distances[index, candidates])]
            unwrapped[index] = unwrapped[anchor] + vectors[anchor, index]
            visited[index] = True
    return unwrapped


def get_cluster(path: Path) -> Atoms | None:
    """Read a VASP structure as an unwrapped, centered, non-periodic cluster."""
    if not path.is_file():
        return None
    try:
        atoms = read(path)
        if len(atoms) == 0:
            return None
        positions = unwrap(
            np.asarray(atoms.get_positions(), dtype=float), atoms.get_cell()
        )
        return Atoms(
            symbols=atoms.get_chemical_symbols(),
            positions=positions - positions.mean(axis=0),
            cell=None,
            pbc=False,
        )
    except Exception:  # noqa: BLE001
        return None


def read_cell(path: Path) -> np.ndarray | None:
    """Return the cell of a VASP structure, or ``None`` when unreadable."""
    if not path.is_file():
        return None
    try:
        return np.asarray(read(path).get_cell(), dtype=float)
    except Exception:  # noqa: BLE001
        return None


def cell_edge(cell: np.ndarray | None) -> float | None:
    """Return the cubic-equivalent edge length ``V**(1/3)`` in Angstrom."""
    if cell is None:
        return None
    try:
        volume = abs(float(np.linalg.det(cell)))
    except Exception:  # noqa: BLE001
        return None
    if volume < 1e-8:
        return None
    return volume ** (1.0 / 3.0)


def max_extent(coords: np.ndarray | None) -> float | None:
    """Return the largest cluster width along the three principal axes."""
    if coords is None:
        return None
    if len(coords) < 2:
        return 0.0
    centered = np.asarray(coords, dtype=float)
    centered = centered - centered.mean(axis=0)
    try:
        _values, vectors = np.linalg.eigh(np.cov(centered.T))
    except np.linalg.LinAlgError:
        return None
    projected = centered @ vectors
    return float((projected.max(axis=0) - projected.min(axis=0)).max())


def pairwise_sorted(coords: np.ndarray) -> np.ndarray:
    """Return all pair distances in ascending order."""
    count = len(coords)
    if count < 2:
        return np.array([])
    diff = coords[:, None, :] - coords[None, :, :]
    distances = np.sqrt((diff**2).sum(-1))
    upper = np.triu_indices(count, k=1)
    return np.sort(distances[upper])


def similarity_s(pos_a: np.ndarray | None, pos_b: np.ndarray | None) -> float | None:
    """Return the distance similarity ``S = 1 / (1 + q / UL)``."""
    if pos_a is None or pos_b is None or len(pos_a) != len(pos_b) or len(pos_a) < 2:
        return None
    distances_a = pairwise_sorted(pos_a)
    distances_b = pairwise_sorted(pos_b)
    if (
        len(distances_a) != len(distances_b)
        or distances_a[0] <= 0
        or distances_b[0] <= 0
    ):
        return None
    count = len(pos_a)
    q = float(
        np.sqrt(
            (2.0 / (count * (count - 1)))
            * np.sum((distances_a / distances_a[0] - distances_b / distances_b[0]) ** 2)
        )
    )
    return 1.0 / (1.0 + q / UL)


def cos_angle(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Return row-wise cosines between two vector arrays."""
    norm_u = np.linalg.norm(u, axis=1)
    norm_v = np.linalg.norm(v, axis=1)
    denominator = norm_u * norm_v
    dot = (u * v).sum(axis=1)
    cosine = np.ones_like(dot)
    valid = denominator > 1e-12
    cosine[valid] = np.clip(dot[valid] / denominator[valid], -1.0, 1.0)
    return cosine


def sorted_cosines(coords: np.ndarray) -> np.ndarray:
    """Return the cosines of all triangle vertex angles in ascending order."""
    count = len(coords)
    if count < 3:
        return np.array([])
    triangles = np.array(list(combinations(range(count), 3)), dtype=int)
    first, second, third = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    point_a, point_b, point_c = coords[first], coords[second], coords[third]
    angle_a = cos_angle(point_b - point_a, point_c - point_a)
    angle_b = cos_angle(point_a - point_b, point_c - point_b)
    angle_c = cos_angle(point_a - point_c, point_b - point_c)
    return np.sort(np.concatenate([angle_a, angle_b, angle_c]))


def similarity_s2(pos_a: np.ndarray | None, pos_b: np.ndarray | None) -> float | None:
    """Return the angle similarity ``S2 = 1 / (1 + 2 q2)``."""
    if pos_a is None or pos_b is None or len(pos_a) != len(pos_b) or len(pos_a) < 3:
        return None
    cosines_a = sorted_cosines(np.asarray(pos_a, dtype=float))
    cosines_b = sorted_cosines(np.asarray(pos_b, dtype=float))
    if cosines_a.size == 0 or cosines_a.size != cosines_b.size:
        return None
    count = len(pos_a)
    q2 = float(
        np.sqrt(
            (2.0 / (count * (count - 1) * (count - 2)))
            * np.sum((cosines_a - cosines_b) ** 2)
        )
    )
    return 1.0 / (1.0 + 2.0 * q2)


def point_group_symbol(atoms: Atoms | None) -> str | None:
    """Return the Schoenflies symbol from pymatgen, or ``None``."""
    if (
        PointGroupAnalyzer is None
        or Molecule is None
        or atoms is None
        or len(atoms) < 1
    ):
        return None
    try:
        analyzer = PointGroupAnalyzer(
            Molecule(atoms.get_chemical_symbols(), atoms.get_positions()),
            tolerance=PG_TOL,
        )
        return analyzer.get_pointgroup().sch_symbol
    except Exception:  # noqa: BLE001
        return None


def make_plot(rows: list[dict[str, object]], outfile: Path) -> None:
    """Plot S and S2 against cluster size, shading point-group changes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(rows, key=lambda row: (row["n"], row["name"]))
    sizes = [int(row["n"]) for row in rows]
    s_values = [row["s"] for row in rows]
    s2_values = [row["s2"] for row in rows]
    changed = [int(row["n"]) for row in rows if row["pg_changed"]]

    figure, axes = plt.subplots(figsize=(10, 6))
    for index, size in enumerate(sorted(set(changed))):
        axes.axvspan(
            size - 0.5,
            size + 0.5,
            color=PG_BAND,
            alpha=PG_BAND_ALPHA,
            lw=0,
            label="PG change (POSCAR->CONTCAR)" if index == 0 else None,
            zorder=0,
        )
    xs = [size for size, value in zip(sizes, s_values) if value is not None]
    ys = [value for value in s_values if value is not None]
    if xs:
        axes.plot(xs, ys, "o-", color=COLOR_S, label="S  (POSCAR vs CONTCAR)")
    xs2 = [size for size, value in zip(sizes, s2_values) if value is not None]
    ys2 = [value for value in s2_values if value is not None]
    if xs2:
        axes.plot(xs2, ys2, "s--", color=COLOR_S2, label="S2 (angle, cos)")

    values = ys + ys2
    if values:
        axes.set_ylim(max(0.0, min(values) - 0.01), 1.002)
    axes.set_xlabel("N")
    axes.set_ylabel("Similarity  S / S2")
    axes.set_xticks(np.arange(0, max(sizes) + 1, X_STEP))
    axes.grid(True, alpha=0.3)
    axes.legend(loc="lower left")
    figure.tight_layout()
    figure.savefig(outfile, format="svg" if outfile.suffix.lower() == ".svg" else None)
    plt.close(figure)


def main() -> int:
    """Run the check and optional JSON export."""
    parser = argparse.ArgumentParser(description="VASP relaxation reliability check.")
    parser.add_argument("path", nargs="?", default=".", metavar="VASP_ROOT")
    parser.add_argument(
        "--json-out",
        default=None,
        metavar="DIR",
        help="also write one ASE JSON file per calculation directory",
    )
    parser.add_argument("--plot", default=PLOT_FILE, metavar="OUT.svg")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    root = Path(args.path)
    if not root.is_dir():
        print(f"error: not a directory: {root}", file=sys.stderr)
        return 1

    json_dir = Path(args.json_out) if args.json_out else None
    if json_dir is not None:
        json_dir.mkdir(parents=True, exist_ok=True)

    print(
        "system,natoms,PG_POSCAR,PG_CONTCAR,S,S2,(E_CONTCAR-E_POSCAR)/N,"
        "relax_converged,scf_converged,nionic_steps,box,max_extent,box_ratio"
    )

    plot_rows: list[dict[str, object]] = []
    exported = 0
    for directory in find_calc_dirs(root):
        name = system_name(root, directory)
        lines = read_lines(directory / "OSZICAR")

        atoms_p = get_cluster(directory / "POSCAR")
        atoms_c = get_cluster(directory / "CONTCAR")
        natoms = (
            len(atoms_c)
            if atoms_c is not None
            else (len(atoms_p) if atoms_p is not None else 0)
        )

        pg_p = point_group_symbol(atoms_p)
        pg_c = point_group_symbol(atoms_c)

        s_pc = similarity_s(
            None if atoms_p is None else atoms_p.get_positions(),
            None if atoms_c is None else atoms_c.get_positions(),
        )
        s2_pc = similarity_s2(
            None if atoms_p is None else atoms_p.get_positions(),
            None if atoms_c is None else atoms_c.get_positions(),
        )

        energies = oszicar_energies(lines)
        energy_per_atom = (
            ((energies[-1] - energies[0]) / natoms) if (energies and natoms) else None
        )

        relax_ok = int(file_has(directory / "OUTCAR", STRUCT_CONV_MARKER))
        scf_ok = int(scf_last_ok(lines))
        steps = n_ionic_steps(lines)

        cell = read_cell(directory / "CONTCAR")
        if cell is None:
            cell = read_cell(directory / "POSCAR")
        box = cell_edge(cell)
        extent = max_extent(None if atoms_c is None else atoms_c.get_positions())
        ratio = (extent / box) if (box and extent is not None) else None

        print(
            ",".join(
                [
                    name,
                    str(natoms),
                    "" if pg_p is None else pg_p,
                    "" if pg_c is None else pg_c,
                    "" if s_pc is None else f"{s_pc:.8f}",
                    "" if s2_pc is None else f"{s2_pc:.8f}",
                    "" if energy_per_atom is None else f"{energy_per_atom:.8f}",
                    str(relax_ok),
                    str(scf_ok),
                    str(steps),
                    "" if box is None else f"{box:.3f}",
                    "" if extent is None else f"{extent:.3f}",
                    "" if ratio is None else f"{ratio:.3f}",
                ]
            )
        )
        plot_rows.append(
            {
                "name": name,
                "n": natoms,
                "s": s_pc,
                "s2": s2_pc,
                "pg_changed": (pg_p is not None and pg_c is not None and pg_p != pg_c),
            }
        )

        if json_dir is not None:
            cluster = atoms_c if atoms_c is not None else atoms_p
            if cluster is not None:
                if energies:
                    cluster.info["energy"] = float(energies[-1])
                cluster.info["is_ref"] = 1 if (directory / "IS_REF").is_file() else 0
                write_json(json_dir / f"{name}.json", cluster)
                exported += 1

    if json_dir is not None:
        if exported == 0:
            print(
                f"[error] no VASP calculation directories under {root}", file=sys.stderr
            )
        else:
            print(f"[json] {exported} systems -> {json_dir}", file=sys.stderr)

    if not args.no_plot and plot_rows:
        make_plot(plot_rows, Path(args.plot))
        print(f"[plot] {args.plot}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
