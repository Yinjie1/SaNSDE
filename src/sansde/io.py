"""Utilities for reading xyz directories and writing VASP task directories."""

import csv
import json
import re
import shutil
import subprocess
import tarfile
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.io import iread, write

from sansde.pop import Clusters, MultiAtoms

__all__ = [
    "DEFAULT_RUN_PARAMS",
    "append_stats_row",
    "load_clusters",
    "load_ref_energies",
    "merge_multi_atoms",
    "merge_xyz_directories",
    "read_run_params",
    "write_vasp_tasks",
]

LARGE_CLUSTER_ATOM_COUNT: int = 40
LARGE_CLUSTER_CELL: tuple[float, ...] = (20.0, 20.0, 20.0, 90.0, 90.0, 90.0)
DEFAULT_CELL: tuple[float, ...] = (15.0, 15.0, 15.0, 90.0, 90.0, 90.0)
LARGE_CLUSTER_UNIQUE: bool = True


# ===== run IO: configuration and statistics =====

DEFAULT_RUN_PARAMS: dict[str, Any] = {
    'EXPORT': {},
    'SCRIPTS': ["from ase.calculators.lj import LennardJones"],
    'ELEMENT': 'Au',
    'ELEMENT2': '',
    'TOTAL_ATOM': 0,
    'LATTICE_CONST': 2.88,
    'MIN_ATOM': 3,
    'MAX_ATOM': 20,
    'ROUND': 1,
    'INIT_DIR': 'init',
    'INIT_POP_DIR': '',
    'BEST_DIR': 'best',
    'BETTER_DIR': 'better',
    'KEEP_SNAPSHOTS': 1,
    'POP_DIR': 'Populations',
    'POP_SIZE': 12,
    'MAX_GEN': 500,
    'CONV_STEPS': 250,
    'EDIFF': 1e-5,
    'S_STRICT': 0.95,
    'S_LOOSE': 0.98,
    'CAND_SIZE': 10,
    'FP_ADAPTIVE': True,
    'LEARNING_PERIOD': 50,
    'CR_PERIOD': 5,
    'CRM_PERIOD': 25,
    'CONTINUE': False,
    'POTENTIAL': "LennardJones(epsilon=0.4443, sigma=2.6367)",
    'USE_ZBL': True,
    'ZBL_CUTOFF': 1.5,
    'ZBL_SMOOTH': True,
    'RELAX_FMAX': 0.05,
    'RELAX_STEPS': 300,
}


def read_run_params(params_path: Path) -> dict[str, Any]:
    """Read run parameters from a JSON file.

    Values missing from the file are filled with the built-in defaults, so the
    returned dictionary always contains every run parameter.

    Parameters
    ----------
    params_path : Path
        Path to the JSON parameter file.

    Returns
    -------
    dict[str, Any]
        Parsed parameters merged over the built-in defaults.

    Raises
    ------
    FileNotFoundError
        If ``params_path`` does not exist.
    TypeError
        If the JSON content is not an object.
    """
    if not params_path.is_file():
        raise FileNotFoundError(f"run parameter file not found: {params_path}")
    with params_path.open('r', encoding='utf-8') as handle:
        user_params = json.load(handle)
    if not isinstance(user_params, dict):
        raise TypeError(f"{params_path} must contain a JSON object")
    print(f"load parameters from {params_path}")
    return {**DEFAULT_RUN_PARAMS, **user_params}


def append_stats_row(
    stats_path: Path,
    symbol: str,
    atom: int,
    gen: int,
    pop: MultiAtoms,
    best_energy: float,
    best_pg: str | None,
    best_drop: float | None,
    similarity: float | None,
    stagnation_count: int,
    candidate_size: int,
    elapsed_time: float,
) -> None:
    """Append one generation's statistics to the run CSVs."""
    candidates = list(pop.candidate) # type: ignore
    n_sub = min(max(candidate_size - 1, 0), 2)
    # Candidates are ordered worst-first, so candidate[-1] is the best and the
    # next-best entries are found by walking backwards.
    sub_candidates = list(reversed(candidates[:-1]))[:n_sub]
    sub_energies = [float(cluster.info['energy']) for cluster in sub_candidates]
    sub_pgs = [
        "" if cluster.info.get('PG') is None else str(cluster.info['PG'])
        for cluster in sub_candidates
    ]

    header = ["symbol", "atoms", "gen", "e_best"]
    header.extend(f"e_cand{i}" for i in range(1, n_sub + 1))
    header.extend(["delta_e", "PG"])
    header.extend(f"PG_cand{i}" for i in range(1, n_sub + 1))
    header.extend(["S", "best_streak", "time"])

    row = [symbol, atom, gen, f"{best_energy:.8f}"]
    row.extend(
        "" if energy is None else f"{energy:.8f}"
        for energy in sub_energies
    )
    row.extend([""] * (n_sub - len(sub_energies)))
    row.append("" if best_drop is None else f"{best_drop:.8f}")
    row.append("" if best_pg is None else str(best_pg))
    row.extend(sub_pgs)
    row.extend([""] * (n_sub - len(sub_pgs)))
    row.append("" if similarity is None else f"{similarity:.2f}")
    row.extend([
        stagnation_count,
        f"{elapsed_time:.0f}",
    ])

    write_header = not stats_path.exists() or stats_path.stat().st_size == 0
    with stats_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)

    pop_stats_path = stats_path.with_stem(f"{stats_path.stem}_pop")
    pop_items = [
        (key, value)
        for key, value in pop.info.items()
        if key not in ('cr_rec', 'delta_rec')
    ]
    pop_header = ["symbol", "atoms", "gen", *(key for key, _ in pop_items)]
    pop_row: list[object] = [symbol, atom, gen]
    for _, value in pop_items:
        if isinstance(value, (list, dict)):
            pop_row.append(json.dumps(value))
        else:
            pop_row.append(value)
    pop_write_header = (
        not pop_stats_path.exists() or pop_stats_path.stat().st_size == 0
    )
    with pop_stats_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if pop_write_header:
            writer.writerow(pop_header)
        writer.writerow(pop_row)


# ===== cluster analysis IO: ASE JSON loading =====

def _natural_key(value: str) -> list[object]:
    """Split a string into integer and lowercase text chunks for sorting."""
    return [
        int(chunk) if chunk.isdigit() else chunk.lower()
        for chunk in re.split(r"(\d+)", value)
    ]


def load_clusters(directory: str | Path) -> Clusters:
    """Load ASE JSON structures into a :class:`Clusters` container.

    One JSON file contains one ``Atoms`` object. The ``system`` metadata is
    set from the file stem when absent, and ``is_ref`` defaults to 0. When
    several files share a chemical formula, the lowest-energy structure is
    kept.

    Parameters
    ----------
    directory : str | pathlib.Path
        Directory containing ``*.json`` files, or a single JSON file.

    Returns
    -------
    Clusters
        Clusters in the order returned by ``load_json`` calls.

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    TypeError
        If a file does not contain an ASE ``Atoms`` object.
    ValueError
        If a file cannot be read or no structure is loaded.
    """
    source = Path(directory)
    if source.is_dir():
        files = sorted(
            source.glob("*.json"),
            key=lambda path: _natural_key(path.stem),
        )
    elif source.is_file():
        files = [source]
    else:
        raise FileNotFoundError(f"path does not exist: {source}")

    clusters = Clusters()
    for file_path in files:
        try:
            formula = clusters.load_json(file_path, keep_lower=True)
        except TypeError:
            raise
        except Exception as e:
            raise ValueError(f"cannot read {file_path}: {e}") from e

        index = clusters.symbols.index(formula)
        cluster = clusters[index]
        cluster.info.setdefault("system", file_path.stem)
        cluster.info.setdefault("is_ref", 0)

    if len(clusters) == 0:
        raise ValueError(f"no cluster structures found in {source}")
    return clusters


def load_ref_energies(source: str | Path) -> dict[int, float]:
    """Load reference energies keyed by atom count.

    Parameters
    ----------
    source : str | pathlib.Path
        ASE JSON directory or CSV file. A CSV file must contain a
        ``natoms``/``N`` column and an ``energy`` or ``energy_per_atom``
        column.

    Returns
    -------
    dict[int, float]
        Lowest reference energy for each atom count.

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    ValueError
        If a CSV file lacks the required columns.
    """
    path = Path(source)
    if path.is_dir():
        clusters = load_clusters(path)
        energies: dict[int, float] = {}
        for cluster in clusters:
            energy = cluster.info.get("energy")
            if energy is None:
                continue
            count = len(cluster)
            value = float(energy)
            if count not in energies or value < energies[count]:
                energies[count] = value
        return energies

    if not path.is_file():
        raise FileNotFoundError(f"path does not exist: {path}")

    energies = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        count_column = next(
            (name for name in ("natoms", "N", "n") if name in fields),
            None,
        )
        energy_column = next(
            (name for name in ("energy", "E") if name in fields),
            None,
        )
        per_atom_column = "energy_per_atom" if "energy_per_atom" in fields else None
        if count_column is None or (
            energy_column is None and per_atom_column is None
        ):
            raise ValueError(
                "reference CSV needs a natoms/N column and an energy or "
                "energy_per_atom column"
            )
        for row in reader:
            try:
                count = int(float(row[count_column]))
                if energy_column is not None:
                    value = float(row[energy_column])
                else:
                    value = float(row[per_atom_column]) * count
            except (TypeError, ValueError):
                continue
            if count not in energies or value < energies[count]:
                energies[count] = value
    return energies


# ===== xyz2vasp IO: xyz reading, merging and VASP export =====

def _normalize_frame(frame: Atoms) -> None:
    """Copy calculator results into info/arrays and clear the calculator."""
    calc = frame.calc
    if calc is None:
        return
    results = getattr(calc, 'results', {})
    energy = results.get('energy')
    if energy is not None and frame.info.get('energy') is None:
        frame.info['energy'] = float(energy)
    forces = results.get('forces')
    if forces is not None and frame.arrays.get('forces') is None:
        frame.arrays['forces'] = forces
    frame.calc = None


def merge_multi_atoms(
    items: Sequence[MultiAtoms],
    s: float = 0.99,
    s_c1: float = 0.98,
    large_unique: bool = LARGE_CLUSTER_UNIQUE,
    large_atom_count: int = LARGE_CLUSTER_ATOM_COUNT,
) -> MultiAtoms:
    """Merge MultiAtoms containers and remove duplicates.

    Parameters
    ----------
    items : Sequence[MultiAtoms]
        Containers whose clusters should be merged.
    s : float, optional
        Similarity threshold for non-C1 point groups. Default is 0.99.
    s_c1 : float, optional
        Similarity threshold for C1 point groups. Default is 0.98.
    large_unique : bool, optional
        If True, clusters with at least ``large_atom_count`` atoms are made
        unique per chemical formula, keeping the lowest-energy one.
        Default is ``LARGE_CLUSTER_UNIQUE``.
    large_atom_count : int, optional
        Atom count threshold for large-cluster handling. Default is
        ``LARGE_CLUSTER_ATOM_COUNT``.

    Returns
    -------
    MultiAtoms
        Merged container after deduplication.
    """
    merged = MultiAtoms()
    for item in items:
        for cluster in item:
            merged.add_cluster(cluster)
    merged.deduplicate(s=s, s_c1=s_c1)
    if not large_unique:
        return merged

    unique = Clusters()
    remaining = MultiAtoms()
    for cluster in merged:
        if len(cluster) < large_atom_count:
            remaining.add_cluster(cluster)
            continue
        index = unique.add_cluster(cluster, sort=False)
        stored = unique[index]
        energy_stored = stored.info.get('energy')
        energy_cluster = cluster.info.get('energy')
        if energy_cluster is not None and (
            energy_stored is None
            or float(energy_cluster) < float(energy_stored)
        ):
            unique.add_cluster(cluster, replace=True, sort=False)
    for cluster in unique:
        remaining.add_cluster(cluster)
    return remaining


def merge_xyz_directories(
    xyz_dirs: Sequence[str | Path],
    n: int = 3,
    s: float = 0.99,
    s_c1: float = 0.98,
    large_unique: bool = LARGE_CLUSTER_UNIQUE,
    large_atom_count: int = LARGE_CLUSTER_ATOM_COUNT,
) -> MultiAtoms:
    """Read xyz directories, deduplicate them, and return a MultiAtoms.

    Each directory (or single xyz file) contributes the last ``n`` frames per
    xyz file, because sansde writes candidates worst-first (best last). The
    frames from one directory are deduplicated before all directories are
    merged and deduplicated again.

    Parameters
    ----------
    xyz_dirs : Sequence[str | Path]
        Directories containing xyz files, or single xyz files.
    n : int, optional
        Maximum number of frames read from each file. Default is 3.
    s : float, optional
        Similarity threshold for non-C1 point groups. Default is 0.99.
    s_c1 : float, optional
        Similarity threshold for C1 point groups. Default is 0.98.
    large_unique : bool, optional
        If True, clusters with at least ``large_atom_count`` atoms are made
        unique per chemical formula, keeping the lowest-energy one.
        Default is ``LARGE_CLUSTER_UNIQUE``.
    large_atom_count : int, optional
        Atom count threshold for large-cluster handling. Default is
        ``LARGE_CLUSTER_ATOM_COUNT``.

    Returns
    -------
    MultiAtoms
        Final merged and deduplicated container.

    Raises
    ------
    ValueError
        If ``n`` is not positive.
    FileNotFoundError
        If an input path does not exist.
    """
    if n <= 0:
        raise ValueError("n must be positive")

    containers: list[MultiAtoms] = []
    for xyz_dir in xyz_dirs:
        path = Path(xyz_dir)
        if not path.exists():
            raise FileNotFoundError(f"xyz path does not exist: {path}")

        if path.is_file():
            files = [path]
        else:
            files = sorted([*path.glob("*.xyz"), *path.glob("*.extxyz")])

        container = MultiAtoms()
        for file_path in files:
            # Candidate frames are written worst-first, so the last n frames
            # hold the best structures.
            for frame in deque(iread(file_path), maxlen=n):
                _normalize_frame(frame)
                container.add_cluster(frame)
        container.deduplicate(s=s, s_c1=s_c1)
        containers.append(container)

    return merge_multi_atoms(
        containers,
        s=s,
        s_c1=s_c1,
        large_unique=large_unique,
        large_atom_count=large_atom_count,
    )


def _write_poscar(
    cluster: Atoms,
    path: Path,
    cell: Sequence[float] | None,
    large_cell: Sequence[float],
    default_cell: Sequence[float],
    large_atom_count: int,
) -> None:
    """Write a cluster as a POSCAR.

    ``cell``, ``large_cell`` and ``default_cell`` are six-element sequences
    ``(a, b, c, alpha, beta, gamma)`` giving cell vector lengths in Angstrom
    followed by the angles between them in degrees. Clusters with at least
    ``large_atom_count`` atoms always use ``large_cell``. Smaller clusters use
    ``cell`` when given, otherwise they keep their own cell; if a small
    cluster has no cell (zero volume), ``default_cell`` is used. Atoms are
    sorted by chemical symbol so that each species forms a single contiguous
    block, as required for a matching concatenated POTCAR.
    """
    atoms = cluster.copy()
    if len(atoms) >= large_atom_count:
        atoms.cell = list(large_cell)
    elif cell is not None:
        atoms.cell = list(cell)
    elif atoms.cell.volume == 0.0:
        atoms.cell = list(default_cell)
    atoms.pbc = True
    write(path, atoms, format="vasp", direct=False, sort=True)


def _copy_template_files(src_dir: Path, dst_dir: Path) -> None:
    """Copy all regular files from a template directory into a task directory."""
    for src_file in src_dir.iterdir():
        if src_file.is_file():
            shutil.copy2(src_file, dst_dir / src_file.name)


def _patch_job_script(script: Path, job_name: str, cores: int) -> None:
    """Set the '#BSUB -J' job name and '#BSUB -n' core count in a job script."""
    lines: list[str] = script.read_text().splitlines(keepends=True)
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#BSUB -J"):
            lines[index] = f"#BSUB -J {job_name}\n"
        elif stripped.startswith("#BSUB -n"):
            lines[index] = f"#BSUB -n {cores}\n"
    script.write_text("".join(lines))


def _generate_potcar(task_dir: Path, timeout: float = 120.0) -> bool:
    """Generate POTCAR in a task directory with 'vaspkit -task 103'."""
    try:
        result = subprocess.run(
            ["vaspkit", "-task", "103"],
            cwd=task_dir,
            check=False,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=timeout,
        )
    except FileNotFoundError:
        print("Warning: vaspkit not found; POTCAR was not generated")
        return False
    except subprocess.TimeoutExpired:
        print(f"Warning: 'vaspkit -task 103' timed out in {task_dir}")
        return False

    potcar = task_dir / "POTCAR"
    if result.returncode != 0 or not potcar.is_file() or potcar.stat().st_size == 0:
        print(f"Warning: failed to generate POTCAR in {task_dir}")
        output = (result.stderr or result.stdout).strip().splitlines()
        message = next(
            (line.strip() for line in output if "error" in line.lower()),
            output[-1].strip() if output else "",
        )
        if message:
            print(f"  vaspkit: {message}")
        return False
    return True


def write_vasp_tasks(params: str | Path) -> list[Path]:
    """Run the xyz2vasp pipeline from a JSON parameter file.

    The JSON file drives the whole pipeline: the xyz input is read into a
    :class:`~sansde.pop.MultiAtoms` and deduplicated, ``merged.xyz``,
    ``merged_xyz/`` and ``merged_json/`` are written to the current directory,
    and one VASP task directory is created per cluster.

    Parameters
    ----------
    params : str | Path
        JSON parameter file.

    Returns
    -------
    list[Path]
        Paths of the created VASP task directories. When ``compress`` is true
        these directories no longer exist because they have been archived.

    Raises
    ------
    FileNotFoundError
        If the template directory does not exist.
    TypeError
        If the JSON content is not an object.

    Examples
    --------
    >>> write_vasp_tasks("params.json")

    ``params.json`` with default values:

    .. code-block:: json

        {
            "input": "best",
            "template": "input",
            "frame": 3,
            "similarity": [0.99, 0.98],
            "cell": null,
            "large_cluster_cell": [20.0, 20.0, 20.0, 90.0, 90.0, 90.0],
            "large_cluster_atom_count": 40,
            "large_cluster_unique": true,
            "output": null,
            "compress": false
        }

    ``frame`` is the number of frames read from the end of each xyz file; the
    best candidates are stored last. ``cell`` is six values
    ``(a, b, c, alpha, beta, gamma)`` and overrides the cell of clusters
    smaller than ``large_cluster_atom_count``.
    ``large_cluster_cell`` sets the cell used for clusters with at least
    ``large_cluster_atom_count`` atoms (it takes precedence over ``cell``),
    and ``large_cluster_unique`` makes those clusters unique per formula.
    ``output`` defaults to ``relax.<elements>``.
    """
    settings: dict[str, Any] = json.loads(Path(params).read_text())
    if not isinstance(settings, dict):
        raise TypeError(f"JSON parameters must be an object: {params}")

    input_value = settings.get("input", "best")
    if isinstance(input_value, (str, Path)):
        input_value = [input_value]
    input_dirs = [Path(item) for item in input_value]

    frame = int(settings.get("frame", 3))
    similarity = settings.get("similarity", (0.99, 0.98))
    s, s_c1 = (float(value) for value in similarity)
    cell_value = settings.get("cell")
    cell = (
        [float(value) for value in cell_value]
        if cell_value is not None
        else None
    )
    compress = bool(settings.get("compress", False))
    template_path = Path(str(settings.get("template", "input")))
    large_cell_value = settings.get("large_cluster_cell")
    large_cell = (
        [float(value) for value in large_cell_value]
        if large_cell_value is not None
        else list(LARGE_CLUSTER_CELL)
    )
    large_atom_count = int(
        settings.get("large_cluster_atom_count", LARGE_CLUSTER_ATOM_COUNT)
    )
    large_unique = bool(
        settings.get("large_cluster_unique", LARGE_CLUSTER_UNIQUE)
    )

    clusters = merge_xyz_directories(
        input_dirs,
        n=frame,
        s=s,
        s_c1=s_c1,
        large_unique=large_unique,
        large_atom_count=large_atom_count,
    )
    print(f"loaded {len(clusters)} clusters from {len(input_dirs)} input path(s)")

    work_dir = Path.cwd()
    if clusters:
        write(work_dir / "merged.xyz", list(clusters), format="extxyz")
    xyz_dir = work_dir / "merged_xyz"
    json_dir = work_dir / "merged_json"
    xyz_dir.mkdir(exist_ok=True)
    json_dir.mkdir(exist_ok=True)
    counts: dict[str, int] = {}
    for cluster in clusters:
        formula = cluster.get_chemical_formula()
        index = counts.get(formula, 0)
        counts[formula] = index + 1
        write(xyz_dir / f"{formula}.{index}.xyz", cluster, format="extxyz")
        record = {
            "symbols": cluster.get_chemical_symbols(),
            "positions": cluster.get_positions().tolist(),
            "cell": cluster.cell.array.tolist(),
            "pbc": cluster.pbc.tolist(),
            "info": {
                key: value.tolist() if hasattr(value, "tolist") else value
                for key, value in cluster.info.items()
            },
        }
        (json_dir / f"{formula}.{index}.json").write_text(
            json.dumps(record, indent=2)
        )
    print(
        f"wrote merged.xyz, merged_xyz/ and merged_json/ "
        f"({len(clusters)} frames)"
    )

    elements = sorted(
        {symbol for cluster in clusters for symbol in cluster.get_chemical_symbols()}
    )
    output_value = settings.get("output")
    output_path = (
        Path(str(output_value))
        if output_value is not None
        else Path(f"relax.{''.join(elements)}")
    )

    if not template_path.is_dir():
        raise FileNotFoundError(f"VASP template directory not found: {template_path}")

    for name in ("INCAR", "KPOINTS"):
        if not (template_path / name).is_file():
            print(f"Warning: {name} not found in template directory {template_path}")

    has_potcar = (template_path / "POTCAR").is_file()
    if not has_potcar:
        print(
            "Warning: POTCAR not found in template directory "
            f"{template_path}; trying 'vaspkit -task 103' in each task directory"
        )

    output_path.mkdir(parents=True, exist_ok=True)

    counts = {}
    task_dirs: list[Path] = []
    for cluster in clusters:
        formula = cluster.get_chemical_formula()
        index = counts.get(formula, 0)
        task_dir = output_path / f"{formula}.{index}"
        while task_dir.exists():
            index += 1
            task_dir = output_path / f"{formula}.{index}"
        counts[formula] = index + 1
        task_dir.mkdir()

        _write_poscar(
            cluster,
            task_dir / "POSCAR",
            cell,
            large_cell,
            DEFAULT_CELL,
            large_atom_count,
        )
        _copy_template_files(template_path, task_dir)
        cores = 12 if len(cluster) <= 40 else 20
        for script in task_dir.glob("*.lsf"):
            _patch_job_script(script, formula, cores)
        if not has_potcar:
            _generate_potcar(task_dir)
        task_dirs.append(task_dir)

    if compress and task_dirs:
        archive_path = output_path.parent / f"{output_path.name}.tgz"
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(output_path, arcname=output_path.name)
        shutil.rmtree(output_path)
        print(f"Packed {output_path} into {archive_path} and removed {output_path}")

    print(f"wrote {len(task_dirs)} VASP tasks to {output_path}")
    return task_dirs
