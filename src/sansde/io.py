"""Utilities for reading xyz directories and writing VASP task directories."""

import json
import shutil
import subprocess
import tarfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.io import iread, write

from sansde.pop import Clusters, MultiAtoms

__all__ = [
    "merge_multi_atoms",
    "merge_xyz_directories",
    "write_vasp_tasks",
]

LARGE_CLUSTER_ATOM_COUNT: int = 40
LARGE_CLUSTER_CELL: tuple[float, ...] = (20.0, 20.0, 20.0, 90.0, 90.0, 90.0)
DEFAULT_CELL: tuple[float, ...] = (15.0, 15.0, 15.0, 90.0, 90.0, 90.0)
LARGE_CLUSTER_UNIQUE: bool = True


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
        If True, clusters with at least ``LARGE_CLUSTER_ATOM_COUNT`` atoms are
        made unique per chemical formula, keeping the lowest-energy one.
        Default is ``LARGE_CLUSTER_UNIQUE``.

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
        if len(cluster) < LARGE_CLUSTER_ATOM_COUNT:
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
) -> MultiAtoms:
    """Read xyz directories, deduplicate them, and return a MultiAtoms.

    Each directory (or single xyz file) contributes up to ``n`` frames per
    xyz file. The frames from one directory are deduplicated before all
    directories are merged and deduplicated again.

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
        If True, clusters with at least ``LARGE_CLUSTER_ATOM_COUNT`` atoms are
        made unique per chemical formula, keeping the lowest-energy one.
        Default is ``LARGE_CLUSTER_UNIQUE``.

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
            for count, frame in enumerate(iread(file_path), start=1):
                _normalize_frame(frame)
                container.add_cluster(frame)
                if count >= n:
                    break
        container.deduplicate(s=s, s_c1=s_c1)
        containers.append(container)

    return merge_multi_atoms(
        containers,
        s=s,
        s_c1=s_c1,
        large_unique=large_unique,
    )


def _write_poscar(
    cluster: Atoms,
    path: Path,
    cell: Sequence[float] | None,
    large_cell: Sequence[float],
    default_cell: Sequence[float],
) -> None:
    """Write a cluster as a POSCAR.

    ``cell``, ``large_cell`` and ``default_cell`` are six-element sequences
    ``(a, b, c, alpha, beta, gamma)`` giving cell vector lengths in Angstrom
    followed by the angles between them in degrees. When ``cell`` is given it
    overrides the cluster cell. Otherwise clusters with at least
    ``LARGE_CLUSTER_ATOM_COUNT`` atoms use ``large_cell``, while smaller
    clusters keep their own cell; if a small cluster has no cell (zero
    volume), ``default_cell`` is used. Atoms are sorted by chemical symbol so
    that each species forms a single contiguous block, as required for a
    matching concatenated POTCAR.
    """
    atoms = cluster.copy()
    if cell is not None:
        atoms.cell = list(cell)
    elif len(atoms) >= LARGE_CLUSTER_ATOM_COUNT:
        atoms.cell = list(large_cell)
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
            "output": null,
            "compress": false
        }

    ``cell`` is six values ``(a, b, c, alpha, beta, gamma)`` and overrides the
    cluster cell when given. ``output`` defaults to ``relax.<elements>``.
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

    clusters = merge_xyz_directories(
        input_dirs,
        n=frame,
        s=s,
        s_c1=s_c1,
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
            LARGE_CLUSTER_CELL,
            DEFAULT_CELL,
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
