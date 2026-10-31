"""Utilities for reading xyz directories and writing VASP task directories."""

import shutil
import subprocess
import tarfile
from collections.abc import Iterable, Sequence
from pathlib import Path

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


def write_vasp_tasks(
    clusters: Iterable[Atoms],
    output_dir: str | Path,
    template_dir: str | Path,
    cell: Sequence[float] | None = None,
    large_cell: Sequence[float] = LARGE_CLUSTER_CELL,
    default_cell: Sequence[float] = DEFAULT_CELL,
    pack: bool = True,
) -> list[Path]:
    """Write VASP task directories for the given clusters.

    Every cluster is written as a Cartesian POSCAR in a subdirectory named
    ``<formula>.<index>``, for example ``Au3.0`` and ``Au3.1``. Cell
    parameters are six-element sequences ``(a, b, c, alpha, beta, gamma)``
    with lengths in Angstrom and angles in degrees. When ``cell`` is given it
    overrides the cluster cell. Otherwise clusters with at least
    ``LARGE_CLUSTER_ATOM_COUNT`` atoms use ``large_cell``, while smaller
    clusters keep their own cell; if a small cluster has no cell (zero
    volume), ``default_cell`` is used. All files from ``template_dir`` are
    copied into each task directory. In copied ``.lsf`` scripts, the
    ``#BSUB -J`` line is renamed to the chemical formula and the ``#BSUB -n``
    line is set to 12 for clusters with at most
    ``LARGE_CLUSTER_ATOM_COUNT`` atoms and to 20 for larger clusters.

    When ``pack`` is true, the task directories are packed into
    ``<output_dir>.tgz`` and the original directory is removed. The input
    xyz files are never touched.

    A warning is printed when INCAR or KPOINTS is missing from
    ``template_dir``. When POTCAR is missing, ``vaspkit -task 103`` is
    executed in each task directory to generate it.

    Parameters
    ----------
    clusters : Iterable[Atoms]
        Clusters to convert. A :class:`~sansde.pop.MultiAtoms` instance can
        be passed directly.
    output_dir : str | Path
        Directory in which the task subdirectories are created.
    template_dir : str | Path
        Directory containing VASP input template files.
    cell : Sequence[float], optional
        Six cell parameters ``(a, b, c, alpha, beta, gamma)`` that override
        the cluster cell. Default is ``None`` (keep the cluster cell).
    large_cell : Sequence[float], optional
        Six cell parameters for clusters with at least
        ``LARGE_CLUSTER_ATOM_COUNT`` atoms. Default is
        ``(20, 20, 20, 90, 90, 90)``.
    default_cell : Sequence[float], optional
        Six cell parameters for clusters without a cell. Default is
        ``(15, 15, 15, 90, 90, 90)``.
    pack : bool, optional
        If ``True`` (default), pack the task directories into
        ``<output_dir>.tgz`` and remove ``output_dir`` afterwards.

    Returns
    -------
    list[Path]
        Paths of the created task directories. When ``pack`` is true these
        directories no longer exist because they have been archived.

    Raises
    ------
    FileNotFoundError
        If ``template_dir`` does not exist.

    Examples
    --------
    >>> tasks = write_vasp_tasks(
    ...     clusters,
    ...     "tasks",
    ...     "vasp_input",
    ...     cell=(15.0, 15.0, 15.0, 90.0, 90.0, 90.0),
    ... )
    """
    template_path = Path(template_dir)
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

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
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
            default_cell,
        )
        _copy_template_files(template_path, task_dir)
        cores = 12 if len(cluster) <= 40 else 20
        for script in task_dir.glob("*.lsf"):
            _patch_job_script(script, formula, cores)
        if not has_potcar:
            _generate_potcar(task_dir)
        task_dirs.append(task_dir)

    if pack and task_dirs:
        archive_path = output_path.parent / f"{output_path.name}.tgz"
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(output_path, arcname=output_path.name)
        shutil.rmtree(output_path)
        print(f"Packed {output_path} into {archive_path} and removed {output_path}")

    return task_dirs
