"""Utilities for reading xyz directories and writing VASP task directories."""

import shutil
import subprocess
from collections.abc import Iterable, Sequence
from pathlib import Path

from ase import Atoms
from ase.io import iread, write

from sansde.pop import MultiAtoms

__all__ = [
    "merge_multi_atoms",
    "merge_xyz_directories",
    "read_xyz_frames",
    "write_vasp_tasks",
]


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


def read_xyz_frames(
    xyz_dir: str | Path,
    n: int = 3,
) -> list[Atoms]:
    """Read up to ``n`` frames from the xyz files in a directory.

    Files are processed in filename order, and frames are counted across all
    ``.xyz`` and ``.extxyz`` files in the directory.

    Parameters
    ----------
    xyz_dir : str | Path
        Directory containing xyz files, or a single xyz file.
    n : int, optional
        Maximum number of frames to read. Default is 3.

    Returns
    -------
    list[Atoms]
        Frames read from the directory, in file order.

    Raises
    ------
    ValueError
        If ``n`` is not positive.
    FileNotFoundError
        If ``xyz_dir`` does not exist.
    """
    if n <= 0:
        raise ValueError("n must be positive")

    path = Path(xyz_dir)
    if not path.exists():
        raise FileNotFoundError(f"xyz path does not exist: {path}")

    if path.is_file():
        files = [path]
    else:
        files = sorted([*path.glob("*.xyz"), *path.glob("*.extxyz")])

    frames: list[Atoms] = []
    for file_path in files:
        for frame in iread(file_path):
            _normalize_frame(frame)
            frames.append(frame)
            if len(frames) >= n:
                return frames
    return frames


def merge_multi_atoms(
    items: Sequence[MultiAtoms],
    s: float = 0.99,
    s_c1: float = 0.98,
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
    return merged


def merge_xyz_directories(
    xyz_dirs: Sequence[str | Path],
    n: int = 3,
    s: float = 0.99,
    s_c1: float = 0.98,
) -> MultiAtoms:
    """Build and merge MultiAtoms from multiple xyz directories.

    Each directory contributes up to ``n`` frames. The frames from one
    directory are deduplicated before all directories are merged and
    deduplicated again.

    Parameters
    ----------
    xyz_dirs : Sequence[str | Path]
        Directories containing xyz files.
    n : int, optional
        Maximum number of frames read from each directory. Default is 3.
    s : float, optional
        Similarity threshold for non-C1 point groups. Default is 0.99.
    s_c1 : float, optional
        Similarity threshold for C1 point groups. Default is 0.98.

    Returns
    -------
    MultiAtoms
        Final merged and deduplicated container.
    """
    containers: list[MultiAtoms] = []
    for xyz_dir in xyz_dirs:
        container = MultiAtoms()
        for frame in read_xyz_frames(xyz_dir, n=n):
            container.add_cluster(frame)
        container.deduplicate(s=s, s_c1=s_c1)
        containers.append(container)

    return merge_multi_atoms(containers, s=s, s_c1=s_c1)


def _write_poscar(
    cluster: Atoms,
    path: Path,
    a: float,
    b: float,
    c: float,
) -> None:
    """Write a cluster as a POSCAR inside an orthorhombic cell.

    Atoms are sorted by chemical symbol so that each species forms a single
    contiguous block, as required for a matching concatenated POTCAR.
    """
    atoms = cluster.copy()
    atoms.cell = [[a, 0.0, 0.0], [0.0, b, 0.0], [0.0, 0.0, c]]
    atoms.pbc = True
    write(path, atoms, format="vasp", direct=False, sort=True)


def _copy_template_files(src_dir: Path, dst_dir: Path) -> None:
    """Copy all regular files from a template directory into a task directory."""
    for src_file in src_dir.iterdir():
        if src_file.is_file():
            shutil.copy2(src_file, dst_dir / src_file.name)


def _patch_job_name(script: Path, job_name: str) -> None:
    """Replace the '#BSUB -J' job name line in a job script."""
    lines: list[str] = script.read_text().splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.strip().startswith("#BSUB -J"):
            lines[index] = f"#BSUB -J {job_name}\n"
            break
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
    a: float = 15.0,
    b: float = 15.0,
    c: float = 15.0,
) -> list[Path]:
    """Write VASP task directories for the given clusters.

    Every cluster is written as a Cartesian POSCAR inside an orthorhombic
    cell ``(a, b, c)`` in a subdirectory named ``<formula>.<index>``, for
    example ``Au3.0`` and ``Au3.1``. All files from ``template_dir`` are
    copied into each task directory, and ``.lsf`` scripts get their
    ``#BSUB -J`` line renamed to the chemical formula.

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
    a : float, optional
        Cell length along x in Angstrom. Default is 15.0.
    b : float, optional
        Cell length along y in Angstrom. Default is 15.0.
    c : float, optional
        Cell length along z in Angstrom. Default is 15.0.

    Returns
    -------
    list[Path]
        Paths of the created task directories.

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
    ...     a=15.0,
    ...     b=15.0,
    ...     c=15.0,
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

        _write_poscar(cluster, task_dir / "POSCAR", a, b, c)
        _copy_template_files(template_path, task_dir)
        for script in task_dir.glob("*.lsf"):
            _patch_job_name(script, formula)
        if not has_potcar:
            _generate_potcar(task_dir)
        task_dirs.append(task_dir)

    return task_dirs
