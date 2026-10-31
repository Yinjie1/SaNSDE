"""Utilities for reading xyz directories into MultiAtoms containers."""

from collections.abc import Sequence
from pathlib import Path

from ase import Atoms
from ase.io import iread

from sansde.pop import MultiAtoms

__all__ = [
    "merge_multi_atoms",
    "merge_xyz_directories",
    "read_xyz_frames",
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
