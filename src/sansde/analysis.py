"""Cluster energy, shape, and similarity analysis from ASE JSON files."""

import csv
import io
import re
import sys
from collections.abc import Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import TypeAlias

import numpy as np
from ase import Atoms
from ase.io.jsonio import read_json

from sansde.operators import compute_point_group
from sansde.pop import Clusters

SHAPE_RATIO = 0.8
UL = 1.0

AnalysisRow: TypeAlias = dict[str, object]

__all__ = [
    "AnalysisRow",
    "analyze_clusters",
    "angle_similarity_add_atom",
    "eval_fit",
    "fit_energy",
    "format_cluster_csv",
    "load_clusters",
    "load_ref_energies",
    "shape_indicator",
    "shape_label",
    "similarity_add_atom",
    "write_cluster_csv",
]

BASE_FIELDS = [
    "system",
    "symbol",
    "natoms",
    "is_ref",
    "energy",
    "energy_per_atom",
    "energy_minus_fit",
    "second_difference",
    "PG",
    "lambda1",
    "lambda2",
    "lambda3",
    "shape",
    "S",
]

FLOAT8_FIELDS = {
    "energy",
    "energy_per_atom",
    "energy_minus_fit",
    "second_difference",
    "S",
    "S2",
    "binding_energy_per_atom",
    "energy_diff_ref_per_atom",
}

FLOAT6_FIELDS = {"lambda1", "lambda2", "lambda3"}


def natural_key(value: str) -> list[object]:
    """Split a string into integer and lowercase text chunks for sorting.

    Parameters
    ----------
    value : str
        String to split.

    Returns
    -------
    list[object]
        Chunks used as a natural-order sort key.
    """
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
        Clusters ordered by atom count.

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    TypeError
        If a file does not contain an ASE ``Atoms`` object.
    ValueError
        If no structure is loaded.
    """
    source = Path(directory)
    if source.is_dir():
        files = sorted(source.glob("*.json"), key=lambda path: natural_key(path.stem))
    elif source.is_file():
        files = [source]
    else:
        raise FileNotFoundError(f"path does not exist: {source}")

    clusters = Clusters()
    for file_path in files:
        try:
            atoms = read_json(file_path)
        except Exception as e:
            raise ValueError(f"cannot read {file_path}: {e}") from e
        if not isinstance(atoms, Atoms):
            raise TypeError(f"{file_path} does not contain an ASE Atoms object")

        atoms.info.setdefault("system", file_path.stem)
        atoms.info.setdefault("is_ref", 0)
        formula = atoms.get_chemical_formula()
        if formula not in clusters.symbols:
            clusters.add_cluster(atoms)
            continue

        index = clusters.symbols.index(formula)
        current = clusters[index]
        new_energy = atoms.info.get("energy")
        old_energy = current.info.get("energy")
        if new_energy is not None and (
            old_energy is None or float(new_energy) < float(old_energy)
        ):
            clusters.add_cluster(atoms, replace=True)

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


def fit_energy(
    natoms: Sequence[int],
    energies: Sequence[float | None],
) -> np.ndarray | None:
    """Fit ``E = a*N + b*N^(2/3) + c*N^(1/3) + d`` by least squares.

    Parameters
    ----------
    natoms : Sequence[int]
        Atom counts.
    energies : Sequence[float | None]
        Total energies. Entries with ``None`` are ignored.

    Returns
    -------
    numpy.ndarray | None
        Coefficients in ``[a, b, c, d]`` order, or ``None`` when fewer than
        four valid points are available or the fit is rank deficient.
    """
    rows = [
        (float(count), float(energy))
        for count, energy in zip(natoms, energies)
        if count and energy is not None
    ]
    if len(rows) < 4:
        return None
    counts = np.asarray([row[0] for row in rows], dtype=float)
    values = np.asarray([row[1] for row in rows], dtype=float)
    basis = np.column_stack(
        [
            counts,
            counts ** (2.0 / 3.0),
            counts ** (1.0 / 3.0),
            np.ones_like(counts),
        ]
    )
    coefficients, _residuals, rank, _singular = np.linalg.lstsq(
        basis,
        values,
        rcond=None,
    )
    if rank < 4:
        return None
    return coefficients


def eval_fit(coefficients: np.ndarray, natoms: int) -> float:
    """Evaluate fitted energy coefficients at one atom count.

    Parameters
    ----------
    coefficients : numpy.ndarray
        Coefficients in ``[a, b, c, d]`` order.
    natoms : int
        Atom count.

    Returns
    -------
    float
        Fitted total energy.
    """
    a, b, c, d = coefficients
    count = float(natoms)
    return float(
        a * count + b * count ** (2.0 / 3.0) + c * count ** (1.0 / 3.0) + d
    )


def pairwise_sorted(coords: np.ndarray) -> np.ndarray:
    """Return sorted pairwise distances.

    Parameters
    ----------
    coords : numpy.ndarray
        Atomic coordinates with shape ``(N, 3)``.

    Returns
    -------
    numpy.ndarray
        ``N(N-1)/2`` distances in ascending order.
    """
    count = len(coords)
    if count < 2:
        return np.array([])
    diff = coords[:, None, :] - coords[None, :, :]
    distances = np.sqrt((diff**2).sum(axis=-1))
    upper = np.triu_indices(count, k=1)
    return np.sort(distances[upper])


def shape_indicator(coords: np.ndarray | None) -> np.ndarray | None:
    """Compute normalized second-moment eigenvalues.

    Parameters
    ----------
    coords : numpy.ndarray | None
        Atomic coordinates with shape ``(N, 3)``.

    Returns
    -------
    numpy.ndarray | None
        Eigenvalues ``lambda_i / N^(2/3)`` in descending order, or ``None``
        when fewer than two atoms are available.
    """
    if coords is None or len(coords) < 2:
        return None
    count = len(coords)
    centered = np.asarray(coords, dtype=float) - np.asarray(coords, dtype=float).mean(
        axis=0
    )
    matrix = (centered.T @ centered) / (count * UL**2)
    eigenvalues = np.linalg.eigvalsh(matrix)[::-1]
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    return eigenvalues / (count ** (2.0 / 3.0))


def shape_label(
    eigenvalues: np.ndarray | None,
    ratio: float = SHAPE_RATIO,
) -> str | None:
    """Classify a cluster shape from normalized eigenvalues.

    Parameters
    ----------
    eigenvalues : numpy.ndarray | None
        Normalized eigenvalues in descending order.
    ratio : float, optional
        Threshold for the second and third eigenvalues relative to the first.
        Default is ``SHAPE_RATIO``.

    Returns
    -------
    str | None
        ``"球形"``, ``"饼形"``, or ``"直线形"``; ``None`` when eigenvalues are
        unavailable or the first eigenvalue is not positive.
    """
    if eigenvalues is None:
        return None
    first = float(eigenvalues[0])
    if first <= 0:
        return None
    ratio_12 = float(eigenvalues[1]) / first
    ratio_13 = float(eigenvalues[2]) / first
    if ratio_12 >= ratio and ratio_13 >= ratio:
        return "球形"
    if ratio_12 >= ratio and ratio_13 < ratio:
        return "饼形"
    return "直线形"


def similarity_add_atom(
    coords_n: np.ndarray | None,
    coords_ref: np.ndarray | None,
) -> float | None:
    """Compute the distance similarity ``S(N, N-1)``.

    Each atom is removed from the N-atom cluster in turn, and the minimum
    distance-based ``q`` against the reference cluster defines
    ``S = 1 / (1 + q / UL)``.

    Parameters
    ----------
    coords_n : numpy.ndarray | None
        Coordinates of the N-atom cluster.
    coords_ref : numpy.ndarray | None
        Coordinates of the reference cluster with ``N - 1`` atoms.

    Returns
    -------
    float | None
        Similarity value, or ``None`` when the input sizes are inconsistent.
    """
    if coords_n is None or coords_ref is None:
        return None
    count = len(coords_n)
    reference_count = len(coords_ref)
    if reference_count != count - 1 or count < 3:
        return None
    if reference_count < 2:
        return None
    reference_distances = pairwise_sorted(coords_ref)
    if reference_distances[0] <= 0:
        return None

    minimum_q: float | None = None
    for index in range(count):
        trial = pairwise_sorted(np.delete(coords_n, index, axis=0))
        if len(trial) != len(reference_distances) or trial[0] <= 0:
            return None
        q = float(
            np.sqrt(
                (2.0 / (reference_count * (reference_count - 1)))
                * np.sum(
                    (reference_distances / reference_distances[0] - trial / trial[0])
                    ** 2
                )
            )
        )
        minimum_q = q if minimum_q is None else min(minimum_q, q)
    if minimum_q is None:
        return None
    return 1.0 / (1.0 + minimum_q / UL)


def _cos_angle(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return row-wise cosine values clipped to ``[-1, 1]``."""
    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)
    denominator = left_norm * right_norm
    dot = (left * right).sum(axis=1)
    cosine = np.ones_like(dot)
    valid = denominator > 1e-12
    cosine[valid] = np.clip(dot[valid] / denominator[valid], -1.0, 1.0)
    return cosine


def _triple_cosines(
    coords: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return unordered triples and the cosine at each vertex."""
    count = len(coords)
    triples = np.asarray(list(combinations(range(count), 3)), dtype=int)
    i, j, k = triples[:, 0], triples[:, 1], triples[:, 2]
    pi, pj, pk = coords[i], coords[j], coords[k]
    cosine_i = _cos_angle(pj - pi, pk - pi)
    cosine_j = _cos_angle(pi - pj, pk - pj)
    cosine_k = _cos_angle(pi - pk, pj - pk)
    return i, j, k, cosine_i, cosine_j, cosine_k


def sorted_cosines(coords: np.ndarray | None) -> np.ndarray:
    """Return sorted vertex cosines for all unordered triples.

    Parameters
    ----------
    coords : numpy.ndarray | None
        Atomic coordinates with shape ``(N, 3)``.

    Returns
    -------
    numpy.ndarray
        ``N(N-1)(N-2)/2`` cosine values in ascending order.
    """
    if coords is None or len(coords) < 3:
        return np.array([])
    _i, _j, _k, cosine_i, cosine_j, cosine_k = _triple_cosines(coords)
    return np.sort(np.concatenate([cosine_i, cosine_j, cosine_k]))


def angle_similarity_add_atom(
    coords_n: np.ndarray | None,
    coords_ref: np.ndarray | None,
) -> float | None:
    """Compute the angle similarity ``S2(N, N-1)``.

    Parameters
    ----------
    coords_n : numpy.ndarray | None
        Coordinates of the N-atom cluster.
    coords_ref : numpy.ndarray | None
        Coordinates of the reference cluster with ``N - 1`` atoms.

    Returns
    -------
    float | None
        Similarity value, or ``None`` when fewer than four atoms are
        available or the input sizes are inconsistent.
    """
    if coords_n is None or coords_ref is None:
        return None
    count = len(coords_n)
    if len(coords_ref) != count - 1 or count < 4:
        return None
    reference_cosines = sorted_cosines(coords_ref)
    if reference_cosines.size == 0:
        return None

    i, j, k, cosine_i, cosine_j, cosine_k = _triple_cosines(coords_n)
    minimum_q2: float | None = None
    for removed in range(count):
        keep = (i != removed) & (j != removed) & (k != removed)
        trial = np.sort(
            np.concatenate([cosine_i[keep], cosine_j[keep], cosine_k[keep]])
        )
        if trial.size != reference_cosines.size:
            return None
        q2 = float(np.sqrt(np.mean((trial - reference_cosines) ** 2)))
        minimum_q2 = q2 if minimum_q2 is None else min(minimum_q2, q2)
    if minimum_q2 is None:
        return None
    return 1.0 / (1.0 + 2.0 * minimum_q2)


def analyze_clusters(
    clusters: Clusters,
    *,
    e1: float | None = None,
    ref: str | Path | Mapping[int, float] | None = None,
    compute_s2: bool = False,
    shape_ratio: float = SHAPE_RATIO,
    pg_tolerance: float = 0.3,
) -> list[AnalysisRow]:
    """Compute energy, point group, shape, and similarity quantities.

    Parameters
    ----------
    clusters : Clusters
        Clusters to analyze.
    e1 : float | None, optional
        Isolated-atom energy. When omitted, the energy of the N=1 cluster is
        used. Binding energy is omitted when neither source is available.
    ref : str | pathlib.Path | Mapping[int, float] | None, optional
        Reference energies keyed by atom count, or a JSON directory/CSV file
        accepted by :func:`load_ref_energies`.
    compute_s2 : bool, optional
        Whether to compute and output the angle similarity ``S2``. Default
        False.
    shape_ratio : float, optional
        Shape classification threshold. Default is ``SHAPE_RATIO``.
    pg_tolerance : float, optional
        Point-group tolerance passed to
        :func:`sansde.operators.compute_point_group`. Default 0.3.

    Returns
    -------
    list[dict[str, object]]
        One row per cluster, sorted by atom count and system name.
    """
    entries = sorted(
        clusters,
        key=lambda cluster: (
            len(cluster),
            str(cluster.info.get("system", cluster.get_chemical_formula())),
        ),
    )
    lowest_by_count: dict[int, Atoms] = {}
    for cluster in entries:
        energy = cluster.info.get("energy")
        if energy is None:
            continue
        count = len(cluster)
        current = lowest_by_count.get(count)
        if current is None or float(energy) < float(current.info["energy"]):
            lowest_by_count[count] = cluster

    coefficients = fit_energy(
        [len(cluster) for cluster in entries],
        [cluster.info.get("energy") for cluster in entries],
    )

    if ref is None:
        reference: dict[int, float] | None = None
    elif isinstance(ref, Mapping):
        reference = {int(key): float(value) for key, value in ref.items()}
    else:
        reference = load_ref_energies(ref)

    if e1 is None and 1 in lowest_by_count:
        energy_one = lowest_by_count[1].info.get("energy")
        e1_value = None if energy_one is None else float(energy_one)
    else:
        e1_value = None if e1 is None else float(e1)
    include_binding = e1_value is not None

    rows: list[AnalysisRow] = []
    for cluster in entries:
        count = len(cluster)
        formula = cluster.get_chemical_formula()
        system = str(cluster.info.get("system", formula))
        energy_raw = cluster.info.get("energy")
        energy = None if energy_raw is None else float(energy_raw)

        second_difference = None
        if energy is not None:
            previous = lowest_by_count.get(count - 1)
            following = lowest_by_count.get(count + 1)
            previous_energy = (
                None if previous is None else previous.info.get("energy")
            )
            following_energy = (
                None if following is None else following.info.get("energy")
            )
            if previous_energy is not None and following_energy is not None:
                second_difference = (
                    float(previous_energy)
                    + float(following_energy)
                    - 2.0 * energy
                )

        energy_minus_fit = None
        if coefficients is not None and energy is not None and count:
            energy_minus_fit = energy - eval_fit(coefficients, count)

        point_group = compute_point_group(cluster, tolerance=pg_tolerance)
        positions = cluster.get_positions()
        eigenvalues = shape_indicator(positions)
        shape = shape_label(eigenvalues, shape_ratio)

        previous_cluster = lowest_by_count.get(count - 1)
        previous_positions = (
            None if previous_cluster is None else previous_cluster.get_positions()
        )
        similarity = similarity_add_atom(positions, previous_positions)
        similarity_angle = None
        if compute_s2:
            similarity_angle = angle_similarity_add_atom(
                positions,
                previous_positions,
            )

        binding_energy = None
        if include_binding and energy is not None and count:
            binding_energy = e1_value - energy / count

        reference_difference = None
        if reference is not None and energy is not None and count in reference:
            reference_difference = (energy - reference[count]) / count

        row: AnalysisRow = {
            "system": system,
            "symbol": formula,
            "natoms": count,
            "is_ref": int(cluster.info.get("is_ref", 0) or 0),
            "energy": energy,
            "energy_per_atom": None if energy is None or not count else energy / count,
            "energy_minus_fit": energy_minus_fit,
            "second_difference": second_difference,
            "PG": point_group,
            "lambda1": None if eigenvalues is None else float(eigenvalues[0]),
            "lambda2": None if eigenvalues is None else float(eigenvalues[1]),
            "lambda3": None if eigenvalues is None else float(eigenvalues[2]),
            "shape": shape,
            "S": similarity,
        }
        if compute_s2:
            row["S2"] = similarity_angle
        if include_binding:
            row["binding_energy_per_atom"] = binding_energy
        row["energy_diff_ref_per_atom"] = reference_difference
        rows.append(row)

    return rows


def format_cluster_csv(rows: Sequence[AnalysisRow]) -> str:
    """Format analysis rows as CSV text.

    Parameters
    ----------
    rows : Sequence[dict[str, object]]
        Rows returned by :func:`analyze_clusters`.

    Returns
    -------
    str
        CSV text with a header and one line per row.
    """
    fields = list(BASE_FIELDS)
    if any("S2" in row for row in rows):
        fields.append("S2")
    if any("binding_energy_per_atom" in row for row in rows):
        fields.append("binding_energy_per_atom")
    fields.append("energy_diff_ref_per_atom")

    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=fields,
        extrasaction="ignore",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        formatted: dict[str, str] = {}
        for field in fields:
            value = row.get(field)
            if value is None:
                formatted[field] = ""
            elif field in FLOAT8_FIELDS:
                formatted[field] = f"{float(value):.8f}"
            elif field in FLOAT6_FIELDS:
                formatted[field] = f"{float(value):.6f}"
            else:
                formatted[field] = str(value)
        writer.writerow(formatted)
    return buffer.getvalue()


def write_cluster_csv(
    rows: Sequence[AnalysisRow],
    path: str | Path | None = None,
) -> None:
    """Write analysis rows as CSV.

    Parameters
    ----------
    rows : Sequence[dict[str, object]]
        Rows returned by :func:`analyze_clusters`.
    path : str | pathlib.Path | None, optional
        Output CSV path. When ``None``, write to standard output.
    """
    text = format_cluster_csv(rows)
    if path is None:
        sys.stdout.write(text)
    else:
        Path(path).write_text(text, encoding="utf-8")
