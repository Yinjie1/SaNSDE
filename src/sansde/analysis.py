"""Cluster energy, shape, and similarity analysis from ASE JSON files."""

import csv
import io
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import TypeAlias

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.neighborlist import neighbor_list

from sansde.io import load_ref_energies
from sansde.operators import compute_point_group, similarity_function
from sansde.pop import Clusters

SHAPE_RATIO = 0.8
UL = 1.0

AnalysisRow: TypeAlias = dict[str, object]
CoordinationRow: TypeAlias = dict[str, object]

__all__ = [
    "AnalysisRow",
    "CoordinationRow",
    "RadialRow",
    "analyze_clusters",
    "analyze_coordination",
    "analyze_radial",
    "angle_similarity_add_atom",
    "average_bond_length",
    "bulk_cn",
    "bulk_cutoff",
    "coordination_numbers",
    "eval_fit",
    "fit_energy",
    "format_cluster_csv",
    "format_coordination_csv",
    "format_radial_csv",
    "radial_distances",
    "shape_indicator",
    "shape_label",
    "similarity_add_atom",
    "split_layers",
    "write_cluster_csv",
    "write_coordination_csv",
    "write_radial_csv",
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
    atoms_n: Atoms | None,
    atoms_ref: Atoms | None,
) -> float | None:
    """Compute the distance similarity ``S(N, N-1)``.

    Each atom is removed from the N-atom cluster in turn. The maximum
    :func:`sansde.operators.similarity_function` value against the reference
    cluster is returned.

    Parameters
    ----------
    atoms_n : Atoms | None
        N-atom cluster.
    atoms_ref : Atoms | None
        Reference cluster with ``N - 1`` atoms.

    Returns
    -------
    float | None
        Similarity value, or ``None`` when the input sizes are inconsistent.
    """
    if atoms_n is None or atoms_ref is None:
        return None
    if len(atoms_ref) != len(atoms_n) - 1 or len(atoms_n) < 3:
        return None

    best: float | None = None
    for index in range(len(atoms_n)):
        trial = atoms_n.copy()
        del trial[index]
        value = similarity_function(trial, atoms_ref)
        if value <= 0.0:
            continue
        best = value if best is None else max(best, value)
    return best


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
        similarity = similarity_add_atom(cluster, previous_cluster)
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


RadialRow: TypeAlias = dict[str, object]

RADIAL_FIELDS = [
    "system",
    "symbol",
    "natoms",
    "is_ref",
    "PG",
    "n_layers",
    "layer_sizes",
    "layer_pg",
    "radial_distances",
]

COORDINATION_FIELDS = [
    "system",
    "symbol",
    "natoms",
    "is_ref",
    "cn_atoms",
    "cn_ratios",
    "cn_avg",
    "bond_length_avg",
]


def radial_distances(coords: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted radial distances and the matching atom indices.

    Parameters
    ----------
    coords : numpy.ndarray | None
        Atomic coordinates with shape ``(N, 3)``.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Ascending radial distances and the original atom indices.
    """
    if coords is None or len(coords) == 0:
        return np.array([]), np.array([], dtype=int)
    positions = np.asarray(coords, dtype=float)
    radii = np.linalg.norm(positions - positions.mean(axis=0), axis=1)
    order = np.argsort(radii, kind="stable")
    return radii[order], order


def _kmeans1d_dp(
    values: np.ndarray,
    kmax: int,
) -> tuple[list[float], np.ndarray]:
    """Return exact one-dimensional K-means inertias and cut points."""
    values = np.asarray(values, dtype=float)
    count = len(values)
    kmax = max(1, min(kmax, count))
    prefix = np.concatenate(([0.0], np.cumsum(values)))
    prefix_sq = np.concatenate(([0.0], np.cumsum(values * values)))

    infinity = float("inf")
    dp = np.full((kmax + 1, count), infinity)
    cut = np.full((kmax + 1, count), -1, dtype=int)
    for index in range(count):
        size = index + 1
        total = prefix[index + 1]
        total_sq = prefix_sq[index + 1]
        dp[1][index] = total_sq - total * total / size

    for layers in range(2, kmax + 1):
        for index in range(layers - 1, count):
            candidates = np.arange(layers - 2, index)
            starts = candidates + 1
            sizes = (index - starts + 1).astype(float)
            totals = prefix[index + 1] - prefix[starts]
            totals_sq = prefix_sq[index + 1] - prefix_sq[starts]
            costs = totals_sq - totals * totals / sizes
            values = dp[layers - 1, candidates] + costs
            best = int(values.argmin())
            dp[layers][index] = values[best]
            cut[layers][index] = candidates[best]

    inertias = [float(dp[layers][count - 1]) for layers in range(1, kmax + 1)]
    return inertias, cut


def split_layers(
    radii_sorted: np.ndarray,
    k: int | None,
) -> list[list[int]] | None:
    """Split sorted radial distances into layers with a manual layer count.

    Returns ``None`` when ``k`` is not provided.
    """
    if k is None:
        return None
    radii = np.asarray(radii_sorted, dtype=float)
    count = len(radii)
    if count == 0:
        return []
    layer_count = max(1, min(int(k), count))
    if layer_count <= 1:
        return [list(range(count))]

    _inertias, cut = _kmeans1d_dp(radii, layer_count)
    bounds: list[tuple[int, int]] = []
    index = count - 1
    for layers in range(layer_count, 1, -1):
        previous = cut[layers][index]
        bounds.append((previous + 1, index))
        index = previous
    bounds.append((0, index))
    bounds.reverse()
    return [list(range(low, high + 1)) for low, high in bounds]


def analyze_radial(
    clusters: Clusters,
    *,
    k: Sequence[int] | None = None,
    pg_tolerance: float = 0.3,
) -> list[RadialRow]:
    """Analyze radial layer structure for every atom count.

    Parameters
    ----------
    clusters : Clusters
        Clusters to analyze.
    k : Sequence[int] | None, optional
        Manual layer counts in ascending atom-count order. The length must
        match the number of systems. When omitted, layer fields are left
        empty.
    pg_tolerance : float, optional
        Point-group tolerance. Default 0.3.

    Returns
    -------
    list[dict[str, object]]
        One row per atom count.

    Raises
    ------
    ValueError
        If manual layer counts do not match the number of systems.
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

    systems = sorted(lowest_by_count)
    if not systems:
        return []
    manual: dict[int, int] = {}
    if k is not None:
        if len(k) != len(systems):
            raise ValueError(
                f"manual k has {len(k)} values for {len(systems)} systems"
            )
        manual = dict(zip(systems, k))

    rows: list[RadialRow] = []
    for count in systems:
        cluster = lowest_by_count[count]
        radii, order = radial_distances(cluster.get_positions())
        layers = split_layers(radii, manual.get(count))
        sizes: list[int] | None = None
        layer_groups: list[str | None] | None = None
        if layers is not None:
            sizes = []
            layer_groups = []
            for positions in layers:
                atom_indices = order[np.asarray(positions, dtype=int)].tolist()
                layer_atoms = cluster[atom_indices]
                sizes.append(len(atom_indices))
                layer_groups.append(
                    None
                    if len(atom_indices) < 2
                    else compute_point_group(layer_atoms, tolerance=pg_tolerance)
                )
        point_group = (
            None
            if count < 2
            else compute_point_group(cluster, tolerance=pg_tolerance)
        )
        rows.append(
            {
                "system": str(
                    cluster.info.get("system", cluster.get_chemical_formula())
                ),
                "symbol": cluster.get_chemical_formula(),
                "natoms": count,
                "is_ref": int(cluster.info.get("is_ref", 0) or 0),
                "PG": point_group,
                "n_layers": None if layers is None else len(layers),
                "layer_sizes": sizes,
                "layer_pg": layer_groups,
                "radial_distances": [float(value) for value in radii],
            }
        )
    return rows


def format_radial_csv(rows: Sequence[RadialRow]) -> str:
    """Format radial analysis rows as CSV text."""
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=RADIAL_FIELDS,
        extrasaction="ignore",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        layer_sizes = row["layer_sizes"]
        layer_groups = row["layer_pg"]
        distances = row["radial_distances"]
        writer.writerow(
            {
                "system": str(row["system"]),
                "symbol": str(row["symbol"]),
                "natoms": str(row["natoms"]),
                "is_ref": str(row["is_ref"]),
                "PG": "null" if row["PG"] is None else str(row["PG"]),
                "n_layers": (
                    "" if row["n_layers"] is None else str(row["n_layers"])
                ),
                "layer_sizes": (
                    ""
                    if layer_sizes is None
                    else "("
                    + " ".join(str(int(value)) for value in layer_sizes)
                    + ")"
                ),
                "layer_pg": (
                    ""
                    if layer_groups is None
                    else "("
                    + " ".join(
                        "null" if value is None else str(value)
                        for value in layer_groups
                    )
                    + ")"
                ),
                "radial_distances": "("
                + " ".join(f"{float(value):.3f}" for value in distances)
                + ")",
            }
        )
    return buffer.getvalue()


def write_radial_csv(
    rows: Sequence[RadialRow],
    path: str | Path | None = None,
) -> None:
    """Write radial analysis rows as CSV to a file or stdout."""
    text = format_radial_csv(rows)
    if path is None:
        sys.stdout.write(text)
    else:
        Path(path).write_text(text, encoding="utf-8")


def bulk_cutoff(
    symbol: str,
    structure: str | None = None,
    a: float | None = None,
) -> tuple[float | None, float | None, float | None]:
    """Return the bulk first/second neighbor mean distance.

    Parameters
    ----------
    symbol : str
        Element symbol.
    structure : str | None, optional
        Crystal structure passed to :func:`ase.build.bulk`.
    a : float | None, optional
        Lattice constant passed to :func:`ase.build.bulk`.

    Returns
    -------
    tuple[float | None, float | None, float | None]
        Bonding cutoff ``(d1 + d2) / 2``, first distance, and second distance.
    """
    kwargs: dict[str, object] = {}
    if structure is not None:
        kwargs["crystalstructure"] = structure
    if a is not None:
        kwargs["a"] = a
    atoms = bulk(symbol, **kwargs).repeat((3, 3, 3))
    distances = atoms.get_all_distances(mic=True)
    upper = np.triu_indices(len(atoms), k=1)
    values = np.unique(np.round(distances[upper], 6))
    values = values[values > 1e-6]
    if len(values) == 0:
        return None, None, None
    if len(values) < 2:
        return float(values[0]), None, None
    first = float(values[0])
    second = float(values[1])
    return 0.5 * (first + second), first, second


def bulk_cn(
    symbol: str,
    cutoff: float,
    structure: str | None = None,
    a: float | None = None,
) -> float:
    """Return the average bulk coordination number at a cutoff."""
    kwargs: dict[str, object] = {}
    if structure is not None:
        kwargs["crystalstructure"] = structure
    if a is not None:
        kwargs["a"] = a
    atoms = bulk(symbol, **kwargs).repeat((3, 3, 3))
    indices, _neighbors = neighbor_list("ij", atoms, cutoff)
    return float(np.bincount(indices, minlength=len(atoms)).mean())


def coordination_numbers(atoms: Atoms, cutoff: float) -> np.ndarray:
    """Return the coordination number of every atom."""
    count = len(atoms)
    if count < 2:
        return np.zeros(count, dtype=int)
    indices, _neighbors = neighbor_list("ij", atoms, cutoff)
    return np.bincount(indices, minlength=count).astype(int)


def average_bond_length(atoms: Atoms, cutoff: float) -> float | None:
    """Return the mean distance over all bonded pairs."""
    if len(atoms) < 2:
        return None
    indices, neighbors = neighbor_list("ij", atoms, cutoff)
    if len(indices) == 0:
        return None
    positions = atoms.get_positions()
    distances = np.linalg.norm(
        positions[indices] - positions[neighbors],
        axis=1,
    )
    return float(distances.mean())


def analyze_coordination(
    clusters: Clusters,
    *,
    cutoff: float | None = None,
    show_bulk_cn: bool = False,
) -> list[CoordinationRow]:
    """Analyze coordination numbers and average bond lengths.

    Parameters
    ----------
    clusters : Clusters
        Clusters to analyze.
    cutoff : float | None, optional
        Bonding cutoff. When omitted, the bulk first/second neighbor mean
        distance of the most common element is used.
    show_bulk_cn : bool, optional
        Whether to compute the bulk average coordination number. Default
        False.

    Returns
    -------
    list[dict[str, object]]
        One row per cluster.

    Raises
    ------
    ValueError
        If the bulk cutoff cannot be determined.
    """
    entries = sorted(
        clusters,
        key=lambda cluster: (
            len(cluster),
            str(cluster.info.get("system", cluster.get_chemical_formula())),
        ),
    )
    if not entries:
        return []

    symbols = [
        symbol
        for cluster in entries
        for symbol in cluster.get_chemical_symbols()
    ]
    element = Counter(symbols).most_common(1)[0][0]
    if cutoff is None:
        cutoff_value, _first, _second = bulk_cutoff(element)
        if cutoff_value is None:
            raise ValueError(f"cannot determine bulk cutoff for {element}")
    else:
        cutoff_value = float(cutoff)

    bulk_value = bulk_cn(element, cutoff_value) if show_bulk_cn else None
    numbers: list[np.ndarray] = []
    maximum = 0
    for cluster in entries:
        values = coordination_numbers(cluster, cutoff_value)
        numbers.append(values)
        if values.size:
            maximum = max(maximum, int(values.max()))

    rows: list[CoordinationRow] = []
    for cluster, values in zip(entries, numbers):
        count = len(cluster)
        counts = [0] * maximum
        for value in values:
            if 1 <= value <= maximum:
                counts[value - 1] += 1
        ratios = [
            100.0 * value / count if count else 0.0
            for value in counts
        ]
        average = float(np.mean(values)) if count else 0.0
        bond_length = average_bond_length(cluster, cutoff_value)
        rows.append(
            {
                "system": str(
                    cluster.info.get("system", cluster.get_chemical_formula())
                ),
                "symbol": cluster.get_chemical_formula(),
                "natoms": count,
                "is_ref": int(cluster.info.get("is_ref", 0) or 0),
                "cn_atoms": counts,
                "cn_ratios": ratios,
                "cn_avg": average,
                "bond_length_avg": bond_length,
                "cutoff": cutoff_value,
                "bulk_cn": bulk_value,
            }
        )
    return rows


def format_coordination_csv(rows: Sequence[CoordinationRow]) -> str:
    """Format coordination analysis rows as CSV text."""
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=COORDINATION_FIELDS,
        extrasaction="ignore",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        bond_length = row["bond_length_avg"]
        writer.writerow(
            {
                "system": str(row["system"]),
                "symbol": str(row["symbol"]),
                "natoms": str(row["natoms"]),
                "is_ref": str(row["is_ref"]),
                "cn_atoms": "("
                + " ".join(str(int(value)) for value in row["cn_atoms"])
                + ")",
                "cn_ratios": "("
                + " ".join(f"{float(value):g}" for value in row["cn_ratios"])
                + ")",
                "cn_avg": f"{float(row['cn_avg']):.4f}",
                "bond_length_avg": (
                    "" if bond_length is None else f"{float(bond_length):.4f}"
                ),
            }
        )
    return buffer.getvalue()


def write_coordination_csv(
    rows: Sequence[CoordinationRow],
    path: str | Path | None = None,
) -> None:
    """Write coordination analysis rows as CSV to a file or stdout."""
    text = format_coordination_csv(rows)
    if path is None:
        sys.stdout.write(text)
    else:
        Path(path).write_text(text, encoding="utf-8")


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
