"""Plotting helpers for cluster analysis results."""

from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from sansde.analysis import AnalysisRow, RadialRow

X_STEP = 5.0
Y_STEP = 1.0
SHAPE_YSTEP = 0.2
S_YSTEP = 0.05

COLOR_EDIFF = "#0000ff"
COLOR_D2E = "#ff0000"
COLOR_L1 = "#ff0000"
COLOR_L2 = "#00ff00"
COLOR_L3 = "#0000ff"
COLOR_LAVG = "#800080"
COLOR_S = "#0000ff"
COLOR_S2 = "#ff8000"
COLOR_EREF = "#0000ff"
COLOR_EB = "#0000ff"
REF_BAND = "0.86"
REF_BAND_DARK = "0.72"
RADIAL_Y_STEP = 2.0
RADIAL_TOLERANCE = 0.1
CMAP_CYAN_POS = 1.0 / 6.0
CMAP_YELLOW_POS = 0.5


def _spans(sizes: set[int]) -> list[tuple[int, int]]:
    """Merge sorted integer sizes into contiguous intervals."""
    values = sorted(sizes)
    if not values:
        return []
    spans: list[tuple[int, int]] = []
    low = high = values[0]
    for value in values[1:]:
        if value <= high + 1:
            high = value
        else:
            spans.append((low, high))
            low = high = value
    spans.append((low, high))
    return spans


def _draw_ref_bands(
    ax: plt.Axes,
    sizes: set[int],
    color: str = REF_BAND,
    label: str | None = None,
    zorder: float = 0,
) -> None:
    """Draw shaded reference-affected size ranges."""
    for index, (low, high) in enumerate(_spans(sizes)):
        ax.axvspan(
            low - 0.5,
            high + 0.5,
            color=color,
            lw=0,
            zorder=zorder,
            label=label if index == 0 else None,
        )


def _tick_start(value: float, step: float) -> float:
    """Return the first tick at or below a value."""
    return float(np.floor(value / step) * step)


def _save_figure(fig: plt.Figure, path: str | Path) -> Path:
    """Save a figure and close it."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure_format = "svg" if output.suffix.lower() == ".svg" else None
    fig.savefig(output, format=figure_format)
    plt.close(fig)
    return output


def _ref_sizes(rows: Sequence[AnalysisRow]) -> set[int]:
    """Return atom counts whose structures come from reference data."""
    return {
        int(row["natoms"])
        for row in rows
        if int(row.get("is_ref", 0) or 0)
    }


def plot_energy(rows: Sequence[AnalysisRow], path: str | Path) -> Path | None:
    """Plot ``E - E_fit`` and the second energy difference."""
    xs_energy: list[int] = []
    ys_energy: list[float] = []
    xs_difference: list[int] = []
    ys_difference: list[float] = []
    for row in rows:
        count = int(row["natoms"])
        value = row.get("energy_minus_fit")
        if value is not None:
            xs_energy.append(count)
            ys_energy.append(float(value))
        value = row.get("second_difference")
        if value is not None:
            xs_difference.append(count)
            ys_difference.append(float(value))
    if not xs_energy and not xs_difference:
        return None

    reference_sizes = _ref_sizes(rows)
    fig, ax = plt.subplots(figsize=(10, 6))
    _draw_ref_bands(
        ax,
        {count + offset for count in reference_sizes for offset in (-1, 0, 1)},
        color=REF_BAND,
        label="ref-affected (delta2E: N+-1)",
    )
    _draw_ref_bands(
        ax,
        reference_sizes,
        color=REF_BAND_DARK,
        label="ref-affected (E-E_fit: N)",
        zorder=0.1,
    )
    if xs_energy:
        ax.plot(xs_energy, ys_energy, "o-", color=COLOR_EDIFF, label="E - E_fit")
    if xs_difference:
        ax.plot(xs_difference, ys_difference, "o-", color=COLOR_D2E, label="delta2E")

    values = np.asarray(ys_energy + ys_difference, dtype=float)
    low = np.ceil(values.min() - 1.0)
    high = np.floor(values.max() + 1.0)
    ax.set_ylim(low, high)
    ax.set_yticks(np.arange(low, high + 1e-9, Y_STEP))
    all_x = np.asarray(xs_energy + xs_difference, dtype=float)
    ax.set_xticks(np.arange(_tick_start(all_x.min(), X_STEP), all_x.max() + 1e-9, X_STEP))
    ax.set_xlabel("N")
    ax.set_ylabel("Energy (eV)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return _save_figure(fig, path)


def plot_shape(
    rows: Sequence[AnalysisRow],
    path: str | Path,
    shape_start: float = 6.0,
) -> Path | None:
    """Plot normalized shape eigenvalues."""
    selected = [
        row
        for row in rows
        if row.get("lambda1") is not None and int(row["natoms"]) >= shape_start
    ]
    if not selected:
        return None
    xs = [int(row["natoms"]) for row in selected]
    first = [float(row["lambda1"]) for row in selected]
    second = [float(row["lambda2"]) for row in selected]
    third = [float(row["lambda3"]) for row in selected]
    average = [
        (value_1 + value_2 + value_3) / 3.0
        for value_1, value_2, value_3 in zip(first, second, third)
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    _draw_ref_bands(ax, _ref_sizes(rows), label="ref-affected (N)")
    ax.plot(xs, first, "o-", color=COLOR_L1, label="lambda1")
    ax.plot(xs, second, "o-", color=COLOR_L2, label="lambda2")
    ax.plot(xs, third, "o-", color=COLOR_L3, label="lambda3")
    ax.plot(xs, average, "--", color=COLOR_LAVG, label="lambda_avg")
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.arange(0.0, 1.0 + 1e-9, SHAPE_YSTEP))
    first_tick = np.floor((shape_start - 1.0) / X_STEP) * X_STEP
    ax.set_xticks(np.arange(first_tick, max(xs) + 1e-9, X_STEP))
    ax.set_xlabel("N")
    ax.set_ylabel("lambda' = lambda / N^(2/3)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return _save_figure(fig, path)


def plot_similarity(rows: Sequence[AnalysisRow], path: str | Path) -> Path | None:
    """Plot S(N, N-1) and optional S2(N, N-1)."""
    xs_s: list[int] = []
    ys_s: list[float] = []
    xs_s2: list[int] = []
    ys_s2: list[float] = []
    for row in rows:
        count = int(row["natoms"])
        value = row.get("S")
        if value is not None:
            xs_s.append(count)
            ys_s.append(float(value))
        value = row.get("S2")
        if value is not None:
            xs_s2.append(count)
            ys_s2.append(float(value))
    if not xs_s and not xs_s2:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    _draw_ref_bands(
        ax,
        {count + offset for count in _ref_sizes(rows) for offset in (0, 1)},
        label="ref-affected (S/S2: N, N+1)",
    )
    if xs_s:
        ax.plot(xs_s, ys_s, "o-", color=COLOR_S, label="S(N, N-1) distance")
    if xs_s2:
        ax.plot(xs_s2, ys_s2, "s--", color=COLOR_S2, label="S2(N, N-1) angle")
    minimum = min(ys_s + ys_s2)
    steps = int(np.floor((1.0 - minimum) / S_YSTEP + 1e-9)) + 1
    low = max(0.0, 1.0 - S_YSTEP * steps)
    ax.set_ylim(low, 1.0)
    ax.set_yticks(np.arange(low, 1.0 + 1e-9, S_YSTEP))
    all_x = xs_s + xs_s2
    ax.set_xticks(
        np.arange(_tick_start(min(all_x), X_STEP), max(all_x) + 1e-9, X_STEP)
    )
    ax.set_xlabel("N")
    ax.set_ylabel("Similarity function S / S2" if xs_s2 else "Similarity function S")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return _save_figure(fig, path)


def plot_reference(rows: Sequence[AnalysisRow], path: str | Path) -> Path | None:
    """Plot the per-atom energy difference from reference data."""
    xs: list[int] = []
    ys: list[float] = []
    for row in rows:
        value = row.get("energy_diff_ref_per_atom")
        if value is not None:
            xs.append(int(row["natoms"]))
            ys.append(float(value))
    if not xs:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    _draw_ref_bands(ax, _ref_sizes(rows), label="ref-affected (N)")
    ax.plot(xs, ys, "o-", color=COLOR_EREF, label="[E(N) - E_ref(N)] / N")
    ax.axhline(0.0, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("N")
    ax.set_ylabel("[E - E_ref] / N (eV/atom)")
    ax.set_xticks(np.arange(_tick_start(min(xs), X_STEP), max(xs) + 1e-9, X_STEP))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return _save_figure(fig, path)


def plot_binding(rows: Sequence[AnalysisRow], path: str | Path) -> Path | None:
    """Plot per-atom binding energy."""
    xs: list[int] = []
    ys: list[float] = []
    for row in rows:
        value = row.get("binding_energy_per_atom")
        count = int(row["natoms"])
        if value is not None and count > 1:
            xs.append(count)
            ys.append(float(value))
    if not xs:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    _draw_ref_bands(ax, _ref_sizes(rows), label="ref-affected (N)")
    ax.plot(xs, ys, "o-", color=COLOR_EB, label="E_b (this work)")
    ax.set_xlabel("N")
    ax.set_ylabel("binding energy (eV/atom)")
    ax.set_xticks(np.arange(_tick_start(min(xs), X_STEP), max(xs) + 1e-9, X_STEP))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return _save_figure(fig, path)


def plot_radial(
    rows: Sequence[RadialRow],
    path: str | Path,
    cyan_pos: float = CMAP_CYAN_POS,
    yellow_pos: float = CMAP_YELLOW_POS,
) -> Path | None:
    """Plot radial distances with an overlap-count color scale."""
    counts: list[int] = []
    distances: list[float] = []
    for row in rows:
        count = int(row["natoms"])
        for radius in row["radial_distances"]:
            counts.append(count)
            distances.append(float(radius))
    if not distances:
        return None

    ns = np.asarray(counts, dtype=int)
    rs = np.asarray(distances, dtype=float)
    keys = list(
        zip(
            ns.tolist(),
            np.round(rs / RADIAL_TOLERANCE).astype(int).tolist(),
        )
    )
    overlap = Counter(keys)
    colors = np.asarray([overlap[key] for key in keys], dtype=float)

    cmap = LinearSegmentedColormap.from_list(
        "blue_yellow_red",
        [
            (0.0, "#0000ff"),
            (cyan_pos, "#00ffff"),
            (yellow_pos, "#ffff00"),
            (1.0, "#ff0000"),
        ],
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    _draw_ref_bands(ax, _ref_sizes(rows), label="ref-affected (N)")
    scatter = ax.scatter(
        ns,
        rs,
        c=colors,
        cmap=cmap,
        vmin=1,
        vmax=max(2, float(colors.max())),
        s=18,
        linewidths=0,
    )
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("overlap count")
    ax.set_xlabel("N")
    ax.set_ylabel("Radial Distance (A)")
    xmin, xmax = int(ns.min()), int(ns.max())
    ax.set_xticks(np.arange(xmin - xmin % X_STEP, xmax + 1e-9, X_STEP))
    ax.set_yticks(np.arange(0, rs.max() + RADIAL_Y_STEP, RADIAL_Y_STEP))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save_figure(fig, path)
