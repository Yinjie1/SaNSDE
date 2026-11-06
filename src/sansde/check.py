"""Checks for sansde run parameter files and working directories."""

import json
from pathlib import Path


def _read_run_ranges(directory: Path) -> list[tuple[Path, str, int, int]]:
    """Read run parameter JSON files and return ``(path, element, min, max)``.

    Files that are not run parameter files, cannot be parsed, or contain an
    invalid size range are reported and skipped.
    """
    ranges: list[tuple[Path, str, int, int]] = []
    for path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            print(f"skip {path.name}: cannot read JSON ({e})")
            continue
        if not isinstance(data, dict):
            print(f"skip {path.name}: JSON root is not an object")
            continue
        if "MIN_ATOM" not in data or "MAX_ATOM" not in data:
            print(f"skip {path.name}: not a run parameter file")
            continue
        try:
            min_atom = int(data["MIN_ATOM"])
            max_atom = int(data["MAX_ATOM"])
        except (TypeError, ValueError) as e:
            print(f"skip {path.name}: invalid atom range ({e})")
            continue
        if min_atom > max_atom:
            print(f"skip {path.name}: MIN_ATOM > MAX_ATOM")
            continue
        element = str(data.get("ELEMENT", "Au"))
        ranges.append((path, element, min_atom, max_atom))
    return ranges


def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping or adjacent inclusive integer ranges."""
    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _format_range(element: str, start: int, end: int) -> str:
    """Format an element and inclusive atom-count range."""
    if start == end:
        return f"{element}{start}"
    return f"{element}{start}-{end}"


def _find_overlaps(
    ranges: list[tuple[Path, str, int, int]],
) -> list[tuple[Path, Path, str, int, int]]:
    """Return pairwise overlapping systems as ``(path_a, path_b, element, start, end)``."""
    overlaps: list[tuple[Path, Path, str, int, int]] = []
    for index, (path_a, element_a, min_a, max_a) in enumerate(ranges):
        for path_b, element_b, min_b, max_b in ranges[index + 1:]:
            if element_a != element_b:
                continue
            start = max(min_a, min_b)
            end = min(max_a, max_b)
            if start <= end:
                overlaps.append((path_a, path_b, element_a, start, end))
    return overlaps


def run_check(directory: str | Path = ".") -> None:
    """Check run parameter files in a directory and print the target systems.

    Parameters
    ----------
    directory : str | pathlib.Path, optional
        Directory searched for ``*.json`` run parameter files. Default ``"."``.
    """
    path = Path(directory)
    if not path.is_dir():
        print(f"directory not found: {path}")
        return
    ranges = _read_run_ranges(path)
    if not ranges:
        print(f"no run parameter files found in {path.resolve()}")
        return

    ranges_by_element: dict[str, list[tuple[int, int]]] = {}
    for _, element, min_atom, max_atom in ranges:
        ranges_by_element.setdefault(element, []).append((min_atom, max_atom))

    print(f"read {len(ranges)} run parameter file(s)")
    parts: list[str] = []
    for element in sorted(ranges_by_element):
        merged = _merge_ranges(ranges_by_element[element])
        parts.extend(
            _format_range(element, start, end)
            for start, end in merged
        )
    print(f"optimization systems: {', '.join(parts)}")

    overlaps = _find_overlaps(ranges)
    if overlaps:
        print("overlapping optimization systems:")
        for path_a, path_b, element, start, end in overlaps:
            print(
                f"  {path_a.name} <-> {path_b.name}: "
                f"{_format_range(element, start, end)}"
            )
    else:
        print("no overlapping optimization systems")
