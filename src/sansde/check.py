"""Checks for sansde run parameter files and working directories."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class _RunConfig:
    """Run parameters needed by the checks."""

    path: Path
    element: str
    min_atom: int
    max_atom: int
    best_dir: str
    better_dir: str
    pop_dir: str
    init_dir: str
    init_pop_dir: str
    rounds: int


def _read_run_configs(directory: Path) -> list[_RunConfig]:
    """Read run parameter JSON files and return the parsed configurations.

    Files that are not run parameter files, cannot be parsed, or contain
    invalid values are reported and skipped.
    """
    configs: list[_RunConfig] = []
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
            rounds = int(data.get("ROUND", 1))
        except (TypeError, ValueError) as e:
            print(f"skip {path.name}: invalid parameter ({e})")
            continue
        if min_atom > max_atom:
            print(f"skip {path.name}: MIN_ATOM > MAX_ATOM")
            continue
        if rounds < 1:
            print(f"skip {path.name}: ROUND must be >= 1")
            continue
        configs.append(
            _RunConfig(
                path=path,
                element=str(data.get("ELEMENT", "Au")),
                min_atom=min_atom,
                max_atom=max_atom,
                best_dir=str(data.get("BEST_DIR", "best")),
                better_dir=str(data.get("BETTER_DIR", "better")),
                pop_dir=str(data.get("POP_DIR", "Populations")),
                init_dir=str(data.get("INIT_DIR", "init")),
                init_pop_dir=str(data.get("INIT_POP_DIR", "")),
                rounds=rounds,
            )
        )
    return configs


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
    configs: list[_RunConfig],
) -> list[tuple[Path, Path, str, int, int]]:
    """Return pairwise overlaps as ``(path_a, path_b, element, start, end)``."""
    overlaps: list[tuple[Path, Path, str, int, int]] = []
    for index, config_a in enumerate(configs):
        for config_b in configs[index + 1:]:
            if config_a.element != config_b.element:
                continue
            start = max(config_a.min_atom, config_b.min_atom)
            end = min(config_a.max_atom, config_b.max_atom)
            if start <= end:
                overlaps.append(
                    (
                        config_a.path,
                        config_b.path,
                        config_a.element,
                        start,
                        end,
                    )
                )
    return overlaps


def _resolve_dir(directory: Path, value: str) -> Path:
    """Resolve a configured directory against the checked directory."""
    path = Path(value)
    return path if path.is_absolute() else directory / path


def _display_path(path: Path, base: Path) -> str:
    """Return a path relative to ``base`` when possible."""
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def _existing_outputs(
    config: _RunConfig,
    directory: Path,
) -> dict[str, list[Path]]:
    """Return existing output files for one configuration, grouped by directory."""
    if config.rounds > 1:
        best_dirs = [directory / f"best{index}" for index in range(1, config.rounds + 1)]
    else:
        best_dirs = [_resolve_dir(directory, config.best_dir)]
    better_dir = _resolve_dir(directory, config.better_dir)
    pop_dir = _resolve_dir(directory, config.pop_dir)

    existing: dict[str, list[Path]] = {}
    for atom in range(config.min_atom, config.max_atom + 1):
        symbol = f"{config.element}{atom}"

        for best_dir in best_dirs:
            best_path = best_dir / f"{symbol}_best.xyz"
            if best_path.is_file():
                label = _display_path(best_dir, directory)
                existing.setdefault(label, []).append(best_path)

        better_path = better_dir / f"{symbol}_better.xyz"
        if better_path.is_file():
            label = _display_path(better_dir, directory)
            existing.setdefault(label, []).append(better_path)

        pop_path = pop_dir / f"{symbol}.json"
        if pop_path.is_file():
            label = _display_path(pop_dir, directory)
            existing.setdefault(label, []).append(pop_path)
        for snapshot in sorted(pop_dir.glob(f"{symbol}_*.json")):
            label = _display_path(pop_dir, directory)
            existing.setdefault(label, []).append(snapshot)

    return existing


def _warn_existing_outputs(config: _RunConfig, directory: Path) -> bool:
    """Warn when output files for a configuration already exist."""
    existing = _existing_outputs(config, directory)
    if not existing:
        return False

    systems = _format_range(config.element, config.min_atom, config.max_atom)
    print(
        f"warning: {config.path.name} ({systems}) has existing output files:"
    )
    for label in sorted(existing):
        files = existing[label]
        examples = ", ".join(
            _display_path(path, directory) for path in files[:3]
        )
        suffix = "" if len(files) <= 3 else f", ... ({len(files)} files)"
        print(f"  {label}/: {examples}{suffix}")
    print("  these files may be modified or overwritten by the run")
    return True


def run_check(directory: str | Path = ".") -> None:
    """Check run parameter files and output directories without modifying files.

    Parameters
    ----------
    directory : str | pathlib.Path, optional
        Directory searched for ``*.json`` run parameter files. Default ``"."``.
    """
    path = Path(directory)
    if not path.is_dir():
        print(f"directory not found: {path}")
        return
    configs = _read_run_configs(path)
    if not configs:
        print(f"no run parameter files found in {path.resolve()}")
        return

    ranges_by_element: dict[str, list[tuple[int, int]]] = {}
    for config in configs:
        ranges_by_element.setdefault(config.element, []).append(
            (config.min_atom, config.max_atom)
        )

    print(f"read {len(configs)} run parameter file(s)")
    parts: list[str] = []
    for element in sorted(ranges_by_element):
        merged = _merge_ranges(ranges_by_element[element])
        parts.extend(
            _format_range(element, start, end)
            for start, end in merged
        )
    print(f"optimization systems: {', '.join(parts)}")

    overlaps = _find_overlaps(configs)
    if overlaps:
        print("overlapping optimization systems:")
        for path_a, path_b, element, start, end in overlaps:
            print(
                f"  {path_a.name} <-> {path_b.name}: "
                f"{_format_range(element, start, end)}"
            )
    else:
        print("no overlapping optimization systems")

    warned = False
    for config in configs:
        warned = _warn_existing_outputs(config, path) or warned
    if not warned:
        print("no existing output files for the target systems")
