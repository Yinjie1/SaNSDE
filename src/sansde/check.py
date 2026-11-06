"""Checks for sansde run parameter files and working directories."""

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from ase.io import read

from sansde.utils import print_banner

_RED = "31"
_GREEN = "32"
_YELLOW = "33"
_BLUE = "34"


def _supports_color() -> bool:
    """Return whether ANSI colors should be written to stdout."""
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    return sys.stdout.isatty()


def _paint(text: str, color: str, enabled: bool) -> str:
    """Wrap text in an ANSI color code when colors are enabled."""
    if not enabled:
        return text
    return f"\033[{color}m{text}\033[0m"


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


def _dir_setting(data: dict[str, object], key: str, default: str) -> str:
    """Return a directory setting, using the default only when it is missing."""
    value = data.get(key, default)
    return default if value is None else str(value)


def _read_run_configs(directory: Path, color: bool) -> list[_RunConfig]:
    """Read run parameter JSON files and return the parsed configurations.

    Files that are not run parameter files, cannot be parsed, or contain
    invalid values are reported and skipped.
    """
    configs: list[_RunConfig] = []
    for path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            print(_paint(f"skip {path.name}: cannot read JSON ({e})", _YELLOW, color))
            continue
        if not isinstance(data, dict):
            print(_paint(f"skip {path.name}: JSON root is not an object", _YELLOW, color))
            continue
        if "MIN_ATOM" not in data or "MAX_ATOM" not in data:
            print(_paint(f"skip {path.name}: not a run parameter file", _YELLOW, color))
            continue
        try:
            min_atom = int(data["MIN_ATOM"])
            max_atom = int(data["MAX_ATOM"])
            rounds = int(data.get("ROUND", 1))
        except (TypeError, ValueError) as e:
            print(_paint(f"skip {path.name}: invalid parameter ({e})", _YELLOW, color))
            continue
        if min_atom > max_atom:
            print(_paint(f"skip {path.name}: MIN_ATOM > MAX_ATOM", _YELLOW, color))
            continue
        if rounds < 1:
            print(_paint(f"skip {path.name}: ROUND must be >= 1", _YELLOW, color))
            continue
        configs.append(
            _RunConfig(
                path=path,
                element=str(data.get("ELEMENT", "Au")),
                min_atom=min_atom,
                max_atom=max_atom,
                best_dir=_dir_setting(data, "BEST_DIR", "best"),
                better_dir=_dir_setting(data, "BETTER_DIR", "better"),
                pop_dir=_dir_setting(data, "POP_DIR", "Populations"),
                init_dir=_dir_setting(data, "INIT_DIR", "init"),
                init_pop_dir=_dir_setting(data, "INIT_POP_DIR", ""),
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


def _warn_existing_outputs(
    config: _RunConfig,
    directory: Path,
    color: bool,
) -> bool:
    """Warn when output files for a configuration already exist."""
    existing = _existing_outputs(config, directory)
    if not existing:
        return False

    systems = _format_range(config.element, config.min_atom, config.max_atom)
    print(
        f"{_paint('warning:', _RED, color)} "
        f"{config.path.name} ({systems}) has existing output files:"
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


def _available_xyz_symbols(directory: Path, color: bool) -> set[str]:
    """Return the symbols of valid single-formula xyz files in a directory."""
    symbols: set[str] = set()
    if not directory.is_dir():
        return symbols
    for path in directory.glob("*.xyz"):
        try:
            frames = read(path, index=":")  # type: ignore
        except Exception as e:  # noqa: BLE001
            print(_paint(f"skip {path.name}: cannot read xyz ({e})", _YELLOW, color))
            continue
        formulas = {frame.get_chemical_formula() for frame in frames}
        if len(formulas) == 1:
            symbols.update(formulas)
    return symbols


def _available_population_symbols(directory: Path, color: bool) -> set[str]:
    """Return the symbols stored in population JSON snapshots in a directory."""
    symbols: set[str] = set()
    if not directory.is_dir():
        return symbols
    for path in directory.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            print(_paint(f"skip {path.name}: cannot read JSON ({e})", _YELLOW, color))
            continue
        if not isinstance(data, dict):
            continue
        symbol = data.get("symbol")
        if isinstance(symbol, str):
            symbols.add(symbol)
    return symbols


def _missing_systems(
    directory: Path,
    element: str,
    min_atom: int,
    max_atom: int,
    suffix: str,
    color: bool,
) -> list[str]:
    """Return target symbols that are not present in ``directory``."""
    if suffix == ".xyz":
        available = _available_xyz_symbols(directory, color)
    else:
        available = _available_population_symbols(directory, color)
    return [
        f"{element}{atom}"
        for atom in range(min_atom, max_atom + 1)
        if f"{element}{atom}" not in available
    ]


def _warn_initial_inputs(
    config: _RunConfig,
    directory: Path,
    color: bool,
) -> None:
    """Warn about missing initial structures or populations."""
    init_dir_value = config.init_dir
    init_pop_dir_value = config.init_pop_dir
    has_init_dir = bool(init_dir_value.strip())
    has_init_pop = bool(init_pop_dir_value.strip())
    systems = _format_range(config.element, config.min_atom, config.max_atom)

    if not has_init_dir and not has_init_pop:
        print(
            f"{config.path.name} ({systems}) has no initial structures configured"
        )
        print("  the optimization will start from random clusters")
        return

    if has_init_pop:
        init_path = _resolve_dir(directory, init_pop_dir_value)
        missing = _missing_systems(
            init_path,
            config.element,
            config.min_atom,
            config.max_atom,
            ".json",
            color,
        )
        if missing:
            label = _display_path(init_path, directory)
            print(
                f"{_paint('warning:', _YELLOW, color)} "
                f"{config.path.name} ({systems}) is missing initial "
                f"populations in {label}:"
            )
            print(f"  missing: {', '.join(missing)}")
        return

    init_path = _resolve_dir(directory, init_dir_value)
    missing = _missing_systems(
        init_path,
        config.element,
        config.min_atom,
        config.max_atom,
        ".xyz",
        color,
    )
    if missing:
        label = _display_path(init_path, directory)
        print(
            f"{_paint('warning:', _YELLOW, color)} "
            f"{config.path.name} ({systems}) is missing initial "
            f"structures in {label}:"
        )
        print(f"  missing: {', '.join(missing)}")


def run_check(directory: str | Path = ".") -> None:
    """Check run parameter files and output directories without modifying files.

    The parameter JSON files are read from ``directory``. Output and initial
    structure directories are resolved against the current working directory,
    because ``sansde run`` executes there even when the JSON lives elsewhere.

    Parameters
    ----------
    directory : str | pathlib.Path, optional
        Directory containing the ``*.json`` run parameter files. Default ``"."``.
    """
    print_banner()
    params_dir = Path(directory)
    if not params_dir.is_dir():
        print(f"directory not found: {params_dir}")
        return
    color = _supports_color()
    configs = _read_run_configs(params_dir, color)
    if not configs:
        print(f"no run parameter files found in {params_dir.resolve()}")
        return

    work_dir = Path.cwd()
    if params_dir.resolve() != work_dir:
        print(f"parameter directory: {params_dir.resolve()}")
        print(f"working directory: {work_dir}")

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
    print(f"{_paint('optimization systems:', _BLUE, color)} {', '.join(parts)}")

    overlaps = _find_overlaps(configs)
    if overlaps:
        print(
            f"{_paint('warning:', _RED, color)} "
            "overlapping optimization systems exist"
        )
        for path_a, path_b, element, start, end in overlaps:
            print(
                f"  {path_a.name} <-> {path_b.name}: "
                f"{_format_range(element, start, end)}"
            )
    else:
        print(_paint("no overlapping optimization systems", _GREEN, color))

    warned = False
    for config in configs:
        warned = _warn_existing_outputs(config, work_dir, color) or warned
    if not warned:
        print(
            _paint(
                "no existing output files for the target systems",
                _GREEN,
                color,
            )
        )

    for config in configs:
        _warn_initial_inputs(config, work_dir, color)
