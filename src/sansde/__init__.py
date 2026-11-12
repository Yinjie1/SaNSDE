"""SaNSDE package."""

import json
import os
import sys
from pathlib import Path


def _apply_cli_export() -> None:
    """Apply ``EXPORT`` from the run parameter file before the imports below.

    Thread limits such as ``OMP_NUM_THREADS`` are read when numpy and the
    calculator libraries load, so they must be in the environment before the
    eager imports of this package run. Only the ``run`` command is handled;
    other commands and plain library imports are left untouched.
    """
    if len(sys.argv) < 3 or sys.argv[1] != "run":
        return
    for token in sys.argv[2:]:
        if token.startswith("-"):
            continue
        try:
            data = json.loads(Path(token).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if isinstance(data, dict):
            export = data.get("EXPORT")
            if isinstance(export, dict):
                for name, value in export.items():
                    os.environ[str(name)] = str(value)
        return


_apply_cli_export()

from sansde.operators import (
    crossover,
    local_optimize,
    local_transformation,
    mutate,
    random_cluster,
    select,
    sphere_cut_splice,
)
from sansde.pop import Clusters, Population

__version__ = "0.2.16"

__all__ = [
    'Clusters',
    'Population',
    '__version__',
    'crossover',
    'local_optimize',
    'local_transformation',
    'mutate',
    'random_cluster',
    'select',
    'sphere_cut_splice',
]
