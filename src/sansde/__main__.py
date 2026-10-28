"""Command line entry point for the sansde package.

Usage:
    python -m sansde --opt params.json
    sansde --opt params.json
"""
import argparse
from pathlib import Path

from sansde.main import run


def main(argv: list[str] | None = None) -> None:
    """Parse command line arguments and dispatch the requested task.

    Parameters
    ----------
    argv : list[str] | None, optional
        Argument list. ``None`` uses ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(prog="sansde")
    parser.add_argument(
        "--opt",
        type=Path,
        metavar="PARAMS",
        help="run the optimization with the given params JSON file",
    )
    args = parser.parse_args(argv)
    if args.opt is None:
        parser.error("missing --opt PARAMS")
    run(args.opt)


if __name__ == "__main__":
    main()
