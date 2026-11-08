"""Command line interface for the sansde package.

Usage:
    sansde run params.json            # run in the background, log to params.log
    sansde run params.json --fg       # run in the foreground
    sansde xyz2vasp params.json       # export VASP tasks from JSON parameters
    sansde check [directory]          # inspect run parameter files and outputs
    sansde analyse -i json_dir        # analyse ASE JSON cluster structures
    python -m sansde run params.json --fg
"""
import argparse
import subprocess
import sys
from pathlib import Path

from sansde.analysis import analyze_clusters, write_cluster_csv
from sansde.check import run_check
from sansde.io import load_clusters, write_vasp_tasks
from sansde.main import main as run_opt
from sansde.plots import (
    plot_binding,
    plot_energy,
    plot_reference,
    plot_shape,
    plot_similarity,
)


def main(argv: list[str] | None = None) -> None:
    """Parse command line arguments and dispatch the requested task."""
    parser = argparse.ArgumentParser(prog="sansde")
    subparsers = parser.add_subparsers(dest="command")

    analyser = subparsers.add_parser(
        "analyse",
        help="run cluster analysis on ASE JSON structures",
    )
    analyser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        metavar="JSON_DIR",
        help="directory containing ASE JSON structures",
    )
    analyser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        metavar="CSV",
        help="CSV output path (default: standard output)",
    )
    analyser.add_argument(
        "--s2",
        action="store_true",
        help="also compute the angle similarity S2",
    )
    analyser.add_argument(
        "--ref",
        type=Path,
        default=None,
        metavar="PATH",
        help="reference ASE JSON directory or CSV file",
    )
    analyser.add_argument(
        "--e1",
        type=float,
        default=None,
        metavar="E1",
        help="isolated atom energy; defaults to the N=1 structure",
    )
    analyser.add_argument(
        "--no-plot",
        action="store_true",
        help="do not write plot files",
    )
    analyser.add_argument(
        "--plot",
        type=Path,
        default=Path("cluster_energy.svg"),
        metavar="SVG",
        help="E - E_fit and delta2E plot (default: cluster_energy.svg)",
    )
    analyser.add_argument(
        "--plot-shape",
        type=Path,
        default=Path("cluster_shape.svg"),
        metavar="SVG",
        help="shape eigenvalue plot (default: cluster_shape.svg)",
    )
    analyser.add_argument(
        "--plot-s",
        type=Path,
        default=Path("cluster_s.svg"),
        metavar="SVG",
        help="similarity plot (default: cluster_s.svg)",
    )
    analyser.add_argument(
        "--plot-ref",
        type=Path,
        default=Path("cluster_energy_ref.svg"),
        metavar="SVG",
        help="reference energy difference plot (default: cluster_energy_ref.svg)",
    )
    analyser.add_argument(
        "--plot-eb",
        type=Path,
        default=Path("cluster_eb.svg"),
        metavar="SVG",
        help="binding energy plot (default: cluster_eb.svg)",
    )
    analyser.add_argument(
        "--shape-start",
        type=float,
        default=6.0,
        metavar="N",
        help="first atom count shown in the shape plot (default: 6)",
    )

    checker = subparsers.add_parser(
        "check",
        help="check run parameter files and output directories in a directory",
    )
    checker.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=Path("."),
        help="directory containing parameter JSON files (default: current directory)",
    )
    checker.add_argument(
        "--no-banner",
        action="store_true",
        help="do not print the SANSDE banner",
    )

    export = subparsers.add_parser(
        "xyz2vasp",
        help="export xyz directories to VASP task directories using JSON parameters",
    )
    export.add_argument(
        "json",
        type=Path,
        metavar="PARAMS",
        help="JSON parameter file, e.g. sansde xyz2vasp params.json",
    )

    runner = subparsers.add_parser(
        "run",
        help="run the optimization with a params JSON file",
    )
    runner.add_argument(
        "params",
        type=Path,
        metavar="PARAMS",
        help="JSON parameter file, e.g. sansde run params.json",
    )
    runner.add_argument(
        "--fg",
        "--foreground",
        dest="foreground",
        action="store_true",
        help="run in the foreground (default: run in the background)",
    )
    args = parser.parse_args(argv)

    if args.command == "analyse":
        clusters = load_clusters(args.input)
        rows = analyze_clusters(
            clusters,
            e1=args.e1,
            ref=args.ref,
            compute_s2=args.s2,
        )
        write_cluster_csv(rows, args.output)
        if not args.no_plot:
            plots = [
                plot_energy(rows, args.plot),
                plot_shape(rows, args.plot_shape, shape_start=args.shape_start),
                plot_similarity(rows, args.plot_s),
                plot_reference(rows, args.plot_ref),
                plot_binding(rows, args.plot_eb),
            ]
            for path in plots:
                if path is not None:
                    print(f"[plot] {path}", file=sys.stderr)
        return

    if args.command == "check":
        run_check(args.directory, show_banner=not args.no_banner)
        return

    if args.command == "xyz2vasp":
        write_vasp_tasks(args.json)
        return

    if args.command != "run":
        parser.error("missing command")

    if args.foreground:
        # Explicit foreground: run in the current process, output stays on the terminal.
        run_opt(args.params)
        return

    # Default background mode: detach a child process from the current session.
    # The child carries --fg so it does not enter the background branch again.
    log_path = args.params.with_suffix(".log")
    command = [
        sys.executable,
        "-u",  # unbuffered log, convenient for live tail
        "-m",
        "sansde",
        "run",
        str(args.params),
        "--fg",
    ]
    with log_path.open("ab") as log_file:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,  # do not depend on terminal input
            stdout=log_file,
            stderr=subprocess.STDOUT,  # write stderr to the same log
            # start_new_session=True is equivalent to setsid: the child keeps
            # running after the terminal exits.
            start_new_session=True,
        )
    # Record background pids, one per line, for later batch management.
    pid_path = Path("pid")
    with pid_path.open("a", encoding="utf-8") as pid_file:
        pid_file.write(f"{process.pid}\n")
    print(f"started background process pid={process.pid}, log={log_path}")
