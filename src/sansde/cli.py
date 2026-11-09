"""Command line interface for the sansde package.

Usage:
    sansde init                       # write params.json with default values
    sansde run params.json            # run in the background, log to params.log
    sansde run params.json --fg       # run in the foreground
    sansde xyz2vasp params.json       # export VASP tasks from JSON parameters
    sansde check [directory]          # inspect run parameter files and outputs
    sansde analyse -i json_dir        # analyse ASE JSON cluster structures
    sansde radial -i json_dir         # analyse radial layer structure
    sansde cn -i json_dir             # analyse coordination numbers
    python -m sansde run params.json --fg
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from sansde.analysis import (
    analyze_clusters,
    analyze_coordination,
    analyze_radial,
    write_cluster_csv,
    write_coordination_csv,
    write_radial_csv,
)
from sansde.check import run_check
from sansde.io import DEFAULT_RUN_PARAMS, load_clusters, write_vasp_tasks
from sansde.main import main as run_opt
from sansde.plots import (
    plot_binding,
    plot_coordination,
    plot_energy,
    plot_radial,
    plot_reference,
    plot_shape,
    plot_similarity,
)


def main(argv: list[str] | None = None) -> None:
    """Parse command line arguments and dispatch the requested task."""
    parser = argparse.ArgumentParser(prog="sansde")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser(
        "init",
        help="write params.json with default values",
    )

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
        "--plot",
        nargs="?",
        const="svg",
        choices=("svg", "png"),
        default=None,
        metavar="FORMAT",
        help="write SVG plots; pass png to also write PNG",
    )
    analyser.add_argument(
        "--shape-start",
        type=float,
        default=6.0,
        metavar="N",
        help="first atom count shown in the shape plot (default: 6)",
    )

    radial = subparsers.add_parser(
        "radial",
        help="analyse radial layer structure from ASE JSON structures",
    )
    radial.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        metavar="JSON_DIR",
        help="directory containing ASE JSON structures",
    )
    radial.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        metavar="CSV",
        help="CSV output path (default: standard output)",
    )
    radial.add_argument(
        "--k",
        nargs="+",
        type=int,
        default=None,
        metavar="K",
        help="manual layer counts in ascending atom-count order",
    )
    radial.add_argument(
        "--plot",
        nargs="?",
        const="svg",
        choices=("svg", "png"),
        default=None,
        metavar="FORMAT",
        help="write SVG plots; pass png to also write PNG",
    )
    radial.add_argument(
        "--cyan-pos",
        type=float,
        default=1.0 / 6.0,
        metavar="POS",
        help="cyan position in the overlap color map (default: 1/6)",
    )

    coordination = subparsers.add_parser(
        "cn",
        help="analyse coordination numbers from ASE JSON structures",
    )
    coordination.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        metavar="JSON_DIR",
        help="directory containing ASE JSON structures",
    )
    coordination.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        metavar="CSV",
        help="CSV output path (default: standard output)",
    )
    coordination.add_argument(
        "--cutoff",
        type=float,
        default=None,
        metavar="A",
        help="bonding cutoff; defaults to the bulk neighbor mean distance",
    )
    coordination.add_argument(
        "--bulk-cn",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="compute and plot the bulk average coordination number",
    )
    coordination.add_argument(
        "--plot",
        nargs="?",
        const="svg",
        choices=("svg", "png"),
        default=None,
        metavar="FORMAT",
        help="write SVG plots; pass png to also write PNG",
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

    if args.command == "init":
        output = Path("params.json")
        output.write_text(
            json.dumps(DEFAULT_RUN_PARAMS, indent=4) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {output}")
        return

    if args.command == "analyse":
        clusters = load_clusters(args.input)
        rows = analyze_clusters(
            clusters,
            e1=args.e1,
            ref=args.ref,
            compute_s2=args.s2,
        )
        write_cluster_csv(rows, args.output)
        if args.plot:
            input_path = args.input.resolve()
            prefix = input_path.stem if input_path.is_file() else input_path.name
            formats = ("svg",) if args.plot == "svg" else ("svg", "png")
            for suffix in formats:
                plot_energy(rows, f"{prefix}_energy.{suffix}")
                plot_shape(
                    rows,
                    f"{prefix}_shape.{suffix}",
                    shape_start=args.shape_start,
                )
                plot_similarity(rows, f"{prefix}_s.{suffix}")
                plot_reference(rows, f"{prefix}_energy_ref.{suffix}")
                plot_binding(rows, f"{prefix}_eb.{suffix}")
        return

    if args.command == "cn":
        clusters = load_clusters(args.input)
        rows = analyze_coordination(
            clusters,
            cutoff=args.cutoff,
            show_bulk_cn=args.bulk_cn,
        )
        write_coordination_csv(rows, args.output)
        if args.plot:
            input_path = args.input.resolve()
            prefix = input_path.stem if input_path.is_file() else input_path.name
            formats = ("svg",) if args.plot == "svg" else ("svg", "png")
            for suffix in formats:
                plot_coordination(rows, f"{prefix}_cn.{suffix}")
        return

    if args.command == "radial":
        clusters = load_clusters(args.input)
        rows = analyze_radial(clusters, k=args.k)
        write_radial_csv(rows, args.output)
        if args.plot:
            input_path = args.input.resolve()
            prefix = input_path.stem if input_path.is_file() else input_path.name
            formats = ("svg",) if args.plot == "svg" else ("svg", "png")
            for suffix in formats:
                plot_radial(
                    rows,
                    f"{prefix}_radial.{suffix}",
                    cyan_pos=args.cyan_pos,
                )
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
